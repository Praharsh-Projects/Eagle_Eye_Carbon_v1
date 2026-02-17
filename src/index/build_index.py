"""Build persistent Chroma indexes for traffic rows and optional PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
from openai import OpenAI

from src.ingest.pdf_ingest import discover_pdfs, ingest_pdfs
from src.ingest.traffic_ingest import TrafficIngestResult, ingest_traffic_csv
from src.utils.config import load_config
from src.utils.runtime import import_chromadb, require_openai_api_key


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_texts(
    client: OpenAI, model: str, texts: Sequence[str], batch_size: int
) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for batch in _batched(texts, batch_size):
        response = client.embeddings.create(model=model, input=list(batch))
        ordered = sorted(response.data, key=lambda d: d.index)
        embeddings.extend([item.embedding for item in ordered])
    return embeddings


def upsert_collection(
    collection: Any,
    ids: Sequence[str],
    texts: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    write_batch_size: int = 500,
) -> None:
    for idx_batch, txt_batch, meta_batch, emb_batch in zip(
        _batched(ids, write_batch_size),
        _batched(texts, write_batch_size),
        _batched(metadatas, write_batch_size),
        _batched(embeddings, write_batch_size),
    ):
        collection.upsert(
            ids=list(idx_batch),
            documents=list(txt_batch),
            metadatas=list(meta_batch),
            embeddings=list(emb_batch),
        )


def write_traffic_metadata_index(
    persist_dir: Path, traffic_result: TrafficIngestResult
) -> Path:
    rows: List[Dict[str, Any]] = []
    for stable_id, text, metadata in zip(
        traffic_result.ids, traffic_result.texts, traffic_result.metadatas
    ):
        row = {"stable_id": stable_id, "serialized_text": text}
        row.update(metadata)
        rows.append(row)
    df = pd.DataFrame(rows)
    output_path = persist_dir / "traffic_metadata_index.csv"
    df.to_csv(output_path, index=False)
    return output_path


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build persistent Chroma indexes.")
    parser.add_argument("--traffic_csv", required=False, help="Primary traffic CSV input")
    parser.add_argument(
        "--traffic_csvs",
        nargs="*",
        default=None,
        help="Additional traffic CSV paths (space-separated)",
    )
    parser.add_argument("--persist_dir", default=None, help="Chroma persistence directory")
    parser.add_argument("--pdf_dir", default=None, help="Optional PDF directory")
    parser.add_argument(
        "--pdf_paths",
        nargs="*",
        default=None,
        help="Optional explicit PDF file paths (space-separated)",
    )
    parser.add_argument(
        "--doc_urls",
        nargs="*",
        default=None,
        help="Optional public documentation URLs to ingest",
    )
    parser.add_argument("--limit_rows", type=int, default=None, help="Subset mode for fast demo")
    parser.add_argument("--config", default="config/config.yaml", help="Config path")
    parser.add_argument("--rebuild", action="store_true", help="Recreate collections from scratch")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    config = load_config(args.config)
    chromadb = import_chromadb()

    embedding_model = config["models"]["embedding_model"]
    batch_size = int(config["index"].get("batch_size", 128))
    traffic_collection_name = config["index"]["traffic_collection"]
    docs_collection_name = config["index"]["docs_collection"]
    rebuild = bool(args.rebuild or config["index"].get("rebuild", False))

    persist_dir = Path(args.persist_dir or config["paths"]["persist_dir"])
    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    openai_client = OpenAI(api_key=require_openai_api_key())

    if rebuild:
        for name in (traffic_collection_name, docs_collection_name):
            try:
                chroma_client.delete_collection(name=name)
            except Exception:
                pass

    traffic_collection = chroma_client.get_or_create_collection(
        name=traffic_collection_name, metadata={"hnsw:space": "cosine"}
    )
    docs_collection = chroma_client.get_or_create_collection(
        name=docs_collection_name, metadata={"hnsw:space": "cosine"}
    )

    traffic_csv_paths: List[str] = []
    if args.traffic_csv:
        traffic_csv_paths.append(args.traffic_csv)
    if args.traffic_csvs:
        traffic_csv_paths.extend(args.traffic_csvs)
    if not traffic_csv_paths:
        raise ValueError("Provide at least one traffic CSV via --traffic_csv and/or --traffic_csvs")

    if rebuild or traffic_collection.count() == 0:
        merged_ids: List[str] = []
        merged_texts: List[str] = []
        merged_metadatas: List[Dict[str, Any]] = []
        merged_rows: List[Dict[str, Any]] = []
        global_seen_ids: Dict[str, int] = {}
        skipped_total = 0

        for csv_path in traffic_csv_paths:
            current = ingest_traffic_csv(csv_path, limit_rows=args.limit_rows)
            for local_id, text, metadata, row in zip(
                current.ids, current.texts, current.metadatas, current.rows
            ):
                if local_id in global_seen_ids:
                    global_seen_ids[local_id] += 1
                    unique_id = f"{local_id}__dup{global_seen_ids[local_id]}"
                else:
                    global_seen_ids[local_id] = 0
                    unique_id = local_id
                metadata["stable_id"] = unique_id
                merged_ids.append(unique_id)
                merged_texts.append(text)
                merged_metadatas.append(metadata)
                merged_rows.append(row)
            skipped_total += current.skipped_rows

        traffic_result = TrafficIngestResult(
            ids=merged_ids,
            texts=merged_texts,
            metadatas=merged_metadatas,
            rows=merged_rows,
            skipped_rows=skipped_total,
        )
        traffic_embeddings = embed_texts(
            openai_client,
            model=embedding_model,
            texts=traffic_result.texts,
            batch_size=batch_size,
        )
        upsert_collection(
            traffic_collection,
            traffic_result.ids,
            traffic_result.texts,
            traffic_result.metadatas,
            traffic_embeddings,
        )
        metadata_path = write_traffic_metadata_index(persist_dir, traffic_result)
        print(
            f"Indexed {len(traffic_result.ids)} traffic rows into '{traffic_collection_name}'. "
            f"Metadata index: {metadata_path}"
        )
        print(f"Traffic sources indexed: {', '.join(traffic_csv_paths)}")
        if traffic_result.skipped_rows:
            print(
                f"Skipped {traffic_result.skipped_rows} rows missing required fields for their schema."
            )
    else:
        print(
            f"Reusing existing traffic collection '{traffic_collection_name}' "
            f"with {traffic_collection.count()} rows."
        )

    pdf_paths: List[Path] = []
    if args.pdf_dir:
        pdf_paths.extend(discover_pdfs(args.pdf_dir))
    if args.pdf_paths:
        for path in args.pdf_paths:
            p = Path(path)
            if p.exists() and p.suffix.lower() == ".pdf":
                pdf_paths.append(p)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_pdf_paths: List[Path] = []
    for p in pdf_paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_pdf_paths.append(p)

    doc_urls = args.doc_urls or []
    if unique_pdf_paths or doc_urls:
        pdf_cfg = config.get("pdf", {})
        all_doc_ids: List[str] = []
        all_doc_texts: List[str] = []
        all_doc_metadatas: List[Dict[str, Any]] = []

        if unique_pdf_paths:
            pdf_result = ingest_pdfs(
                unique_pdf_paths,
                chunk_size=int(pdf_cfg.get("chunk_size", 1200)),
                chunk_overlap=int(pdf_cfg.get("chunk_overlap", 200)),
            )
            all_doc_ids.extend(pdf_result.ids)
            all_doc_texts.extend(pdf_result.texts)
            all_doc_metadatas.extend(pdf_result.metadatas)
            print(
                f"Prepared {len(pdf_result.ids)} chunks from {len(unique_pdf_paths)} PDF files."
            )

        if doc_urls:
            from src.ingest.web_ingest import ingest_web_urls

            web_result = ingest_web_urls(
                doc_urls,
                chunk_size=int(pdf_cfg.get("chunk_size", 1200)),
                chunk_overlap=int(pdf_cfg.get("chunk_overlap", 200)),
            )
            all_doc_ids.extend(web_result.ids)
            all_doc_texts.extend(web_result.texts)
            all_doc_metadatas.extend(web_result.metadatas)
            print(
                f"Prepared {len(web_result.ids)} chunks from {len(doc_urls)} web URLs."
            )

        if all_doc_ids:
            doc_embeddings = embed_texts(
                openai_client,
                model=embedding_model,
                texts=all_doc_texts,
                batch_size=batch_size,
            )
            upsert_collection(
                docs_collection,
                all_doc_ids,
                all_doc_texts,
                all_doc_metadatas,
                doc_embeddings,
            )
            print(
                f"Indexed {len(all_doc_ids)} docs chunks (PDF + web) into "
                f"'{docs_collection_name}'."
            )
        else:
            print("No valid PDF/web docs produced chunks; docs collection unchanged.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
