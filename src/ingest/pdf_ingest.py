"""PDF ingestion: extract text, chunk, and attach metadata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pypdf import PdfReader


@dataclass
class PDFIngestResult:
    ids: List[str]
    texts: List[str]
    metadatas: List[Dict[str, Any]]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    stride = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += stride
    return chunks


def ingest_pdfs(
    pdf_paths: Iterable[str | Path],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> PDFIngestResult:
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        if not path.exists() or path.suffix.lower() != ".pdf":
            continue

        reader = PdfReader(str(path))
        for page_idx, page in enumerate(reader.pages):
            raw_text = (page.extract_text() or "").strip()
            if not raw_text:
                continue
            page_chunks = _chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk_idx, chunk in enumerate(page_chunks):
                stable_id = f"{path.stem}_p{page_idx + 1}_c{chunk_idx}"
                metadata: Dict[str, Any] = {
                    "source_file": path.name,
                    "page": page_idx + 1,
                    "chunk_index": chunk_idx,
                    "stable_id": stable_id,
                    "timestamp_date": "unknown",
                }
                ids.append(stable_id)
                texts.append(chunk)
                metadatas.append(metadata)

    return PDFIngestResult(ids=ids, texts=texts, metadatas=metadatas)


def discover_pdfs(pdf_dir: str | Path) -> List[Path]:
    path = Path(pdf_dir)
    if not path.exists():
        return []
    return sorted(path.glob("*.pdf"))


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDFs into text chunks.")
    parser.add_argument("--pdf_dir", required=True, help="Directory with PDF files")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    pdfs = discover_pdfs(args.pdf_dir)
    result = ingest_pdfs(
        pdfs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    print(f"Found {len(pdfs)} PDFs, created {len(result.ids)} chunks")
    if result.ids:
        print(f"Sample id: {result.ids[0]}")


if __name__ == "__main__":
    main()
