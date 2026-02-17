"""Traffic CSV ingestion: AIS PRJ912 + port-call PRJ896 into serialized docs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.serialization import serialize_traffic_row


@dataclass
class TrafficIngestResult:
    ids: List[str]
    texts: List[str]
    metadatas: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    skipped_rows: int


def load_traffic_df(csv_path: str | Path, limit_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if limit_rows is not None and limit_rows > 0:
        df = df.head(limit_rows)
    return df


def ingest_traffic_csv(
    csv_path: str | Path, limit_rows: Optional[int] = None
) -> TrafficIngestResult:
    path = Path(csv_path)
    df = load_traffic_df(path, limit_rows=limit_rows)

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    seen_ids: Dict[str, int] = {}
    skipped_rows = 0
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        serialized = serialize_traffic_row(row_dict, source_file=path.name)
        if serialized is None:
            skipped_rows += 1
            continue
        text, metadata, base_id = serialized
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        stable_id = f"{base_id}_{count}" if count else base_id
        metadata["row_index"] = int(idx)
        metadata["stable_id"] = stable_id

        ids.append(stable_id)
        texts.append(text)
        metadatas.append(metadata)
        rows.append(row_dict)

    return TrafficIngestResult(
        ids=ids,
        texts=texts,
        metadatas=metadatas,
        rows=rows,
        skipped_rows=skipped_rows,
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest traffic CSV (AIS positions or port-calls) into serialized events."
    )
    parser.add_argument("--traffic_csv", required=True, help="Path to maritime traffic CSV")
    parser.add_argument("--limit_rows", type=int, default=None, help="Subset rows for quick demo")
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    result = ingest_traffic_csv(args.traffic_csv, limit_rows=args.limit_rows)
    print(
        f"Loaded {len(result.ids)} traffic rows from {args.traffic_csv} "
        f"(skipped {result.skipped_rows} rows missing required fields)"
    )
    if result.ids:
        print(f"Sample id: {result.ids[0]}")
        print(f"Sample text: {result.texts[0][:200]}...")


if __name__ == "__main__":
    main()
