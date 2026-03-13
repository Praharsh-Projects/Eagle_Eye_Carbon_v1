"""Package the minimal app runtime data bundle for cloud deployment."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


APP_PROCESSED_FILES = [
    "arrivals_daily.parquet",
    "arrivals_hourly.parquet",
    "congestion_daily.parquet",
    "dwell_time.parquet",
    "occupancy_hourly.parquet",
    "port_catalog.parquet",
    "kpi_capabilities.json",
    "forecast_backtest.json",
]

EVENTS_FILE = "events.parquet"
CHROMA_REQUIRED_FILES = [
    "chroma.sqlite3",
    "traffic_metadata_index.csv",
]


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package cloud runtime assets for Eagle Eye.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out", default="dist/eagle_eye_processed_bundle.tar.gz")
    parser.add_argument(
        "--events_out",
        default="",
        help="Optional output path for a separate events bundle containing events.parquet.",
    )
    parser.add_argument(
        "--chroma_dir",
        default="",
        help="Optional Chroma directory to package for full local retrieval parity.",
    )
    parser.add_argument(
        "--chroma_out",
        default="",
        help="Optional output path for a Chroma bundle archive.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    processed_dir = Path(args.processed_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [name for name in APP_PROCESSED_FILES if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required processed files: " + ", ".join(missing)
        )

    with tarfile.open(out_path, "w:gz") as tf:
        for rel_name in APP_PROCESSED_FILES:
            src = processed_dir / rel_name
            tf.add(src, arcname=rel_name)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Created {out_path} ({size_mb:.1f} MB)")

    if args.events_out:
        events_src = processed_dir / EVENTS_FILE
        if not events_src.exists():
            raise FileNotFoundError(f"Missing required events file: {events_src}")
        events_out = Path(args.events_out)
        events_out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(events_out, "w:gz") as tf:
            tf.add(events_src, arcname=EVENTS_FILE)
        events_size_mb = events_out.stat().st_size / (1024 * 1024)
        print(f"Created {events_out} ({events_size_mb:.1f} MB)")

    if args.chroma_out:
        chroma_dir = Path(args.chroma_dir or "data/chroma")
        missing_chroma = [name for name in CHROMA_REQUIRED_FILES if not (chroma_dir / name).exists()]
        if missing_chroma:
            raise FileNotFoundError(
                "Missing required Chroma files: " + ", ".join(missing_chroma)
            )
        chroma_out = Path(args.chroma_out)
        chroma_out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(chroma_out, "w:gz") as tf:
            for child in chroma_dir.iterdir():
                tf.add(child, arcname=child.name)
        chroma_size_mb = chroma_out.stat().st_size / (1024 * 1024)
        print(f"Created {chroma_out} ({chroma_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
