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


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package cloud runtime assets for Eagle Eye.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out", default="dist/eagle_eye_processed_bundle.tar.gz")
    parser.add_argument(
        "--events_out",
        default="",
        help="Optional output path for a separate events bundle containing events.parquet.",
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


if __name__ == "__main__":
    main()
