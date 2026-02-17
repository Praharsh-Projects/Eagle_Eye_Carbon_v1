"""Train anomaly model and provide rule-based anomaly flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANOMALY_FEATURES: List[str] = [
    "speed",
    "speed_estimated_kn",
    "acceleration_kn_per_h",
    "course_change_deg",
    "course_change_rate_deg_per_h",
    "delta_distance_km",
    "delta_time_s",
    "speed_roll_std_3",
    "speed_roll_std_5",
    "speed_roll_std_10",
]


def _build_pipeline() -> Pipeline:
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def train_anomaly_model(
    training_rows_path: Path,
    model_dir: Path,
    max_rows: Optional[int] = 120_000,
) -> Dict[str, object]:
    if not training_rows_path.exists():
        raise RuntimeError(f"Training rows file not found: {training_rows_path}")
    df = pd.read_parquet(training_rows_path)
    features = [c for c in ANOMALY_FEATURES if c in df.columns]
    if not features:
        return {"skipped": True, "reason": "no anomaly feature columns available"}
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all", subset=features)
    if len(df) < 500:
        return {"skipped": True, "reason": f"insufficient rows ({len(df)})"}
    if max_rows and len(df) > max_rows:
        idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
        idx = np.unique(idx)
        df = df.iloc[idx].copy()

    pipe = _build_pipeline()
    X = df[features].copy()
    print(f"Training anomaly model: rows={len(X)} features={len(features)}")
    pipe.fit(X)

    score = pipe.named_steps["model"].score_samples(
        pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X))
    )
    score = pd.Series(score)
    score_quantiles = {
        "q01": float(score.quantile(0.01)),
        "q05": float(score.quantile(0.05)),
        "q50": float(score.quantile(0.50)),
    }

    thresholds = {
        "speed_spike_acc_kn_per_h": float(df["acceleration_kn_per_h"].abs().quantile(0.99))
        if "acceleration_kn_per_h" in df.columns
        else 25.0,
        "course_jitter_deg": float(df["course_change_deg"].quantile(0.95))
        if "course_change_deg" in df.columns
        else 35.0,
        "position_jump_km": 80.0,
        "max_jump_window_minutes": 30.0,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "anomaly_model.pkl"
    schema_path = model_dir / "anomaly_feature_schema.json"
    metrics_path = model_dir / "anomaly_metrics.json"
    joblib.dump(pipe, model_path)

    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features": features,
                "thresholds": thresholds,
                "score_quantiles": score_quantiles,
            },
            f,
            indent=2,
        )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "skipped": False,
                "rows": int(len(df)),
                "score_quantiles": score_quantiles,
                "max_rows": int(max_rows) if max_rows else None,
            },
            f,
            indent=2,
        )
    return {"skipped": False, "rows": int(len(df)), "score_quantiles": score_quantiles}


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train anomaly model.")
    parser.add_argument(
        "--training_rows",
        default="data/processed/training_rows.parquet",
    )
    parser.add_argument("--model_dir", default="models")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=120000,
        help="Optional cap for anomaly training rows (time-preserving sampling).",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    metrics = train_anomaly_model(
        training_rows_path=Path(args.training_rows),
        model_dir=Path(args.model_dir),
        max_rows=args.max_rows,
    )
    if metrics.get("skipped"):
        print(f"Anomaly training skipped: {metrics['reason']}")
    else:
        print(f"Anomaly model trained on {metrics['rows']} rows.")


if __name__ == "__main__":
    main()
