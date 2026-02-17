"""Train ETA prediction model (regression)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.predict.train_destination import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _time_preserving_sample(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    idx = np.unique(idx)
    return df.iloc[idx].copy()


def _build_regression_pipeline(
    model_kind: str, num_features: List[str], cat_features: List[str]
) -> Pipeline:
    if model_kind == "random_forest":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_features),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("oh", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_features,
                ),
            ],
            remainder="drop",
        )
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=22,
            n_jobs=-1,
            random_state=42,
        )
        return Pipeline([("prep", preprocessor), ("model", model)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_features,
            ),
        ],
        remainder="drop",
    )

    if model_kind == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=10,
            max_iter=180,
            l2_regularization=0.1,
            random_state=42,
        )
    return Pipeline([("prep", preprocessor), ("model", model)])


def _choose_model_kind() -> str:
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except Exception:
        return "histgb"


def train_eta_model(
    training_rows_path: Path,
    model_dir: Path,
    test_fraction: float = 0.2,
    max_eta_minutes: float = 14 * 24 * 60,
    max_train_rows: Optional[int] = 120_000,
    max_test_rows: Optional[int] = 30_000,
) -> Dict[str, object]:
    if not training_rows_path.exists():
        raise RuntimeError(f"Training rows file not found: {training_rows_path}")
    df = pd.read_parquet(training_rows_path)
    if "timestamp" not in df.columns:
        raise RuntimeError("training_rows.parquet must contain 'timestamp' column.")
    if "eta_minutes_label" not in df.columns:
        return {"skipped": True, "reason": "eta_minutes_label column missing"}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "eta_minutes_label"])
    df["eta_minutes_label"] = pd.to_numeric(df["eta_minutes_label"], errors="coerce")
    df = df.dropna(subset=["eta_minutes_label"])
    df = df[(df["eta_minutes_label"] > 0) & (df["eta_minutes_label"] <= max_eta_minutes)]
    if len(df) < 1000:
        return {
            "skipped": True,
            "reason": f"insufficient ETA labels ({len(df)} rows after filtering)",
        }

    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - test_fraction))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    if train.empty or test.empty:
        return {"skipped": True, "reason": "train/test split empty"}

    train = _time_preserving_sample(train, max_rows=max_train_rows)
    test = _time_preserving_sample(test, max_rows=max_test_rows)

    num_features = [c for c in NUMERIC_FEATURES if c in train.columns]
    cat_features = [c for c in CATEGORICAL_FEATURES if c in train.columns]
    if not num_features and not cat_features:
        return {"skipped": True, "reason": "no ETA feature columns available"}

    features = num_features + cat_features
    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train = train["eta_minutes_label"].values
    y_test = test["eta_minutes_label"].values

    model_kind = _choose_model_kind()
    pipeline = _build_regression_pipeline(model_kind, num_features, cat_features)
    print(
        "Training ETA model:",
        f"kind={model_kind}",
        f"train_rows={len(X_train)}",
        f"test_rows={len(X_test)}",
    )
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    medae = float(median_absolute_error(y_test, pred))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "eta_model.pkl"
    schema_path = model_dir / "eta_feature_schema.json"
    metrics_path = model_dir / "eta_metrics.json"

    joblib.dump(pipeline, model_path)

    schema = {
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "target": "eta_minutes_label",
        "model_kind": model_kind,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "max_train_rows": int(max_train_rows) if max_train_rows else None,
        "max_test_rows": int(max_test_rows) if max_test_rows else None,
    }
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    metrics = {
        "skipped": False,
        "mae_minutes": mae,
        "rmse_minutes": rmse,
        "median_absolute_error_minutes": medae,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model_kind": model_kind,
        "max_train_rows": int(max_train_rows) if max_train_rows else None,
        "max_test_rows": int(max_test_rows) if max_test_rows else None,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ETA regression model.")
    parser.add_argument(
        "--training_rows",
        default="data/processed/training_rows.parquet",
    )
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--max_eta_minutes", type=float, default=14 * 24 * 60)
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=120000,
        help="Optional cap for training rows (time-preserving sampling).",
    )
    parser.add_argument(
        "--max_test_rows",
        type=int,
        default=30000,
        help="Optional cap for test rows (time-preserving sampling).",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    metrics = train_eta_model(
        training_rows_path=Path(args.training_rows),
        model_dir=Path(args.model_dir),
        test_fraction=args.test_fraction,
        max_eta_minutes=args.max_eta_minutes,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    if metrics.get("skipped"):
        print(f"ETA training skipped: {metrics['reason']}")
        return
    print(
        "ETA model trained. "
        f"MAE={metrics['mae_minutes']:.2f}m RMSE={metrics['rmse_minutes']:.2f}m"
    )


if __name__ == "__main__":
    main()
