"""Train destination prediction model (classification)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


NUMERIC_FEATURES: List[str] = [
    "latitude",
    "longitude",
    "speed",
    "course",
    "heading",
    "draught",
    "delta_time_s",
    "delta_distance_km",
    "speed_estimated_kn",
    "acceleration_kn_per_h",
    "course_change_deg",
    "course_change_rate_deg_per_h",
    "speed_roll_mean_3",
    "speed_roll_std_3",
    "speed_roll_mean_5",
    "speed_roll_std_5",
    "speed_roll_mean_10",
    "speed_roll_std_10",
    "speed_est_roll_mean_5",
    "speed_est_roll_mean_10",
    "course_change_roll_mean_5",
    "course_change_roll_mean_10",
    "hour_of_day",
    "day_of_week",
]

CATEGORICAL_FEATURES: List[str] = [
    "vessel_type_norm",
    "flag_norm",
    "nav_status_norm",
]


def _load_training_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Training rows file not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise RuntimeError("training_rows.parquet must contain 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    return df


def _select_available_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if not num and not cat:
        raise RuntimeError("No expected feature columns found in training_rows.parquet")
    return num, cat


def _time_preserving_sample(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    # Preserve temporal coverage by taking evenly spaced indices across the split.
    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    idx = np.unique(idx)
    return df.iloc[idx].copy()


def _choose_model_kind() -> str:
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except Exception:
        pass
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except Exception:
        pass
    return "histgb"


def _build_pipeline(model_kind: str, num_features: List[str], cat_features: List[str]) -> Pipeline:
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
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
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

    if model_kind == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    elif model_kind == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=10,
            max_iter=180,
            l2_regularization=0.1,
            random_state=42,
        )
    return Pipeline([("prep", preprocessor), ("model", model)])


def _top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def train_destination_model(
    training_rows_path: Path,
    model_dir: Path,
    min_samples_per_class: int = 200,
    test_fraction: float = 0.2,
    max_classes: Optional[int] = 80,
    max_train_rows: Optional[int] = 120_000,
    max_test_rows: Optional[int] = 30_000,
    report_top_classes: int = 25,
) -> Dict[str, object]:
    df = _load_training_rows(training_rows_path)
    df = df[df["destination_norm"].notna()]
    df = df[~df["destination_norm"].astype(str).isin(["UNKNOWN", "unknown", ""])]

    if df.empty:
        raise RuntimeError("No rows with valid destination_norm found for training.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - test_fraction))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    if train.empty or test.empty:
        raise RuntimeError("Time split produced empty train/test. Increase dataset size.")

    class_counts = train["destination_norm"].value_counts()
    threshold_candidates = sorted(
        {max(1, int(min_samples_per_class)), 100, 50, 20, 5, 1},
        reverse=True,
    )
    selected_threshold: Optional[int] = None
    selected_classes: Optional[pd.Index] = None
    selected_train: Optional[pd.DataFrame] = None
    selected_test: Optional[pd.DataFrame] = None
    for threshold in threshold_candidates:
        keep_classes = class_counts[class_counts >= threshold].index
        if max_classes and len(keep_classes) > max_classes:
            keep_classes = class_counts.loc[list(keep_classes)].head(max_classes).index
        cand_train = train[train["destination_norm"].isin(keep_classes)].copy()
        cand_test = test[test["destination_norm"].isin(keep_classes)].copy()
        if len(keep_classes) >= 2 and not cand_train.empty and not cand_test.empty:
            selected_threshold = threshold
            selected_classes = keep_classes
            selected_train = cand_train
            selected_test = cand_test
            break

    if selected_threshold is None or selected_classes is None:
        raise RuntimeError(
            "No viable destination classes for training after threshold relaxation. "
            "Increase data volume or reduce filtering."
        )

    if selected_threshold < min_samples_per_class:
        print(
            "Relaxed min_samples_per_class due dataset size:",
            f"requested={min_samples_per_class}",
            f"used={selected_threshold}",
        )

    train = selected_train
    test = selected_test

    train = _time_preserving_sample(train, max_rows=max_train_rows)
    test = _time_preserving_sample(test, max_rows=max_test_rows)

    num_features, cat_features = _select_available_features(train)
    features = num_features + cat_features

    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train_raw = train["destination_norm"].astype(str).values
    y_test_raw = test["destination_norm"].astype(str).values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    # Inverse-frequency sample weights for class imbalance.
    class_freq = pd.Series(y_train).value_counts()
    sample_weight = pd.Series(y_train).map(lambda c: 1.0 / class_freq[c]).values

    model_kind = _choose_model_kind()
    pipeline = _build_pipeline(model_kind, num_features, cat_features)
    print(
        "Training destination model:",
        f"kind={model_kind}",
        f"train_rows={len(X_train)}",
        f"test_rows={len(X_test)}",
        f"classes={len(label_encoder.classes_)}",
    )
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)

    proba = pipeline.predict_proba(X_test)
    y_pred = np.argmax(proba, axis=1)
    top1 = float(accuracy_score(y_test, y_pred))
    top5 = _top_k_accuracy(y_test, proba, k=min(5, proba.shape[1]))

    test_class_counts = pd.Series(y_test_raw).value_counts()
    report_labels = test_class_counts.head(max(1, report_top_classes)).index.tolist()
    label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    report_ids = [label_to_id[label] for label in report_labels if label in label_to_id]
    report = classification_report(
        y_test,
        y_pred,
        labels=report_ids if report_ids else None,
        target_names=report_labels if report_ids else None,
        output_dict=True,
        zero_division=0,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "destination_model.pkl"
    encoder_path = model_dir / "destination_label_encoder.pkl"
    schema_path = model_dir / "destination_feature_schema.json"
    metrics_path = model_dir / "destination_metrics.json"

    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, encoder_path)

    schema = {
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "target": "destination_norm",
        "model_kind": model_kind,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "min_samples_per_class": int(min_samples_per_class),
        "min_samples_per_class_used": int(selected_threshold),
        "max_classes": int(max_classes) if max_classes else None,
        "max_train_rows": int(max_train_rows) if max_train_rows else None,
        "max_test_rows": int(max_test_rows) if max_test_rows else None,
    }
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    metrics = {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "num_classes": int(len(label_encoder.classes_)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model_kind": model_kind,
        "per_class_report": report,
        "report_top_classes": int(report_top_classes),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train destination prediction model.")
    parser.add_argument(
        "--training_rows",
        default="data/processed/training_rows.parquet",
        help="Path to training_rows.parquet",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=80,
        help="Keep only top-N most frequent destination classes after min-sample filtering.",
    )
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
    parser.add_argument(
        "--report_top_classes",
        type=int,
        default=25,
        help="Number of most frequent test classes to include in the saved class report.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    metrics = train_destination_model(
        training_rows_path=Path(args.training_rows),
        model_dir=Path(args.model_dir),
        min_samples_per_class=args.min_samples_per_class,
        test_fraction=args.test_fraction,
        max_classes=args.max_classes,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        report_top_classes=args.report_top_classes,
    )
    print(
        "Destination model trained. "
        f"top1={metrics['top1_accuracy']:.4f} top5={metrics['top5_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
