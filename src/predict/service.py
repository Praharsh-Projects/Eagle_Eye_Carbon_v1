"""Prediction service API: destination, ETA, anomaly, and RAG-backed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.predict.data_prep import build_ais_feature_rows_from_raw_df
from src.rag.retriever import QueryFilters, RAGRetriever
from src.utils.serialization import compact_traffic_evidence


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isnan(out):
            return None
        return out
    except Exception:
        return None


class PredictionService:
    def __init__(
        self,
        model_dir: str | Path = "models",
        processed_dir: str | Path = "data/processed",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.processed_dir = Path(processed_dir)
        self.events_path = self.processed_dir / "events.parquet"

        self._events_df: Optional[pd.DataFrame] = None
        self._dest_model = None
        self._dest_encoder = None
        self._dest_schema = None
        self._eta_model = None
        self._eta_schema = None
        self._eta_metrics = None
        self._anomaly_model = None
        self._anomaly_schema = None
        self._dest_aliases = None

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_events_df(self) -> pd.DataFrame:
        if self._events_df is None:
            if not self.events_path.exists():
                raise RuntimeError(
                    f"Processed events file missing: {self.events_path}. "
                    "Run data prep first."
                )
            self._events_df = pd.read_parquet(self.events_path)
        return self._events_df

    def _load_destination_artifacts(self) -> None:
        if self._dest_model is not None:
            return
        model_path = self.model_dir / "destination_model.pkl"
        enc_path = self.model_dir / "destination_label_encoder.pkl"
        schema_path = self.model_dir / "destination_feature_schema.json"
        if not model_path.exists() or not enc_path.exists() or not schema_path.exists():
            raise RuntimeError(
                "Destination model artifacts are missing. Run train_destination.py first."
            )
        self._dest_model = joblib.load(model_path)
        self._dest_encoder = joblib.load(enc_path)
        self._dest_schema = self._load_json(schema_path)

    def _load_eta_artifacts(self) -> bool:
        if self._eta_model is not None:
            return True
        model_path = self.model_dir / "eta_model.pkl"
        schema_path = self.model_dir / "eta_feature_schema.json"
        metrics_path = self.model_dir / "eta_metrics.json"
        if not model_path.exists() or not schema_path.exists():
            return False
        self._eta_model = joblib.load(model_path)
        self._eta_schema = self._load_json(schema_path)
        self._eta_metrics = self._load_json(metrics_path)
        return True

    def _load_anomaly_artifacts(self) -> bool:
        if self._anomaly_model is not None:
            return True
        model_path = self.model_dir / "anomaly_model.pkl"
        schema_path = self.model_dir / "anomaly_feature_schema.json"
        if not model_path.exists() or not schema_path.exists():
            return False
        self._anomaly_model = joblib.load(model_path)
        self._anomaly_schema = self._load_json(schema_path)
        return True

    def _load_destination_aliases(self) -> Dict[str, str]:
        if self._dest_aliases is not None:
            return self._dest_aliases
        alias_path = self.processed_dir / "destination_aliases.json"
        self._dest_aliases = self._load_json(alias_path)
        return self._dest_aliases

    def get_recent_points(
        self,
        mmsi: str,
        last_n: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self._load_events_df()
        work = df[df.get("event_kind", "ais_position") == "ais_position"].copy()
        work = work[work["mmsi"].astype(str) == str(mmsi)]
        if date_from:
            work = work[work["timestamp_date"].astype(str) >= date_from]
        if date_to:
            work = work[work["timestamp_date"].astype(str) <= date_to]
        if "timestamp_full" in work.columns:
            work["timestamp"] = pd.to_datetime(work["timestamp_full"], errors="coerce", utc=True)
            work = work.sort_values("timestamp")
        return work.tail(last_n).copy()

    def _build_feature_rows(self, recent_points_df: pd.DataFrame) -> pd.DataFrame:
        if recent_points_df.empty:
            return recent_points_df

        if {"TimePosition", "Latitude", "Longitude"}.issubset(recent_points_df.columns):
            return build_ais_feature_rows_from_raw_df(
                recent_points_df.copy(),
                source_file="runtime_input",
                destination_aliases=self._load_destination_aliases(),
            )

        # Already engineered rows from events.parquet path.
        rows = recent_points_df.copy()
        if "timestamp" not in rows.columns and "timestamp_full" in rows.columns:
            rows["timestamp"] = pd.to_datetime(rows["timestamp_full"], errors="coerce", utc=True)
        rows = rows.sort_values("timestamp").reset_index(drop=True)
        return rows

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        return out[columns]

    def predict_destination(
        self,
        mmsi: str,
        recent_points_df: Optional[pd.DataFrame] = None,
        last_n: int = 10,
        min_points: int = 5,
        top_k: int = 5,
        retriever: Optional[RAGRetriever] = None,
    ) -> Dict[str, Any]:
        self._load_destination_artifacts()
        if recent_points_df is None:
            recent_points_df = self.get_recent_points(mmsi=mmsi, last_n=last_n)
        if len(recent_points_df) < min_points:
            return {
                "status": "not_enough_data",
                "message": f"Need at least {min_points} recent AIS points.",
            }

        feature_rows = self._build_feature_rows(recent_points_df)
        if feature_rows.empty:
            return {"status": "not_enough_data", "message": "No usable feature rows generated."}

        all_features = (
            self._dest_schema.get("numeric_features", [])
            + self._dest_schema.get("categorical_features", [])
        )
        X_latest = self._ensure_columns(feature_rows.tail(1), all_features)
        proba = self._dest_model.predict_proba(X_latest)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        top_predictions = [
            {
                "destination": str(self._dest_encoder.inverse_transform([idx])[0]),
                "probability": float(proba[idx]),
            }
            for idx in top_idx
        ]

        similar_examples: List[Dict[str, Any]] = []
        if retriever and top_predictions:
            latest = feature_rows.tail(1).iloc[0]
            lat = _safe_float(latest.get("latitude"))
            lon = _safe_float(latest.get("longitude"))
            vessel_type = latest.get("vessel_type_norm")
            filters = QueryFilters(
                destination=top_predictions[0]["destination"],
                vessel_type=str(vessel_type) if vessel_type and vessel_type != "unknown" else None,
                lat_min=(lat - 0.3) if lat is not None else None,
                lat_max=(lat + 0.3) if lat is not None else None,
                lon_min=(lon - 0.3) if lon is not None else None,
                lon_max=(lon + 0.3) if lon is not None else None,
            )
            retrieval = retriever.query_traffic(
                question=f"Historical routes to {top_predictions[0]['destination']}",
                filters=filters,
                top_k=5,
            )
            for item in retrieval.evidence:
                similar_examples.append(
                    {
                        "id": item.id,
                        "distance": item.distance,
                        "evidence": compact_traffic_evidence(item.metadata, item.text),
                    }
                )

        return {
            "status": "ok",
            "mmsi": mmsi,
            "top_k": top_predictions,
            "features_used": all_features,
            "similar_examples": similar_examples,
        }

    def predict_eta(
        self,
        mmsi: str,
        recent_points_df: Optional[pd.DataFrame] = None,
        last_n: int = 10,
        min_points: int = 5,
    ) -> Dict[str, Any]:
        if not self._load_eta_artifacts():
            return {
                "status": "skipped",
                "message": "ETA model artifacts not found. Train eta model first.",
            }
        if recent_points_df is None:
            recent_points_df = self.get_recent_points(mmsi=mmsi, last_n=last_n)
        if len(recent_points_df) < min_points:
            return {
                "status": "not_enough_data",
                "message": f"Need at least {min_points} recent AIS points.",
            }

        feature_rows = self._build_feature_rows(recent_points_df)
        all_features = (
            self._eta_schema.get("numeric_features", [])
            + self._eta_schema.get("categorical_features", [])
        )
        X_latest = self._ensure_columns(feature_rows.tail(1), all_features)
        pred_minutes = float(max(0.0, self._eta_model.predict(X_latest)[0]))

        latest_ts = pd.to_datetime(feature_rows.tail(1)["timestamp"].iloc[0], utc=True, errors="coerce")
        eta_pred_ts = latest_ts + pd.Timedelta(minutes=pred_minutes) if pd.notna(latest_ts) else pd.NaT

        mae = float((self._eta_metrics or {}).get("mae_minutes", 0.0))
        confidence = max(0.0, min(1.0, 1.0 - (mae / max(pred_minutes, 60.0))))

        return {
            "status": "ok",
            "predicted_eta_minutes": pred_minutes,
            "predicted_eta_timestamp": eta_pred_ts.isoformat() if pd.notna(eta_pred_ts) else None,
            "confidence": confidence,
        }

    def score_anomaly(
        self,
        mmsi: str,
        recent_points_df: Optional[pd.DataFrame] = None,
        last_n: int = 10,
        min_points: int = 5,
    ) -> Dict[str, Any]:
        if not self._load_anomaly_artifacts():
            return {
                "status": "skipped",
                "message": "Anomaly model artifacts not found. Train anomaly model first.",
            }
        if recent_points_df is None:
            recent_points_df = self.get_recent_points(mmsi=mmsi, last_n=last_n)
        if len(recent_points_df) < min_points:
            return {
                "status": "not_enough_data",
                "message": f"Need at least {min_points} recent AIS points.",
            }

        rows = self._build_feature_rows(recent_points_df)
        features = self._anomaly_schema.get("features", [])
        X = self._ensure_columns(rows, features)

        transformed = self._anomaly_model[:-1].transform(X)
        score_samples = self._anomaly_model.named_steps["model"].score_samples(transformed)
        latest_score = float(score_samples[-1])

        q01 = float(self._anomaly_schema.get("score_quantiles", {}).get("q01", np.min(score_samples)))
        q50 = float(self._anomaly_schema.get("score_quantiles", {}).get("q50", np.max(score_samples)))
        if latest_score <= q01:
            anomaly_score = 1.0
        elif latest_score >= q50:
            anomaly_score = 0.0
        else:
            anomaly_score = float((q50 - latest_score) / max(q50 - q01, 1e-6))

        thresholds = self._anomaly_schema.get("thresholds", {})
        speed_spike_thr = float(thresholds.get("speed_spike_acc_kn_per_h", 25.0))
        jitter_thr = float(thresholds.get("course_jitter_deg", 35.0))
        jump_thr = float(thresholds.get("position_jump_km", 80.0))
        jump_window_min = float(thresholds.get("max_jump_window_minutes", 30.0))

        flags: List[str] = []
        if "acceleration_kn_per_h" in rows.columns:
            if rows["acceleration_kn_per_h"].abs().max(skipna=True) > speed_spike_thr:
                flags.append("speed_spike")
        if "course_change_deg" in rows.columns:
            if rows["course_change_deg"].std(skipna=True) > jitter_thr:
                flags.append("course_jitter")
        if {"delta_distance_km", "delta_time_s"}.issubset(rows.columns):
            jump_mask = (
                rows["delta_distance_km"].fillna(0) > jump_thr
            ) & (
                rows["delta_time_s"].fillna(1e9) <= jump_window_min * 60.0
            )
            if jump_mask.any():
                flags.append("position_jump")

        return {
            "status": "ok",
            "anomaly_score": anomaly_score,
            "flags": flags,
            "latest_model_score": latest_score,
        }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local prediction service check.")
    parser.add_argument("--mmsi", required=True)
    parser.add_argument("--last_n", type=int, default=10)
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--processed_dir", default="data/processed")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    service = PredictionService(model_dir=args.model_dir, processed_dir=args.processed_dir)
    dest = service.predict_destination(mmsi=args.mmsi, last_n=args.last_n)
    eta = service.predict_eta(mmsi=args.mmsi, last_n=args.last_n)
    anomaly = service.score_anomaly(mmsi=args.mmsi, last_n=args.last_n)
    print(json.dumps({"destination": dest, "eta": eta, "anomaly": anomaly}, indent=2))


if __name__ == "__main__":
    main()
