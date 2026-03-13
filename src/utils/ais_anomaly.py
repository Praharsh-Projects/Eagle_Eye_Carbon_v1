"""AIS anomaly helpers for runtime jump detection from event rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def detect_sudden_jump_events_from_parquet(
    events_path: str | Path,
    mmsi: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_minutes: int = 30,
    km_threshold: float = 80.0,
    limit: int = 200,
) -> Dict[str, Any]:
    path = Path(events_path)
    if not path.exists():
        return {"count": 0, "events": [], "reason": f"Events file missing: {path}"}

    df = pd.read_parquet(path)
    if df.empty:
        return {"count": 0, "events": [], "reason": "Events parquet is empty."}

    work = df.copy()
    if "event_kind" in work.columns:
        work = work[work["event_kind"].astype(str) == "ais_position"]
    if mmsi and "mmsi" in work.columns:
        work = work[work["mmsi"].astype(str).str.strip() == str(mmsi).strip()]

    timestamp_source = "timestamp_full" if "timestamp_full" in work.columns else "timestamp"
    if timestamp_source not in work.columns:
        return {"count": 0, "events": [], "reason": "No timestamp column available for jump detection."}

    work["timestamp_dt"] = pd.to_datetime(work[timestamp_source], errors="coerce", utc=True)
    if date_from:
        work = work[work["timestamp_dt"] >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        work = work[work["timestamp_dt"] <= pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if "latitude" not in work.columns or "longitude" not in work.columns:
        return {"count": 0, "events": [], "reason": "Latitude/longitude columns missing from events parquet."}

    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work.dropna(subset=["timestamp_dt", "latitude", "longitude"])
    if "mmsi" not in work.columns:
        work["mmsi"] = None
    work = work.dropna(subset=["mmsi"])
    if work.empty:
        return {"count": 0, "events": [], "reason": "No AIS points available after filtering."}

    work = work.sort_values(["mmsi", "timestamp_dt"])
    events: List[Dict[str, Any]] = []
    for _, group in work.groupby("mmsi"):
        g = group.copy()
        g["prev_timestamp_dt"] = g["timestamp_dt"].shift(1)
        g["prev_latitude"] = g["latitude"].shift(1)
        g["prev_longitude"] = g["longitude"].shift(1)
        dt_minutes = (g["timestamp_dt"] - g["prev_timestamp_dt"]).dt.total_seconds() / 60.0
        dlat = g["latitude"] - g["prev_latitude"]
        dlon = g["longitude"] - g["prev_longitude"]
        dist_km = ((dlat * 111.0) ** 2 + (dlon * 111.0) ** 2) ** 0.5
        flagged = g[(dt_minutes > 0) & (dt_minutes <= max_minutes) & (dist_km >= km_threshold)].copy()
        if flagged.empty:
            continue
        flagged["dt_minutes"] = dt_minutes.loc[flagged.index].astype(float)
        flagged["distance_km"] = dist_km.loc[flagged.index].astype(float)
        for _, row in flagged.iterrows():
            events.append(
                {
                    "stable_id": str(row.get("stable_id", "")),
                    "mmsi": str(row.get("mmsi", "")),
                    "timestamp_full": row["timestamp_dt"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "latitude": _safe_float(row.get("latitude")),
                    "longitude": _safe_float(row.get("longitude")),
                    "prev_latitude": _safe_float(row.get("prev_latitude")),
                    "prev_longitude": _safe_float(row.get("prev_longitude")),
                    "dt_minutes": float(row.get("dt_minutes", 0.0)),
                    "distance_km": float(row.get("distance_km", 0.0)),
                    "port": row.get("locode_norm")
                    or row.get("locode")
                    or row.get("destination_norm")
                    or row.get("port_name_norm"),
                }
            )
            if len(events) >= limit:
                break
        if len(events) >= limit:
            break

    return {"count": len(events), "events": events, "reason": "Computed from row-level AIS events parquet."}
