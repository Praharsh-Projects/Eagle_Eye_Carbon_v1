"""Create cleaned event and training datasets for destination/ETA/anomaly models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_DEST_ALIASES: Dict[str, str] = {
    "RU LED": "RULED",
    "RULED": "RULED",
    "LT KLJ": "LTKLJ",
    "LTKLJ": "LTKLJ",
    "SE GOT": "SEGOT",
    "SEGOT": "SEGOT",
    "SE KAR": "SEKAR",
    "SEKAR": "SEKAR",
    "FI HEL": "FIHEL",
    "FIHEL": "FIHEL",
}


AIS_REQUIRED_COLUMNS = {
    "MMSI",
    "TimePosition",
    "Latitude",
    "Longitude",
    "Speed",
    "Course",
    "Heading",
    "NavStatus",
    "IMO",
    "Name",
    "Callsign",
    "Flag",
    "VesselType",
    "Destination",
    "TimeETA",
    "Draught",
}


PORT_CALL_REQUIRED_COLUMNS = {
    "portID",
    "portName",
    "portLocode",
    "portArrival",
    "portDeparture",
    "vesselMMSI",
    "vesselIMO",
    "vesselName",
    "vesselDestinationArrival",
    "vesselDestinationDeparture",
    "vesselType",
}


def normalize_destination(value: object, aliases: Dict[str, str]) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).upper().strip()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "UNKNOWN"

    if text in aliases:
        return aliases[text]
    nospace = text.replace(" ", "")
    if nospace in aliases:
        return aliases[nospace]

    if re.match(r"^[A-Z]{2}\s[A-Z]{3}$", text):
        return text.replace(" ", "")
    if re.match(r"^[A-Z]{2}[A-Z]{3}$", nospace):
        return nospace
    return text


def _normalize_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if re.match(r"^\d+\.0+$", text):
        return text.split(".")[0]
    return text


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """
    Vectorized haversine distance in km.
    """
    rad = np.pi / 180.0
    lat1r = lat1 * rad
    lon1r = lon1 * rad
    lat2r = lat2 * rad
    lon2r = lon2 * rad

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0088 * c


def _minimal_angle_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    raw = (a - b).abs()
    return np.minimum(raw, 360 - raw)


def _load_csv_with_schema(csv_path: Path, limit_rows: Optional[int]) -> pd.DataFrame:
    cols: List[str]
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().replace('"', "")
    cols = [c.strip() for c in header.split(",")]
    dtype = {c: "string" for c in cols}
    df = pd.read_csv(
        csv_path,
        dtype=dtype,
        low_memory=False,
        nrows=limit_rows if limit_rows and limit_rows > 0 else None,
    )
    return df


def _is_ais_schema(columns: Sequence[str]) -> bool:
    return len(AIS_REQUIRED_COLUMNS.intersection(set(columns))) >= 8


def _is_port_call_schema(columns: Sequence[str]) -> bool:
    return len(PORT_CALL_REQUIRED_COLUMNS.intersection(set(columns))) >= 6


def _prepare_ais_events(
    df_raw: pd.DataFrame, source_file: str, aliases: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (events_df, training_rows_df) for AIS rows.
    """
    df = df_raw.copy()
    for col in AIS_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["mmsi"] = df["MMSI"].map(_normalize_id)
    df["imo"] = df["IMO"].map(_normalize_id)
    df["timestamp"] = pd.to_datetime(df["TimePosition"], errors="coerce", utc=True)
    df["eta_timestamp"] = pd.to_datetime(df["TimeETA"], errors="coerce", utc=True)
    df["latitude"] = _to_numeric(df["Latitude"])
    df["longitude"] = _to_numeric(df["Longitude"])
    df["speed"] = _to_numeric(df["Speed"])
    df["course"] = _to_numeric(df["Course"])
    df["heading"] = _to_numeric(df["Heading"])
    df["draught"] = _to_numeric(df["Draught"])

    df["name"] = df["Name"].fillna("unknown").astype("string").str.strip()
    df["callsign"] = df["Callsign"].fillna("unknown").astype("string").str.strip()
    df["flag"] = df["Flag"].fillna("unknown").astype("string").str.strip()
    df["flag_norm"] = df["flag"].str.upper().replace("", "UNKNOWN")
    df["vessel_type"] = df["VesselType"].fillna("unknown").astype("string").str.strip()
    df["vessel_type_norm"] = df["vessel_type"].str.lower().replace("", "unknown")
    df["nav_status"] = df["NavStatus"].fillna("unknown").astype("string").str.strip()
    df["nav_status_norm"] = df["nav_status"].str.lower().replace("", "unknown")
    df["destination_raw"] = df["Destination"].fillna("unknown").astype("string").str.strip()
    df["destination_norm"] = df["destination_raw"].map(lambda x: normalize_destination(x, aliases))

    df = df.dropna(subset=["timestamp", "latitude", "longitude"])
    df = df[(df["mmsi"] != "") & (df["mmsi"].str.lower() != "unknown")]
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ]
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    g = df.groupby("mmsi", sort=False)
    prev_ts = g["timestamp"].shift(1)
    prev_lat = g["latitude"].shift(1)
    prev_lon = g["longitude"].shift(1)
    prev_speed = g["speed"].shift(1)
    prev_course = g["course"].shift(1)

    df["delta_time_s"] = (df["timestamp"] - prev_ts).dt.total_seconds()
    df["delta_distance_km"] = _haversine_km(prev_lat, prev_lon, df["latitude"], df["longitude"])
    df.loc[df["delta_time_s"] <= 0, "delta_time_s"] = np.nan

    hours = df["delta_time_s"] / 3600.0
    df["speed_estimated_kn"] = (df["delta_distance_km"] / hours) / 1.852
    df["acceleration_kn_per_h"] = (df["speed"] - prev_speed) / hours
    df["course_change_deg"] = _minimal_angle_diff(df["course"], prev_course)
    df["course_change_rate_deg_per_h"] = df["course_change_deg"] / hours

    for w in (3, 5, 10):
        df[f"speed_roll_mean_{w}"] = g["speed"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"speed_roll_std_{w}"] = g["speed"].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0.0)
        )
        df[f"speed_est_roll_mean_{w}"] = g["speed_estimated_kn"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"course_change_roll_mean_{w}"] = g["course_change_deg"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )

    df["hour_of_day"] = df["timestamp"].dt.hour + (df["timestamp"].dt.minute / 60.0)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["eta_minutes_label"] = (
        (df["eta_timestamp"] - df["timestamp"]).dt.total_seconds() / 60.0
    )
    df.loc[df["eta_minutes_label"] <= 0, "eta_minutes_label"] = np.nan

    df["event_kind"] = "ais_position"
    df["timestamp_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["timestamp_full"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["source_file"] = source_file

    id_ts = df["timestamp"].dt.strftime("%Y-%m-%dT%H-%M-%S")
    df["stable_id"] = (
        df["mmsi"].astype(str)
        + "_"
        + id_ts
        + "_"
        + df["latitude"].round(5).astype(str)
        + "_"
        + df["longitude"].round(5).astype(str)
    )

    # Clean NaN/inf from numeric features.
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    event_cols = [
        "stable_id",
        "event_kind",
        "source_file",
        "mmsi",
        "imo",
        "name",
        "callsign",
        "flag",
        "flag_norm",
        "vessel_type",
        "vessel_type_norm",
        "nav_status",
        "nav_status_norm",
        "destination_raw",
        "destination_norm",
        "timestamp_date",
        "timestamp_full",
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
        "eta_minutes_label",
    ]
    events_df = df[event_cols].copy()
    training_rows_df = events_df.copy()
    training_rows_df["timestamp"] = df["timestamp"]
    return events_df, training_rows_df


def build_ais_feature_rows_from_raw_df(
    df_raw: pd.DataFrame,
    source_file: str = "runtime",
    destination_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Public helper used by prediction service for runtime feature generation.
    """
    aliases = dict(DEFAULT_DEST_ALIASES)
    if destination_aliases:
        aliases.update(destination_aliases)
    _, training_rows = _prepare_ais_events(df_raw, source_file=source_file, aliases=aliases)
    return training_rows


def _prepare_port_call_events(
    df_raw: pd.DataFrame, source_file: str, aliases: Dict[str, str]
) -> pd.DataFrame:
    df = df_raw.copy()
    for col in PORT_CALL_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["mmsi"] = df["vesselMMSI"].map(_normalize_id)
    df["imo"] = df["vesselIMO"].map(_normalize_id)
    df["timestamp"] = pd.to_datetime(df["portArrival"], errors="coerce", utc=True)
    df["departure_ts"] = pd.to_datetime(df["portDeparture"], errors="coerce", utc=True)
    df["port_name"] = df["portName"].fillna("unknown").astype("string").str.strip()
    df["port_name_norm"] = df["port_name"].str.lower().replace("", "unknown")
    df["locode"] = df["portLocode"].fillna("unknown").astype("string").str.strip()
    df["locode_norm"] = df["locode"].str.upper().str.replace(" ", "", regex=False)
    df["vessel_type"] = df["vesselType"].fillna("unknown").astype("string").str.strip()
    df["vessel_type_norm"] = df["vessel_type"].str.lower().replace("", "unknown")
    df["destination_arrival"] = (
        df["vesselDestinationArrival"].fillna("unknown").astype("string").str.strip()
    )
    df["destination_departure"] = (
        df["vesselDestinationDeparture"].fillna("unknown").astype("string").str.strip()
    )
    df["destination_raw"] = np.where(
        df["destination_departure"].str.lower() != "unknown",
        df["destination_departure"],
        df["destination_arrival"],
    )
    df["destination_norm"] = df["destination_raw"].map(lambda x: normalize_destination(x, aliases))

    df = df.dropna(subset=["timestamp"])
    df = df[(df["mmsi"] != "") & (df["mmsi"].str.lower() != "unknown")]
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    id_ts = df["timestamp"].dt.strftime("%Y-%m-%dT%H-%M-%S")
    df["stable_id"] = (
        df["mmsi"].astype(str)
        + "_"
        + id_ts
        + "_"
        + df["locode_norm"].astype(str)
        + "_port_call"
    )
    df["event_kind"] = "port_call"
    df["timestamp_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["timestamp_full"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["source_file"] = source_file

    out_cols = [
        "stable_id",
        "event_kind",
        "source_file",
        "mmsi",
        "imo",
        "vessel_type",
        "vessel_type_norm",
        "destination_raw",
        "destination_norm",
        "timestamp_date",
        "timestamp_full",
        "port_name",
        "port_name_norm",
        "locode",
        "locode_norm",
        "destination_arrival",
        "destination_departure",
    ]
    return df[out_cols].copy()


def prepare_datasets(
    csv_paths: Sequence[Path],
    out_dir: Path,
    limit_rows: Optional[int] = None,
    destination_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path, Dict[str, int]]:
    aliases = dict(DEFAULT_DEST_ALIASES)
    if destination_aliases:
        aliases.update(destination_aliases)

    ais_events: List[pd.DataFrame] = []
    training_rows: List[pd.DataFrame] = []
    port_events: List[pd.DataFrame] = []

    stats = {
        "input_files": len(csv_paths),
        "ais_rows": 0,
        "port_call_rows": 0,
    }

    for csv_path in csv_paths:
        raw = _load_csv_with_schema(csv_path, limit_rows=limit_rows)
        cols = list(raw.columns)
        if _is_ais_schema(cols):
            events_df, train_df = _prepare_ais_events(raw, csv_path.name, aliases)
            ais_events.append(events_df)
            training_rows.append(train_df)
            stats["ais_rows"] += len(events_df)
        elif _is_port_call_schema(cols):
            events_df = _prepare_port_call_events(raw, csv_path.name, aliases)
            port_events.append(events_df)
            stats["port_call_rows"] += len(events_df)
        else:
            print(f"Skipping {csv_path}: unsupported schema")

    if not ais_events:
        raise RuntimeError("No AIS rows were parsed. At least one PRJ912-like CSV is required.")

    out_dir.mkdir(parents=True, exist_ok=True)
    events_out = out_dir / "events.parquet"
    training_out = out_dir / "training_rows.parquet"
    alias_out = out_dir / "destination_aliases.json"

    all_events = pd.concat([*ais_events, *port_events], ignore_index=True, sort=False)
    all_events.to_parquet(events_out, index=False)

    train_df = pd.concat(training_rows, ignore_index=True, sort=False)
    train_df.to_parquet(training_out, index=False)

    with alias_out.open("w", encoding="utf-8") as f:
        json.dump(aliases, f, indent=2)

    return events_out, training_out, stats


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cleaned datasets for prediction training.")
    parser.add_argument(
        "--traffic_csv",
        default=None,
        help="Primary traffic CSV path (PRJ912/PRJ896)",
    )
    parser.add_argument(
        "--traffic_csvs",
        nargs="*",
        default=None,
        help="Additional traffic CSV paths",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed",
        help="Output directory for parquet datasets",
    )
    parser.add_argument(
        "--limit_rows",
        type=int,
        default=None,
        help="Optional row limit per CSV for fast development",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    csv_paths: List[Path] = []
    if args.traffic_csv:
        csv_paths.append(Path(args.traffic_csv))
    if args.traffic_csvs:
        csv_paths.extend(Path(p) for p in args.traffic_csvs)
    if not csv_paths:
        default = Path("data/PRJ912.csv")
        if default.exists():
            csv_paths = [default]
        else:
            raise RuntimeError("Provide --traffic_csv and/or --traffic_csvs.")

    events_out, training_out, stats = prepare_datasets(
        csv_paths=csv_paths,
        out_dir=Path(args.out_dir),
        limit_rows=args.limit_rows,
    )
    print(f"Wrote: {events_out}")
    print(f"Wrote: {training_out}")
    print(
        f"Stats: input_files={stats['input_files']} ais_rows={stats['ais_rows']} "
        f"port_call_rows={stats['port_call_rows']}"
    )


if __name__ == "__main__":
    main()
