"""Build Eagle Eye carbon layer outputs (TTW + WTW + provenance + uncertainty)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.carbon.factors import CarbonFactorRegistry, load_factor_registry


AIS_EVENT_COLUMNS = [
    "stable_id",
    "event_kind",
    "source_file",
    "mmsi",
    "vessel_type_norm",
    "destination_norm",
    "timestamp_full",
    "timestamp_date",
    "speed",
    "speed_estimated_kn",
    "locode_norm",
    "port_name_norm",
    "latitude",
    "longitude",
]

DWELL_COLUMNS = [
    "source_kind",
    "port_key",
    "port_label",
    "locode_norm",
    "port_name_norm",
    "mmsi",
    "vessel_type_norm",
    "arrival_time",
    "departure_time",
    "dwell_minutes",
    "source_file",
]

POLLUTANT_COLUMNS = ["co2_t", "nox_kg", "sox_kg", "pm_kg", "ttw_co2e_t", "wtt_co2e_t", "wtw_co2e_t"]


@dataclass
class CarbonBuildSummary:
    output_paths: Dict[str, str]
    stats: Dict[str, Any]
    params_version: Dict[str, Any]


def _load_events(events_path: Path, limit_ais_rows: Optional[int]) -> pd.DataFrame:
    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing {events_path}. Build processed datasets first (e.g., `./run_demo_pipeline.sh`)."
        )
    df = pd.read_parquet(events_path, columns=AIS_EVENT_COLUMNS)
    df = df[df["event_kind"] == "ais_position"].copy()
    if limit_ais_rows and limit_ais_rows > 0:
        df = df.head(limit_ais_rows).copy()

    df["mmsi"] = df["mmsi"].fillna("").astype(str).str.strip()
    df = df[df["mmsi"] != ""].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp_full"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values(["mmsi", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def _load_dwell(dwell_path: Path) -> pd.DataFrame:
    if not dwell_path.exists():
        return pd.DataFrame(columns=DWELL_COLUMNS)
    dwell = pd.read_parquet(dwell_path, columns=DWELL_COLUMNS)
    for col in ("arrival_time", "departure_time"):
        dwell[col] = pd.to_datetime(dwell[col], errors="coerce", utc=True)
    dwell["mmsi"] = dwell["mmsi"].fillna("").astype(str).str.strip()
    dwell = dwell.dropna(subset=["arrival_time", "departure_time"])
    dwell = dwell[(dwell["mmsi"] != "") & (dwell["departure_time"] >= dwell["arrival_time"])].copy()
    dwell = dwell.sort_values(["mmsi", "arrival_time"], kind="mergesort").reset_index(drop=True)
    dwell["call_id"] = (
        dwell["mmsi"].astype(str)
        + "_"
        + dwell["arrival_time"].dt.strftime("%Y-%m-%dT%H-%M-%S")
        + "_"
        + dwell["port_key"].fillna("").astype(str).str.upper()
    )
    return dwell


def _attach_port_call_windows(ais: pd.DataFrame, dwell: pd.DataFrame) -> pd.DataFrame:
    if dwell.empty:
        out = ais.copy()
        out["arrival_time"] = pd.NaT
        out["departure_time"] = pd.NaT
        out["call_id"] = None
        out["call_port_key"] = None
        out["call_port_label"] = None
        out["call_locode_norm"] = None
        out["call_port_name_norm"] = None
        out["inside_call"] = False
        return out

    right = dwell[
        [
            "mmsi",
            "call_id",
            "arrival_time",
            "departure_time",
            "port_key",
            "port_label",
            "locode_norm",
            "port_name_norm",
            "vessel_type_norm",
        ]
    ].copy()
    right = right.rename(
        columns={
            "port_key": "call_port_key",
            "port_label": "call_port_label",
            "locode_norm": "call_locode_norm",
            "port_name_norm": "call_port_name_norm",
            "vessel_type_norm": "call_vessel_type_norm",
        }
    )

    merged = pd.merge_asof(
        ais.sort_values(["timestamp", "mmsi"], kind="mergesort"),
        right.sort_values(["arrival_time", "mmsi"], kind="mergesort"),
        by="mmsi",
        left_on="timestamp",
        right_on="arrival_time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["inside_call"] = (
        merged["arrival_time"].notna() & merged["departure_time"].notna() & (merged["timestamp"] <= merged["departure_time"])
    )

    outside_mask = ~merged["inside_call"]
    clear_cols = [
        "call_id",
        "arrival_time",
        "departure_time",
        "call_port_key",
        "call_port_label",
        "call_locode_norm",
        "call_port_name_norm",
        "call_vessel_type_norm",
    ]
    for col in clear_cols:
        if col in merged.columns:
            merged.loc[outside_mask, col] = None
    return merged


def _assign_modes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speed_kn"] = pd.to_numeric(out["speed"], errors="coerce")
    out["speed_est_kn"] = pd.to_numeric(out["speed_estimated_kn"], errors="coerce")
    out["speed_kn"] = out["speed_kn"].fillna(out["speed_est_kn"]).fillna(0.0).clip(lower=0.0)
    out["speed_interpolated"] = out["speed"].isna() & out["speed_est_kn"].notna()

    in_call = out["inside_call"].fillna(False)
    ts = out["timestamp"]
    arr = pd.to_datetime(out["arrival_time"], errors="coerce", utc=True)
    dep = pd.to_datetime(out["departure_time"], errors="coerce", utc=True)

    near_arrival = in_call & (ts >= (arr - pd.Timedelta(hours=2))) & (ts <= (arr + pd.Timedelta(hours=1)))
    near_departure = in_call & (ts >= (dep - pd.Timedelta(hours=1))) & (ts <= (dep + pd.Timedelta(hours=2)))
    manoeuvring = in_call & out["speed_kn"].between(0.5, 8.0, inclusive="both") & (near_arrival | near_departure)
    berth = in_call & (~manoeuvring) & (out["speed_kn"] <= 0.5)
    if "berth_timestamp" in out.columns:
        berth_ts = pd.to_datetime(out["berth_timestamp"], errors="coerce", utc=True)
        berth_prioritized = in_call & berth_ts.notna() & (ts >= berth_ts) & (~manoeuvring)
        berth = berth | berth_prioritized
    anchorage = in_call & (~manoeuvring) & out["speed_kn"].between(0.5, 2.0, inclusive="right")

    out["mode"] = "transit"
    out.loc[manoeuvring, "mode"] = "manoeuvring"
    out.loc[berth, "mode"] = "berth"
    out.loc[anchorage, "mode"] = "anchorage"

    # Transit is always outside active port-call windows.
    out.loc[~in_call, "mode"] = "transit"
    return out


def _assign_interval_durations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["next_timestamp"] = out.groupby("mmsi", sort=False)["timestamp"].shift(-1)
    out["duration_h_raw"] = (out["next_timestamp"] - out["timestamp"]).dt.total_seconds() / 3600.0

    med = out["duration_h_raw"].where(
        (out["duration_h_raw"] > 0.0) & (out["duration_h_raw"] <= 2.0)
    ).groupby(out["mmsi"], sort=False).transform("median")
    out["duration_h"] = out["duration_h_raw"]
    out.loc[(out["duration_h"] <= 0.0) | (out["duration_h"] > 6.0) | out["duration_h"].isna(), "duration_h"] = med
    out["duration_h"] = out["duration_h"].fillna(1.0 / 6.0).clip(lower=1.0 / 120.0, upper=6.0)
    out["ais_gap_flag"] = out["duration_h"] > 0.5
    return out


def _add_factor_columns(df: pd.DataFrame, registry: CarbonFactorRegistry) -> pd.DataFrame:
    out = df.copy()
    out["vessel_type_norm"] = out["vessel_type_norm"].fillna("").astype(str).str.lower().str.strip()
    out["vessel_class"] = out["vessel_type_norm"].map(registry.resolve_vessel_class)

    vessel_defaults = {
        cls: registry.vessel_defaults(cls)
        for cls in ["tanker", "cargo", "container", "passenger", "service", "unknown"]
    }
    out["mcr_kw"] = out["vessel_class"].map(lambda x: float(vessel_defaults.get(x, vessel_defaults["unknown"]).get("mcr_kw", 8000)))
    out["ref_speed_kn"] = out["vessel_class"].map(
        lambda x: float(vessel_defaults.get(x, vessel_defaults["unknown"]).get("ref_speed_kn", 12.0))
    )
    out["fuel_type"] = out["vessel_class"].map(
        lambda x: str(vessel_defaults.get(x, vessel_defaults["unknown"]).get("fuel", "MGO")).upper()
    )
    out["engine_family"] = out["vessel_class"].map(
        lambda x: str(vessel_defaults.get(x, vessel_defaults["unknown"]).get("engine_family", "medium_speed_diesel"))
    )

    aux_lookup: Dict[str, float] = {}
    for mode in ("transit", "manoeuvring", "berth", "anchorage"):
        for cls in ("tanker", "cargo", "container", "passenger", "service", "unknown"):
            aux_lookup[f"{mode}|{cls}"] = registry.mode_aux_power_kw(mode, cls)
    out["aux_power_kw"] = (
        (out["mode"].astype(str) + "|" + out["vessel_class"].astype(str))
        .map(aux_lookup)
        .fillna(registry.mode_aux_power_kw("transit", "unknown"))
        .astype(float)
    )

    sfc_main = {m: registry.mode_sfc_main(m) for m in ("transit", "manoeuvring", "berth", "anchorage")}
    sfc_aux = {m: registry.mode_sfc_aux(m) for m in ("transit", "manoeuvring", "berth", "anchorage")}
    sulfur = {m: registry.mode_sulfur_fraction(m) for m in ("transit", "manoeuvring", "berth", "anchorage")}
    out["sfc_main_g_per_kwh"] = out["mode"].map(sfc_main).fillna(0.0).astype(float)
    out["sfc_aux_g_per_kwh"] = out["mode"].map(sfc_aux).fillna(220.0).astype(float)
    out["sulfur_fraction"] = out["mode"].map(sulfur).fillna(0.001).astype(float)

    fuel_co2 = {}
    fuel_wtt = {}
    for fuel in sorted(out["fuel_type"].dropna().unique()):
        factors = registry.fuel_factors(str(fuel))
        fuel_co2[str(fuel)] = float(factors["co2_t_per_t_fuel"])
        fuel_wtt[str(fuel)] = float(factors["wtt_co2e_t_per_t_fuel"])
    out["co2_factor_t_per_t_fuel"] = out["fuel_type"].map(fuel_co2).fillna(registry.fuel_factors("MGO")["co2_t_per_t_fuel"])
    out["wtt_factor_t_per_t_fuel"] = out["fuel_type"].map(fuel_wtt).fillna(registry.fuel_factors("MGO")["wtt_co2e_t_per_t_fuel"])

    nox_lookup: Dict[str, float] = {}
    pm_lookup: Dict[str, float] = {}
    for family in ("slow_speed_diesel", "medium_speed_diesel", "high_speed_diesel"):
        for mode in ("transit", "manoeuvring", "berth", "anchorage"):
            nox_lookup[f"{family}|{mode}"] = registry.nox_factor(family, mode)
            pm_lookup[f"{family}|{mode}"] = registry.pm_factor(family, mode)
    family_mode = out["engine_family"].astype(str) + "|" + out["mode"].astype(str)
    out["nox_kg_per_t_fuel"] = family_mode.map(nox_lookup).fillna(registry.nox_factor("medium_speed_diesel", "transit"))
    out["pm_kg_per_t_fuel"] = family_mode.map(pm_lookup).fillna(registry.pm_factor("medium_speed_diesel", "transit"))

    default_fuel = str(registry.assumptions.get("default_fuel", "MGO")).upper()
    default_engine = str(registry.assumptions.get("default_engine_family", "medium_speed_diesel"))
    out["fallback_factor_flag"] = (
        out["vessel_class"].eq("unknown")
        | out["fuel_type"].eq(default_fuel)
        | out["engine_family"].eq(default_engine)
    )
    return out


def _compute_emissions(df: pd.DataFrame, registry: CarbonFactorRegistry) -> pd.DataFrame:
    out = df.copy()

    speed_ratio = (out["speed_kn"] / out["ref_speed_kn"].replace(0, np.nan)).fillna(0.0)
    lf = speed_ratio.pow(3).clip(lower=0.2, upper=1.0)
    lf = np.where(out["mode"].isin(["transit", "manoeuvring"]), lf, 0.0)
    out["load_factor"] = lf

    out["p_main_kw"] = out["mcr_kw"] * out["load_factor"]
    out["p_aux_kw"] = out["aux_power_kw"]

    out["fuel_t"] = (
        (out["p_main_kw"] * out["sfc_main_g_per_kwh"] + out["p_aux_kw"] * out["sfc_aux_g_per_kwh"])
        * out["duration_h"]
        / 1_000_000.0
    ).clip(lower=0.0)

    out["co2_t"] = out["fuel_t"] * out["co2_factor_t_per_t_fuel"]
    out["nox_kg"] = out["fuel_t"] * out["nox_kg_per_t_fuel"]
    out["pm_kg"] = out["fuel_t"] * out["pm_kg_per_t_fuel"]
    sox_multiplier = float(registry.assumptions.get("sox_multiplier", 2.0))
    out["sox_kg"] = out["fuel_t"] * out["sulfur_fraction"] * 1000.0 * sox_multiplier
    out["ttw_co2e_t"] = out["co2_t"]
    out["wtt_co2e_t"] = out["fuel_t"] * out["wtt_factor_t_per_t_fuel"]
    out["wtw_co2e_t"] = out["ttw_co2e_t"] + out["wtt_co2e_t"]

    u = registry.uncertainty_defaults
    base_sigma = np.sqrt(
        float(u.get("speed_rel_sigma", 0.08)) ** 2
        + float(u.get("sfc_rel_sigma", 0.10)) ** 2
        + float(u.get("factor_rel_sigma", 0.08)) ** 2
    )
    out["rel_sigma"] = (
        base_sigma
        + out["speed_interpolated"].astype(float) * float(u.get("interpolation_penalty_sigma", 0.10))
        + out["ais_gap_flag"].astype(float) * float(u.get("gap_penalty_sigma", 0.08))
        + out["fallback_factor_flag"].astype(float) * float(u.get("fallback_penalty_sigma", 0.12))
    ).clip(lower=0.05, upper=0.95)

    # Deterministic segment boundaries.
    prev_mode = out.groupby("mmsi", sort=False)["mode"].shift(1)
    prev_call = out.groupby("mmsi", sort=False)["call_id"].shift(1).fillna("")
    prev_gap = out.groupby("mmsi", sort=False)["duration_h"].shift(1).fillna(0.0)
    out["new_segment"] = (
        prev_mode.isna()
        | (out["mode"] != prev_mode)
        | (out["call_id"].fillna("") != prev_call)
        | (prev_gap > 0.5)
    )
    out["segment_ord"] = out.groupby("mmsi", sort=False)["new_segment"].cumsum().astype(int)
    out["segment_id"] = out["mmsi"].astype(str) + "_seg_" + out["segment_ord"].astype(str).str.zfill(6)
    return out


def _confidence_label(ci_width_rel: float, fallback_ratio: float) -> str:
    if ci_width_rel <= 0.20 and fallback_ratio <= 0.05:
        return "high"
    if ci_width_rel <= 0.40 or fallback_ratio <= 0.20:
        return "medium"
    return "low"


def _confidence_reason(ci_width_rel: float, fallback_ratio: float, coverage_ratio: float) -> str:
    return (
        f"CI width={ci_width_rel:.2f}, fallback_ratio={fallback_ratio:.2f}, "
        f"coverage_ratio={coverage_ratio:.2f}."
    )


def _add_intervals(df: pd.DataFrame, metric_cols: List[str], rel_sigma_col: str) -> pd.DataFrame:
    out = df.copy()
    sigma = out[rel_sigma_col].fillna(0.30).clip(lower=0.05, upper=0.95)
    for metric in metric_cols:
        center = out[metric].fillna(0.0)
        out[f"{metric}_lower"] = (center * (1.0 - 1.96 * sigma)).clip(lower=0.0)
        out[f"{metric}_upper"] = (center * (1.0 + 1.96 * sigma)).clip(lower=0.0)
    return out


def _aggregate_with_uncertainty(
    df: pd.DataFrame,
    group_cols: List[str],
    draws: int,
    seed: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    agg_spec: Dict[str, Any] = {
        "segment_id": "nunique",
        "duration_h": "sum",
        "row_count": "sum",
        "ais_gap_ratio": "mean",
        "interpolation_ratio": "mean",
        "fallback_usage_ratio": "mean",
        "rel_sigma_weighted": "mean",
    }
    for metric in POLLUTANT_COLUMNS:
        agg_spec[metric] = "sum"

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(agg_spec)
        .reset_index()
        .rename(columns={"segment_id": "segments"})
    )

    rng = np.random.default_rng(seed)
    for metric in POLLUTANT_COLUMNS:
        lower_vals = []
        upper_vals = []
        for total, rel in zip(out[metric].to_numpy(dtype=float), out["rel_sigma_weighted"].to_numpy(dtype=float)):
            rel = float(np.clip(rel, 0.05, 0.95))
            if total <= 0:
                lower_vals.append(0.0)
                upper_vals.append(0.0)
                continue
            samples = rng.normal(loc=total, scale=abs(total) * rel, size=draws)
            samples = np.clip(samples, 0.0, None)
            lower_vals.append(float(np.quantile(samples, 0.025)))
            upper_vals.append(float(np.quantile(samples, 0.975)))
        out[f"{metric}_lower"] = lower_vals
        out[f"{metric}_upper"] = upper_vals

    point = out["wtw_co2e_t"].where(out["wtw_co2e_t"] > 0, out["co2_t"]).fillna(0.0)
    lower = out["wtw_co2e_t_lower"].where(out["wtw_co2e_t_lower"] > 0, out["co2_t_lower"]).fillna(0.0)
    upper = out["wtw_co2e_t_upper"].where(out["wtw_co2e_t_upper"] > 0, out["co2_t_upper"]).fillna(0.0)
    out["ci_width_rel"] = np.where(point > 0, (upper - lower) / point, 1.0)
    out["confidence_label"] = [
        _confidence_label(float(ci), float(fb))
        for ci, fb in zip(out["ci_width_rel"], out["fallback_usage_ratio"])
    ]
    out["confidence_reason"] = [
        _confidence_reason(float(ci), float(fb), float(1.0 - min(1.0, gap)))
        for ci, fb, gap in zip(out["ci_width_rel"], out["fallback_usage_ratio"], out["ais_gap_ratio"])
    ]
    return out


def build_carbon_layer(
    processed_dir: str | Path = "data/processed",
    out_dir: str | Path | None = None,
    factor_registry_path: str | Path = "config/carbon_factors.v1.json",
    monte_carlo_draws: int = 500,
    limit_ais_rows: Optional[int] = None,
) -> CarbonBuildSummary:
    processed = Path(processed_dir)
    out_base = Path(out_dir) if out_dir else processed
    out_base.mkdir(parents=True, exist_ok=True)

    registry = load_factor_registry(factor_registry_path)
    params_hash = hashlib.sha256(
        (
            registry.factor_payload_hash()
            + "|segment_rules:v1|monte_carlo_draws:"
            + str(int(monte_carlo_draws))
        ).encode("utf-8")
    ).hexdigest()

    ais = _load_events(processed / "events.parquet", limit_ais_rows=limit_ais_rows)
    dwell = _load_dwell(processed / "dwell_time.parquet")
    enriched = _attach_port_call_windows(ais, dwell)
    enriched = _assign_modes(enriched)
    enriched = _assign_interval_durations(enriched)
    enriched = _add_factor_columns(enriched, registry=registry)
    enriched = _compute_emissions(enriched, registry=registry)

    enriched["coverage_ratio"] = 1.0 - (
        0.5 * enriched["ais_gap_flag"].astype(float) + 0.5 * enriched["speed_interpolated"].astype(float)
    ).clip(upper=1.0)

    segment_agg = (
        enriched.groupby("segment_id", sort=False, dropna=False)
        .agg(
            mmsi=("mmsi", "first"),
            call_id=("call_id", "first"),
            mode=("mode", "first"),
            vessel_class=("vessel_class", "first"),
            engine_family=("engine_family", "first"),
            fuel_type=("fuel_type", "first"),
            port_key=("call_port_key", "first"),
            port_label=("call_port_label", "first"),
            locode_norm=("call_locode_norm", "first"),
            port_name_norm=("call_port_name_norm", "first"),
            destination_norm=("destination_norm", "first"),
            timestamp_start=("timestamp", "min"),
            timestamp_end=("timestamp", "max"),
            row_count=("stable_id", "size"),
            input_row_start_id=("stable_id", "first"),
            input_row_end_id=("stable_id", "last"),
            duration_h=("duration_h", "sum"),
            ais_gap_ratio=("ais_gap_flag", "mean"),
            interpolation_ratio=("speed_interpolated", "mean"),
            fallback_usage_ratio=("fallback_factor_flag", "mean"),
            fallback_usage_count=("fallback_factor_flag", "sum"),
            coverage_ratio=("coverage_ratio", "mean"),
            rel_sigma_weighted=("rel_sigma", "mean"),
            fuel_t=("fuel_t", "sum"),
            co2_t=("co2_t", "sum"),
            nox_kg=("nox_kg", "sum"),
            sox_kg=("sox_kg", "sum"),
            pm_kg=("pm_kg", "sum"),
            ttw_co2e_t=("ttw_co2e_t", "sum"),
            wtt_co2e_t=("wtt_co2e_t", "sum"),
            wtw_co2e_t=("wtw_co2e_t", "sum"),
        )
        .reset_index()
    )

    segment_agg = _add_intervals(segment_agg, metric_cols=POLLUTANT_COLUMNS, rel_sigma_col="rel_sigma_weighted")
    base_point = segment_agg["wtw_co2e_t"].where(segment_agg["wtw_co2e_t"] > 0, segment_agg["co2_t"]).fillna(0.0)
    base_lower = segment_agg["wtw_co2e_t_lower"].where(
        segment_agg["wtw_co2e_t_lower"] > 0, segment_agg["co2_t_lower"]
    ).fillna(0.0)
    base_upper = segment_agg["wtw_co2e_t_upper"].where(
        segment_agg["wtw_co2e_t_upper"] > 0, segment_agg["co2_t_upper"]
    ).fillna(0.0)
    segment_agg["ci_width_rel"] = np.where(base_point > 0, (base_upper - base_lower) / base_point, 1.0)
    segment_agg["confidence_label"] = [
        _confidence_label(float(ci), float(fallback))
        for ci, fallback in zip(segment_agg["ci_width_rel"], segment_agg["fallback_usage_ratio"])
    ]
    segment_agg["confidence_reason"] = [
        _confidence_reason(float(ci), float(fallback), float(cov))
        for ci, fallback, cov in zip(
            segment_agg["ci_width_rel"], segment_agg["fallback_usage_ratio"], segment_agg["coverage_ratio"]
        )
    ]
    segment_agg["quality_ais_gap_flag"] = segment_agg["ais_gap_ratio"] > 0.25
    segment_agg["quality_interpolation_flag"] = segment_agg["interpolation_ratio"] > 0.30
    segment_agg["quality_factor_fallback_flag"] = segment_agg["fallback_usage_ratio"] > 0.20
    segment_agg["params_version"] = registry.version
    segment_agg["params_hash"] = params_hash
    segment_agg["evidence_id"] = [
        hashlib.sha1(f"{sid}|{params_hash}".encode("utf-8")).hexdigest()[:24]
        for sid in segment_agg["segment_id"].astype(str)
    ]

    segment_agg["date"] = pd.to_datetime(segment_agg["timestamp_start"], errors="coerce", utc=True).dt.floor("D")
    segment_agg["destination_norm"] = segment_agg["destination_norm"].fillna("").astype(str).str.upper()
    segment_agg["port_key"] = (
        segment_agg["port_key"]
        .fillna(segment_agg["locode_norm"])
        .fillna(segment_agg["destination_norm"])
        .replace("", np.nan)
        .fillna("AT_SEA")
    )
    segment_agg["port_label"] = segment_agg["port_label"].fillna(segment_agg["port_key"])
    segment_agg["locode_norm"] = segment_agg["locode_norm"].fillna("")

    daily_port = _aggregate_with_uncertainty(
        segment_agg,
        group_cols=["date", "port_key", "port_label", "locode_norm"],
        draws=max(100, int(monte_carlo_draws)),
        seed=42,
    )
    if not daily_port.empty:
        daily_port["params_version"] = registry.version
        daily_port["params_hash"] = params_hash

    call_segments = segment_agg[segment_agg["call_id"].notna() & (segment_agg["call_id"].astype(str) != "")]
    call_totals = _aggregate_with_uncertainty(
        call_segments,
        group_cols=["call_id", "mmsi", "port_key", "port_label", "locode_norm"],
        draws=max(100, int(monte_carlo_draws)),
        seed=43,
    )
    if not call_totals.empty:
        call_totals["params_version"] = registry.version
        call_totals["params_hash"] = params_hash

    evidence_df = segment_agg[
        [
            "evidence_id",
            "segment_id",
            "mmsi",
            "call_id",
            "port_key",
            "port_label",
            "locode_norm",
            "timestamp_start",
            "timestamp_end",
            "row_count",
            "input_row_start_id",
            "input_row_end_id",
            "ais_gap_ratio",
            "interpolation_ratio",
            "fallback_usage_ratio",
            "coverage_ratio",
            "ci_width_rel",
            "confidence_label",
            "confidence_reason",
            "params_version",
            "params_hash",
        ]
        + POLLUTANT_COLUMNS
    ].copy()

    outputs = {
        "carbon_segments": out_base / "carbon_segments.parquet",
        "carbon_emissions_segment": out_base / "carbon_emissions_segment.parquet",
        "carbon_emissions_daily_port": out_base / "carbon_emissions_daily_port.parquet",
        "carbon_emissions_call": out_base / "carbon_emissions_call.parquet",
        "carbon_evidence": out_base / "carbon_evidence.parquet",
        "carbon_params_version": out_base / "carbon_params_version.json",
    }

    segment_agg.to_parquet(outputs["carbon_segments"], index=False)
    segment_agg[
        [
            "segment_id",
            "mmsi",
            "call_id",
            "mode",
            "port_key",
            "port_label",
            "locode_norm",
            "timestamp_start",
            "timestamp_end",
            "duration_h",
            "row_count",
            "rel_sigma_weighted",
            "ci_width_rel",
            "confidence_label",
            "confidence_reason",
            "evidence_id",
        ]
        + POLLUTANT_COLUMNS
        + [f"{c}_lower" for c in POLLUTANT_COLUMNS]
        + [f"{c}_upper" for c in POLLUTANT_COLUMNS]
    ].to_parquet(outputs["carbon_emissions_segment"], index=False)
    daily_port.to_parquet(outputs["carbon_emissions_daily_port"], index=False)
    call_totals.to_parquet(outputs["carbon_emissions_call"], index=False)
    evidence_df.to_parquet(outputs["carbon_evidence"], index=False)

    params_version = {
        "version": registry.version,
        "factor_registry_path": str(registry.source_path),
        "factor_checksum_sha256": registry.checksum_sha256,
        "params_hash": params_hash,
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "monte_carlo_draws": int(monte_carlo_draws),
        "segment_rules": {
            "transit": "outside active port-call window",
            "manoeuvring": "arrival-2h..arrival+1h or departure-1h..departure+2h and speed 0.5..8 kn",
            "berth": "inside active call, outside manoeuvring, speed <=0.5 kn",
            "anchorage": "inside active call, outside manoeuvring, speed >0.5 and <=2.0 kn",
        },
        "stats": {
            "ais_rows": int(len(enriched)),
            "segments": int(len(segment_agg)),
            "daily_port_rows": int(len(daily_port)),
            "call_rows": int(len(call_totals)),
            "evidence_rows": int(len(evidence_df)),
        },
    }
    outputs["carbon_params_version"].write_text(json.dumps(params_version, indent=2), encoding="utf-8")

    stats = {
        "ais_rows": int(len(enriched)),
        "segments": int(len(segment_agg)),
        "ports": int(daily_port["port_key"].nunique()) if not daily_port.empty else 0,
        "calls": int(call_totals["call_id"].nunique()) if not call_totals.empty else 0,
        "mean_segment_ci_width_rel": float(segment_agg["ci_width_rel"].mean()) if not segment_agg.empty else None,
    }
    return CarbonBuildSummary(
        output_paths={k: str(v) for k, v in outputs.items()},
        stats=stats,
        params_version=params_version,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build carbon layer datasets for Eagle Eye.")
    parser.add_argument("--processed_dir", default="data/processed", help="Directory containing events.parquet and dwell_time.parquet")
    parser.add_argument("--out_dir", default=None, help="Output directory for carbon artifacts (default: processed_dir)")
    parser.add_argument(
        "--factor_registry_path",
        default="config/carbon_factors.v1.json",
        help="Local factor registry JSON path",
    )
    parser.add_argument("--monte_carlo_draws", type=int, default=500, help="Monte Carlo draws for aggregate intervals")
    parser.add_argument("--limit_ais_rows", type=int, default=None, help="Optional row cap for fast development runs")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_carbon_layer(
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        factor_registry_path=args.factor_registry_path,
        monte_carlo_draws=args.monte_carlo_draws,
        limit_ais_rows=args.limit_ais_rows,
    )
    print("Carbon layer build completed")
    print(json.dumps(summary.stats, indent=2))
    print(json.dumps(summary.output_paths, indent=2))
    print(json.dumps(summary.params_version, indent=2))


if __name__ == "__main__":
    main()
