"""Carbon query engine for TTW/WTW analytics, uncertainty, and provenance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.carbon.build import build_carbon_layer
from src.carbon.factors import load_factor_registry


SUPPORTED_POLLUTANTS = ["CO2", "CO2e", "NOx", "SOx", "PM"]

CARBON_STATE_COMPUTED = "COMPUTED"
CARBON_STATE_COMPUTED_ZERO = "COMPUTED_ZERO"
CARBON_STATE_NOT_COMPUTABLE = "NOT_COMPUTABLE"
CARBON_STATE_RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
CARBON_STATE_FORECAST_ONLY = "FORECAST_ONLY"
CARBON_STATE_UNSUPPORTED = "UNSUPPORTED"


def _norm_boundary(value: str) -> str:
    token = (value or "TTW").strip().upper()
    if token in {"WTW", "WELL_TO_WAKE", "WELL-TO-WAKE"}:
        return "WTW"
    return "TTW"


def _norm_pollutants(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return ["CO2e", "NOx", "SOx", "PM"]
    out: List[str] = []
    for item in values:
        token = str(item).strip().upper()
        if token in {"CO2", "NOX", "SOX", "PM", "CO2E"}:
            fixed = token.replace("NOX", "NOx").replace("SOX", "SOx").replace("CO2E", "CO2e")
            if fixed not in out:
                out.append(fixed)
    return out or ["CO2e", "NOx", "SOx", "PM"]


def _metric_column(pollutant: str, boundary: str) -> str:
    if pollutant == "CO2":
        return "co2_t"
    if pollutant == "CO2e":
        return "wtw_co2e_t" if boundary == "WTW" else "ttw_co2e_t"
    if pollutant == "NOx":
        return "nox_kg"
    if pollutant == "SOx":
        return "sox_kg"
    if pollutant == "PM":
        return "pm_kg"
    raise ValueError(f"Unsupported pollutant: {pollutant}")


_MODE_ALIASES: Dict[str, str] = {
    "manoeuvring": "manoeuvring",
    "maneuvering": "manoeuvring",
    "transit": "transit",
    "berth": "berth",
    "anchorage": "anchorage",
    "hoteling": "berth",
}
_DURATION_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", flags=re.IGNORECASE)
_SPEED_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:knots?|kn)\b", flags=re.IGNORECASE)
_FUEL_RE = re.compile(r"\b(mgo|hfo|vlsfo|lng|methanol|ammonia|diesel)\b", flags=re.IGNORECASE)
_MCR_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:kw|kW)\b")
_REF_SPEED_RE = re.compile(r"\bref(?:erence)?\s*speed\s*(\d+(?:\.\d+)?)\s*(?:knots?|kn)\b", flags=re.IGNORECASE)


def _normalize_call_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    return re.sub(r"^[\s:_\-]+", "", raw).strip()


def _canonical_call_id(value: str) -> str:
    normalized = _normalize_call_id(value)
    if not normalized:
        return ""
    return re.sub(r"[^A-Z0-9]", "", normalized.upper())


def _extract_estimate_payload(question: str, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = (question or "").lower()
    is_estimate_query = any(token in q for token in ("estimate", "assum", "scenario"))
    mode = None
    for token, canonical in _MODE_ALIASES.items():
        if token in q:
            mode = canonical
            break

    dur_hit = _DURATION_RE.search(question or "")
    speed_hit = _SPEED_RE.search(question or "")
    has_param_signal = bool(mode or dur_hit or speed_hit)
    if not (is_estimate_query and has_param_signal):
        return None

    vessel_type = str(entities.get("vessel_type") or "").strip().lower()
    if not vessel_type:
        if "tanker" in q:
            vessel_type = "tanker"
        elif "container" in q:
            vessel_type = "container ship"
        elif "cargo" in q:
            vessel_type = "cargo ship"
        elif "ferry" in q:
            vessel_type = "ferry"
        else:
            vessel_type = "unknown"

    payload: Dict[str, Any] = {
        "vessel_type": vessel_type,
        "mode": mode or "transit",
        "duration_h": float(dur_hit.group(1)) if dur_hit else 1.0,
        "speed_kn": float(speed_hit.group(1)) if speed_hit else 10.0,
        "boundary": _norm_boundary(str(entities.get("boundary", "TTW"))),
        "pollutants": _norm_pollutants(entities.get("pollutants")),
    }

    fuel_hit = _FUEL_RE.search(question or "")
    if fuel_hit:
        payload["fuel_type"] = fuel_hit.group(1).upper()
    mcr_hit = _MCR_RE.search(question or "")
    if mcr_hit:
        payload["mcr_kw"] = float(mcr_hit.group(1))
    ref_speed_hit = _REF_SPEED_RE.search(question or "")
    if ref_speed_hit:
        payload["ref_speed_kn"] = float(ref_speed_hit.group(1))

    return payload


def _port_filter(df: pd.DataFrame, port_token: Optional[str]) -> pd.DataFrame:
    if df.empty or not port_token:
        return df
    token = str(port_token).strip()
    code = re.sub(r"[^A-Z0-9]", "", token.upper())
    low = token.lower()
    mask = pd.Series(False, index=df.index)
    if "port_key" in df.columns:
        port_key = df["port_key"].fillna("").astype(str)
        port_key_upper = port_key.str.upper()
        port_key_norm = port_key_upper.str.replace(r"[^A-Z0-9]", "", regex=True)
        mask |= port_key_upper == code
        mask |= port_key_norm == code
        if code:
            mask |= port_key_norm.str.contains(code, regex=False)
        if low:
            mask |= port_key.str.lower().str.contains(low, regex=False)
    if "locode_norm" in df.columns:
        locode = df["locode_norm"].fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        mask |= locode == code
    if "port_label" in df.columns:
        port_label = df["port_label"].fillna("").astype(str)
        port_label_norm = port_label.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        if low:
            mask |= port_label.str.lower().str.contains(low, regex=False)
        if code:
            mask |= port_label_norm.str.contains(code, regex=False)
    return df[mask]


def _date_filter(df: pd.DataFrame, date_col: str, date_from: Optional[str], date_to: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
    if date_from:
        work = work[dates >= pd.Timestamp(date_from, tz="UTC")]
        dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
    if date_to:
        end_ts = pd.Timestamp(date_to, tz="UTC")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date_to).strip()):
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        work = work[dates <= end_ts]
    return work


def _sum_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df), dtype="float64"), index=df.index)


def _mean_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float(s.mean())


@dataclass
class CarbonResult:
    status: str
    answer: str
    table: Optional[pd.DataFrame]
    chart: Optional[pd.DataFrame]
    coverage_notes: List[str]
    caveats: List[str]
    boundary: str
    pollutants: List[str]
    source_label: str
    confidence_label: str
    confidence_reason: str
    uncertainty_interval: Dict[str, Dict[str, float]]
    params_version: str
    evidence_ids: List[str]
    segment_ids: List[str]
    result_state: str = CARBON_STATE_NOT_COMPUTABLE
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    export_csv_path: Optional[str] = None
    export_json_path: Optional[str] = None


class CarbonQueryEngine:
    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        factor_registry_path: str | Path = "config/carbon_factors.v1.json",
        monte_carlo_draws: int = 500,
        sanity_config: Optional[Dict[str, Any]] = None,
        auto_build: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.factor_registry_path = Path(factor_registry_path)
        self.monte_carlo_draws = int(monte_carlo_draws)
        self._daily_port: Optional[pd.DataFrame] = None
        self._calls: Optional[pd.DataFrame] = None
        self._segments: Optional[pd.DataFrame] = None
        self._evidence: Optional[pd.DataFrame] = None
        self._params: Optional[Dict[str, Any]] = None
        self.sanity_config = {
            "max_call_duration_h": 240.0,
            "max_call_tco2e": 500.0,
            "min_baseline_denominator_tco2e": 1.0,
        }
        if sanity_config:
            for key in list(self.sanity_config.keys()):
                if key in sanity_config:
                    try:
                        self.sanity_config[key] = float(sanity_config[key])
                    except Exception:
                        pass
        self.export_dir = self.processed_dir / "carbon_exports"
        self._runtime_export_fallback_dir = Path("/tmp/eagle_eye_carbon_exports")
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only runtimes (e.g., docker volume mounted :ro) should not fail query execution.
            pass
        if auto_build:
            self.ensure_built()

    def ensure_built(self) -> None:
        required = [
            self.processed_dir / "carbon_segments.parquet",
            self.processed_dir / "carbon_emissions_segment.parquet",
            self.processed_dir / "carbon_emissions_daily_port.parquet",
            self.processed_dir / "carbon_emissions_call.parquet",
            self.processed_dir / "carbon_evidence.parquet",
            self.processed_dir / "carbon_params_version.json",
        ]
        if all(p.exists() for p in required):
            return
        events_path = self.processed_dir / "events.parquet"
        if not events_path.exists():
            return
        build_carbon_layer(
            processed_dir=self.processed_dir,
            out_dir=self.processed_dir,
            factor_registry_path=self.factor_registry_path,
            monte_carlo_draws=self.monte_carlo_draws,
        )

    @property
    def available(self) -> bool:
        return (self.processed_dir / "carbon_emissions_daily_port.parquet").exists()

    @property
    def params_version(self) -> Dict[str, Any]:
        if self._params is not None:
            return self._params
        path = self.processed_dir / "carbon_params_version.json"
        if not path.exists():
            self._params = {}
            return self._params
        self._params = json.loads(path.read_text(encoding="utf-8"))
        return self._params

    @property
    def daily_port(self) -> pd.DataFrame:
        if self._daily_port is None:
            path = self.processed_dir / "carbon_emissions_daily_port.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
                self._daily_port = df
            else:
                self._daily_port = pd.DataFrame()
        return self._daily_port

    @property
    def calls(self) -> pd.DataFrame:
        if self._calls is None:
            path = self.processed_dir / "carbon_emissions_call.parquet"
            self._calls = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._calls

    @property
    def segments(self) -> pd.DataFrame:
        if self._segments is None:
            path = self.processed_dir / "carbon_emissions_segment.parquet"
            self._segments = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._segments

    @property
    def evidence(self) -> pd.DataFrame:
        if self._evidence is None:
            path = self.processed_dir / "carbon_evidence.parquet"
            self._evidence = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._evidence

    def _no_data(
        self,
        message: str,
        boundary: str = "TTW",
        pollutants: Optional[List[str]] = None,
        result_state: str = CARBON_STATE_NOT_COMPUTABLE,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> CarbonResult:
        caveats: List[str] = []
        if result_state in {CARBON_STATE_NOT_COMPUTABLE, CARBON_STATE_RETRIEVAL_ONLY}:
            caveats.append("Run `python -m src.carbon.build --processed_dir data/processed` to materialize carbon outputs.")
        if result_state == CARBON_STATE_FORECAST_ONLY:
            caveats.append("Carbon forecast mode is not implemented in this runtime; only deterministic inventory queries are supported.")
        if result_state == CARBON_STATE_UNSUPPORTED:
            caveats.append("This carbon request is outside the supported deterministic scope for current data artifacts.")
        if not caveats:
            caveats.append("Deterministic carbon output is unavailable for this scope.")

        return CarbonResult(
            status="no_data",
            answer=message,
            table=None,
            chart=None,
            coverage_notes=[message],
            caveats=caveats,
            boundary=boundary,
            pollutants=pollutants or ["CO2e"],
            source_label="Not computable from available carbon data",
            confidence_label="low",
            confidence_reason="Deterministic carbon computation unavailable for this scope.",
            uncertainty_interval={},
            params_version=str(self.params_version.get("version", "unknown")),
            evidence_ids=[],
            segment_ids=[],
            result_state=result_state,
            diagnostics=diagnostics or {},
        )

    def _build_uncertainty_summary(
        self,
        df: pd.DataFrame,
        pollutants: List[str],
        boundary: str,
    ) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for pol in pollutants:
            metric = _metric_column(pol, boundary)
            lower_col = f"{metric}_lower"
            upper_col = f"{metric}_upper"
            point = float(_numeric_series(df, metric).sum())
            lower = float(_numeric_series(df, lower_col).sum())
            upper = float(_numeric_series(df, upper_col).sum())
            summary[pol] = {"point": point, "lower": lower, "upper": upper}
        return summary

    def _filtered_segments_scope(
        self,
        port_id: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> pd.DataFrame:
        seg = self.segments.copy()
        if seg.empty:
            return seg
        if "timestamp_start" in seg.columns:
            seg["timestamp_start"] = pd.to_datetime(seg["timestamp_start"], errors="coerce", utc=True)
        seg = _port_filter(seg, port_id)
        if "timestamp_start" in seg.columns:
            seg = _date_filter(seg, "timestamp_start", date_from, date_to)
        return seg

    def _build_scope_diagnostics(
        self,
        seg_scope: pd.DataFrame,
        metric_col: str,
        baseline_values: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        if seg_scope is None or seg_scope.empty:
            return {
                "unique_vessel_calls": 0,
                "raw_rows_before_dedup": 0,
                "rows_after_dedup": 0,
                "duplicates_removed_rows": 0,
                "total_duration_hours": 0.0,
                "median_duration_hours": 0.0,
                "total_tco2e": 0.0,
                "mean_tco2e_per_call": None,
                "median_tco2e_per_call": None,
                "duplicated_call_ids_detected": 0,
                "warnings": ["No deterministic carbon segments matched."],
                "sanity_status": "unstable baseline",
                "min_baseline_denominator_tco2e": float(self.sanity_config["min_baseline_denominator_tco2e"]),
            }

        work = seg_scope.copy()
        call_work = work[work["call_id"].notna() & (work["call_id"].astype(str) != "")]
        calls = (
            call_work.groupby("call_id", dropna=False)[[metric_col, "duration_h"]]
            .sum(numeric_only=True)
            .reset_index()
            if not call_work.empty
            else pd.DataFrame(columns=["call_id", metric_col, "duration_h"])
        )
        unique_calls = int(calls["call_id"].nunique()) if not calls.empty else 0
        raw_rows = int(pd.to_numeric(work.get("raw_row_count"), errors="coerce").fillna(1.0).sum()) if "raw_row_count" in work.columns else int(len(work))
        dedup_rows = int(pd.to_numeric(work.get("row_count"), errors="coerce").fillna(1.0).sum()) if "row_count" in work.columns else int(len(work))
        total_duration_h = float(pd.to_numeric(work.get("duration_h"), errors="coerce").fillna(0.0).sum())
        call_durations = pd.to_numeric(calls.get("duration_h"), errors="coerce").dropna() if not calls.empty else pd.Series(dtype=float)
        total_t = float(pd.to_numeric(work.get(metric_col), errors="coerce").fillna(0.0).sum())
        call_vals = pd.to_numeric(calls.get(metric_col), errors="coerce").dropna() if not calls.empty else pd.Series(dtype=float)
        mean_call_t = float(call_vals.mean()) if not call_vals.empty else None
        med_call_t = float(call_vals.median()) if not call_vals.empty else None

        duplicated_call_ids = int(calls["call_id"].duplicated().sum()) if not calls.empty else 0
        warnings: List[str] = []
        if not call_durations.empty and float(call_durations.max()) > float(self.sanity_config["max_call_duration_h"]):
            warnings.append(
                f"Implausible call duration detected (> {self.sanity_config['max_call_duration_h']:.0f} h)."
            )
        if not call_vals.empty and float(call_vals.max()) > float(self.sanity_config["max_call_tco2e"]):
            warnings.append(
                f"Per-call emissions exceed threshold (> {self.sanity_config['max_call_tco2e']:.1f} tCO2e)."
            )
        if duplicated_call_ids > 0:
            warnings.append("Duplicated call IDs detected in aggregated call scope.")
        if baseline_values is not None:
            base_series = pd.to_numeric(pd.Series(list(baseline_values), dtype="float64"), errors="coerce").dropna()
            if base_series.empty or float(base_series.median()) < float(self.sanity_config["min_baseline_denominator_tco2e"]):
                warnings.append("Baseline denominator is too small for meaningful percentage comparison.")

        sanity = "checked"
        if warnings:
            sanity = "warning"
        if any("Baseline denominator" in w for w in warnings):
            sanity = "unstable baseline"
        if any("Duplicated call IDs" in w for w in warnings):
            sanity = "possible duplication"

        return {
            "unique_vessel_calls": unique_calls,
            "raw_rows_before_dedup": raw_rows,
            "rows_after_dedup": dedup_rows,
            "duplicates_removed_rows": max(0, raw_rows - dedup_rows),
            "total_duration_hours": round(total_duration_h, 3),
            "median_duration_hours": round(float(call_durations.median()), 3) if not call_durations.empty else 0.0,
            "total_tco2e": total_t,
            "mean_tco2e_per_call": mean_call_t,
            "median_tco2e_per_call": med_call_t,
            "duplicated_call_ids_detected": duplicated_call_ids,
            "warnings": warnings,
            "sanity_status": sanity,
            "min_baseline_denominator_tco2e": float(self.sanity_config["min_baseline_denominator_tco2e"]),
        }

    def _aggregate_port_scope_from_segments(
        self,
        seg_scope: pd.DataFrame,
        metric_cols: List[str],
        group_by: str,
        include_uncertainty: bool,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if seg_scope.empty:
            return pd.DataFrame(), None

        work = seg_scope.copy()
        work["date"] = pd.to_datetime(work["timestamp_start"], errors="coerce", utc=True).dt.floor("D")
        work = work.dropna(subset=["date"])
        if work.empty:
            return pd.DataFrame(), None

        if group_by.lower() in {"month", "monthly"}:
            work["bucket"] = work["date"].dt.to_period("M").astype(str)
            group_cols = ["bucket", "port_key", "port_label", "locode_norm"]
        else:
            work["bucket"] = work["date"]
            group_cols = ["bucket", "port_key", "port_label", "locode_norm"]

        agg_map: Dict[str, str] = {m: "sum" for m in metric_cols if m in work.columns}
        if "row_count" in work.columns:
            agg_map["row_count"] = "sum"
        if "duration_h" in work.columns:
            agg_map["duration_h"] = "sum"
        if "fallback_usage_ratio" in work.columns:
            agg_map["fallback_usage_ratio"] = "mean"
        if "ci_width_rel" in work.columns:
            agg_map["ci_width_rel"] = "mean"
        if "confidence_reason" in work.columns:
            agg_map["confidence_reason"] = "first"
        if include_uncertainty:
            for m in metric_cols:
                low_col = f"{m}_lower"
                up_col = f"{m}_upper"
                if low_col in work.columns:
                    agg_map[low_col] = "sum"
                if up_col in work.columns:
                    agg_map[up_col] = "sum"

        table = (
            work.groupby(group_cols, dropna=False)
            .agg(agg_map)
            .reset_index()
            .rename(columns={"bucket": "date"})
            .sort_values("date")
        )

        # recompute confidence rows from aggregated uncertainty width + fallback usage.
        point_col = "wtw_co2e_t" if "wtw_co2e_t" in table.columns else ("ttw_co2e_t" if "ttw_co2e_t" in table.columns else "co2_t")
        low_col = f"{point_col}_lower"
        up_col = f"{point_col}_upper"
        if low_col in table.columns and up_col in table.columns:
            p = pd.to_numeric(table[point_col], errors="coerce").fillna(0.0)
            lo = pd.to_numeric(table[low_col], errors="coerce").fillna(0.0)
            up = pd.to_numeric(table[up_col], errors="coerce").fillna(0.0)
            table["ci_width_rel"] = np.where(p > 0, (up - lo) / p, 1.0)

        labels: List[str] = []
        reasons: List[str] = []
        ci_series = _numeric_series(table, "ci_width_rel").fillna(1.0)
        fallback_series = _numeric_series(table, "fallback_usage_ratio").fillna(0.0)
        for ci, fb in zip(
            ci_series,
            fallback_series,
        ):
            ci_f = float(ci)
            fb_f = float(fb)
            if ci_f <= 0.20 and fb_f <= 0.05:
                labels.append("high")
            elif ci_f <= 0.40 or fb_f <= 0.20:
                labels.append("medium")
            else:
                labels.append("low")
            reasons.append(
                f"CI width={ci_f:.2f}, fallback_ratio={fb_f:.2f}, aggregated from deterministic carbon rows."
            )
        table["confidence_label"] = labels
        table["confidence_reason"] = reasons

        chart_col = metric_cols[0] if metric_cols else None
        chart: Optional[pd.DataFrame] = None
        if chart_col and chart_col in table.columns:
            if group_by.lower() in {"month", "monthly"}:
                chart = table.groupby("date", dropna=False)[chart_col].sum().reset_index().set_index("date")
            else:
                chart = table.groupby("date", dropna=False)[chart_col].sum().reset_index().set_index("date")
        return table, chart

    def _build_call_trace_payload(
        self,
        call_id: str,
        mmsi: str,
        boundary: str,
    ) -> Dict[str, Any]:
        seg = self.segments.copy()
        if seg.empty:
            return {}
        seg["call_id"] = seg["call_id"].fillna("").astype(str)
        seg["mmsi"] = seg["mmsi"].fillna("").astype(str)
        seg = seg[(seg["call_id"] == str(call_id)) & (seg["mmsi"] == str(mmsi))]
        if seg.empty:
            return {}

        seg["timestamp_start"] = pd.to_datetime(seg["timestamp_start"], errors="coerce", utc=True)
        seg["timestamp_end"] = pd.to_datetime(seg["timestamp_end"], errors="coerce", utc=True)
        arrival = seg["timestamp_start"].min()
        departure = seg["timestamp_end"].max()
        duration_h = float(pd.to_numeric(seg.get("duration_h"), errors="coerce").fillna(0.0).sum())
        metric = "wtw_co2e_t" if boundary == "WTW" else "ttw_co2e_t"
        total_t = _sum_col(seg, metric)

        call_counts = self.calls.copy()
        call_counts["call_id"] = call_counts.get("call_id", "").fillna("").astype(str)
        counted_once = int((call_counts["call_id"] == str(call_id)).sum()) == 1 if not call_counts.empty else False

        return {
            "call_id": call_id,
            "mmsi": mmsi,
            "arrival_utc": arrival.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(arrival) else None,
            "departure_utc": departure.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(departure) else None,
            "duration_hours": duration_h,
            "vessel_class": str(seg["vessel_class"].dropna().iloc[0]) if "vessel_class" in seg.columns and seg["vessel_class"].notna().any() else "unknown",
            "proxy_class": str(seg["vessel_class"].dropna().iloc[0]) if "vessel_class" in seg.columns and seg["vessel_class"].notna().any() else "unknown",
            "fuel_t": _sum_col(seg, "fuel_t"),
            "ttw_co2_t": _sum_col(seg, "co2_t"),
            "wtt_co2e_t": _sum_col(seg, "wtt_co2e_t"),
            "ch4_co2e_t": 0.0,
            "n2o_co2e_t": 0.0,
            "final_total_tco2e_t": total_t,
            "final_total_kgco2e": total_t * 1000.0,
            "counted_once": counted_once,
            "segment_count": int(len(seg)),
        }

    def _write_exports(self, prefix: str, table: pd.DataFrame, payload: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        candidate_dirs = [self.export_dir, self._runtime_export_fallback_dir]

        def _json_default(value: Any) -> Any:
            if isinstance(value, pd.Timestamp):
                return value.strftime("%Y-%m-%dT%H:%M:%SZ")
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            return str(value)

        for out_dir in candidate_dirs:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = out_dir / f"{prefix}_{stamp}.csv"
                json_path = out_dir / f"{prefix}_{stamp}.json"
                table.to_csv(csv_path, index=False)
                json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
                return str(csv_path), str(json_path)
            except OSError:
                continue
        return None, None

    def query_port_emissions(
        self,
        port_id: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        group_by: str = "day",
        boundary: str = "TTW",
        pollutants: Optional[Sequence[str]] = None,
        include_uncertainty: bool = True,
        include_evidence: bool = True,
    ) -> CarbonResult:
        boundary = _norm_boundary(boundary)
        pollutants_list = _norm_pollutants(pollutants)
        if date_from and date_to:
            from_ts = pd.to_datetime(date_from, errors="coerce", utc=True)
            to_ts = pd.to_datetime(date_to, errors="coerce", utc=True)
            if pd.notna(from_ts) and pd.notna(to_ts) and from_ts > to_ts:
                return self._no_data("Invalid date range: from > to.", boundary=boundary, pollutants=pollutants_list)
        if not self.available:
            return self._no_data(
                "Carbon outputs are not available.",
                boundary=boundary,
                pollutants=pollutants_list,
                result_state=CARBON_STATE_NOT_COMPUTABLE,
            )

        seg_scope_all = self._filtered_segments_scope(port_id=port_id, date_from=date_from, date_to=date_to)
        metric_cols = [_metric_column(pol, boundary) for pol in pollutants_list]
        metric_primary = metric_cols[0] if metric_cols else ("wtw_co2e_t" if boundary == "WTW" else "ttw_co2e_t")
        daily_scope = _port_filter(self.daily_port.copy(), port_id)
        daily_scope = _date_filter(daily_scope, "date", date_from, date_to)

        if seg_scope_all.empty and daily_scope.empty:
            return self._no_data(
                "No carbon segment rows matched the requested port/date filters.",
                boundary=boundary,
                pollutants=pollutants_list,
                result_state=CARBON_STATE_NOT_COMPUTABLE,
                diagnostics=self._build_scope_diagnostics(pd.DataFrame(), metric_col=metric_primary),
            )
        if "segment_id" in seg_scope_all.columns:
            seg_scope_all = seg_scope_all.drop_duplicates(subset=["segment_id"], keep="first")

        # Deterministic inventory must be call-linked; destination-only proxy segments are not used as numeric truth.
        deterministic = (
            seg_scope_all[
                seg_scope_all["call_id"].notna() & (seg_scope_all["call_id"].astype(str) != "")
            ].copy()
            if not seg_scope_all.empty
            else pd.DataFrame()
        )

        baseline_scope = _port_filter(self.daily_port.copy(), port_id)
        baseline_values = (
            pd.to_numeric(baseline_scope.get(metric_primary), errors="coerce").dropna().tolist()
            if metric_primary in baseline_scope.columns
            else []
        )
        used_proxy_daily = False
        if deterministic.empty:
            if daily_scope.empty:
                diagnostics = self._build_scope_diagnostics(
                    seg_scope_all,
                    metric_col=metric_primary,
                    baseline_values=baseline_values,
                )
                diagnostics["reason"] = "No call-linked segments matched and no daily proxy rows matched."
                return self._no_data(
                    "No deterministic carbon rows matched the requested scope.",
                    boundary=boundary,
                    pollutants=pollutants_list,
                    result_state=CARBON_STATE_NOT_COMPUTABLE,
                    diagnostics=diagnostics,
                )
            deterministic = daily_scope.copy()
            used_proxy_daily = True

        if used_proxy_daily:
            work = deterministic.copy()
            work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True).dt.floor("D")
            work = work.dropna(subset=["date"])
            if group_by.lower() in {"month", "monthly"}:
                work["bucket"] = work["date"].dt.to_period("M").astype(str)
            else:
                work["bucket"] = work["date"]

            group_cols = ["bucket", "port_key", "port_label", "locode_norm"]
            agg_map: Dict[str, str] = {m: "sum" for m in metric_cols if m in work.columns}
            if "row_count" in work.columns:
                agg_map["row_count"] = "sum"
            if "segments" in work.columns:
                agg_map["segments"] = "sum"
            if "duration_h" in work.columns:
                agg_map["duration_h"] = "sum"
            if "fallback_usage_ratio" in work.columns:
                agg_map["fallback_usage_ratio"] = "mean"
            if "ci_width_rel" in work.columns:
                agg_map["ci_width_rel"] = "mean"
            if "confidence_reason" in work.columns:
                agg_map["confidence_reason"] = "first"
            if include_uncertainty:
                for metric in metric_cols:
                    low_col = f"{metric}_lower"
                    up_col = f"{metric}_upper"
                    if low_col in work.columns:
                        agg_map[low_col] = "sum"
                    if up_col in work.columns:
                        agg_map[up_col] = "sum"

            table = (
                work.groupby(group_cols, dropna=False)
                .agg(agg_map)
                .reset_index()
                .rename(columns={"bucket": "date"})
                .sort_values("date")
            )
            chart = None
            if metric_primary in table.columns:
                chart = (
                    table.groupby("date", dropna=False)[metric_primary]
                    .sum()
                    .reset_index()
                    .set_index("date")
                )
        else:
            table, chart = self._aggregate_port_scope_from_segments(
                seg_scope=deterministic,
                metric_cols=metric_cols,
                group_by=group_by,
                include_uncertainty=include_uncertainty,
            )
        if table.empty:
            diagnostics = self._build_scope_diagnostics(
                deterministic,
                metric_col=metric_primary,
                baseline_values=baseline_values,
            )
            return self._no_data(
                "Deterministic carbon aggregation returned no rows for this scope.",
                boundary=boundary,
                pollutants=pollutants_list,
                result_state=CARBON_STATE_NOT_COMPUTABLE,
                diagnostics=diagnostics,
            )

        uncertainty = self._build_uncertainty_summary(deterministic, pollutants_list, boundary) if include_uncertainty else {}
        total_co2e_key = "CO2e" if "CO2e" in uncertainty else pollutants_list[0]
        total_point = float(uncertainty.get(total_co2e_key, {}).get("point", 0.0))
        total_low = float(uncertainty.get(total_co2e_key, {}).get("lower", 0.0))
        total_up = float(uncertainty.get(total_co2e_key, {}).get("upper", 0.0))
        ci_width_rel = (total_up - total_low) / total_point if total_point > 0 else 0.0
        fallback_ratio = _mean_col(deterministic, "fallback_usage_ratio")
        if used_proxy_daily:
            source_label = "Computed from AIS-derived daily carbon inventory (proxy-based, no call-link in scope)"
        else:
            source_label = (
                "Computed with fallback defaults"
                if fallback_ratio > 0.15
                else "Computed from AIS + port-call segmentation"
            )
        conf = (
            "high"
            if ci_width_rel <= 0.20 and fallback_ratio <= 0.05
            else "medium"
            if ci_width_rel <= 0.40 or fallback_ratio <= 0.20
            else "low"
        )
        if used_proxy_daily and conf == "high":
            conf = "medium"
        conf_reason = (
            f"CI width={ci_width_rel:.2f}, fallback_ratio={fallback_ratio:.2f}, "
            f"rows={len(deterministic):,}, group_by={group_by}, mode={'daily_proxy' if used_proxy_daily else 'call_linked'}."
        )
        result_state = CARBON_STATE_COMPUTED_ZERO if abs(total_point) < 1e-12 else CARBON_STATE_COMPUTED

        diagnostics = self._build_scope_diagnostics(
            seg_scope_all if used_proxy_daily else deterministic,
            metric_col=metric_primary,
            baseline_values=baseline_values,
        )
        diagnostics["result_state"] = result_state
        diagnostics["deterministic_mode"] = "daily_proxy" if used_proxy_daily else "call_linked"
        diagnostics["reconciliation_total_tco2e"] = total_point
        if used_proxy_daily:
            diagnostics["reconciliation_total_from_unique_calls_tco2e"] = None
            diagnostics["reconciliation_unique_call_count"] = int(
                seg_scope_all["call_id"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()
            )
        else:
            diagnostics["reconciliation_total_from_unique_calls_tco2e"] = _sum_col(
                deterministic.groupby("call_id", dropna=False)[metric_primary].sum().reset_index(),
                metric_primary,
            )
            diagnostics["reconciliation_unique_call_count"] = int(
                deterministic["call_id"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()
            )

        evidence_ids: List[str] = []
        segment_ids: List[str] = []
        if include_evidence and not self.evidence.empty:
            seg_ids_source = seg_scope_all if used_proxy_daily else deterministic
            seg_ids_all = set(seg_ids_source["segment_id"].astype(str).tolist()) if "segment_id" in seg_ids_source.columns else set()
            edf = _port_filter(self.evidence, port_id)
            edf = _date_filter(edf, "timestamp_start", date_from, date_to)
            if seg_ids_all and "segment_id" in edf.columns:
                edf = edf[edf["segment_id"].astype(str).isin(seg_ids_all)]
            if not edf.empty and "evidence_id" in edf.columns:
                evidence_ids = edf["evidence_id"].astype(str).head(50).tolist()
                if "segment_id" in edf.columns:
                    segment_ids = edf["segment_id"].astype(str).head(50).tolist()

        port_label = port_id or "the selected scope"
        if used_proxy_daily:
            answer = (
                f"{boundary} emissions for {port_label} were computed from deterministic AIS-derived daily inventory "
                f"(proxy-based, no call-link rows in this scope). "
                f"Total {total_co2e_key}={total_point:.2f} tCO2e ({total_low:.2f}-{total_up:.2f} tCO2e)."
            )
        else:
            answer = (
                f"{boundary} emissions for {port_label} were computed from deterministic call-linked segmentation. "
                f"Total {total_co2e_key}={total_point:.2f} tCO2e ({total_low:.2f}-{total_up:.2f} tCO2e)."
            )
        coverage = [
            (
                f"Coverage daily rows: {len(deterministic):,}"
                if used_proxy_daily
                else f"Coverage segments: {len(deterministic):,}"
            ),
            f"Unique vessel-calls: {diagnostics.get('unique_vessel_calls', 0):,}",
            f"Boundary: {boundary}",
            f"Pollutants: {', '.join(pollutants_list)}",
            f"Group by: {group_by}",
            "Source label: " + source_label,
            f"Fallback usage ratio: {fallback_ratio:.2f}",
            (
                "Reconciliation: total displayed emissions equal the sum of unique call-linked emissions."
                if not used_proxy_daily
                else "Reconciliation: total displayed emissions equal the sum of deterministic daily proxy rows in scope."
            ),
            (
                "Reconciliation: intensity denominator uses unique vessel-call count."
                if not used_proxy_daily
                else "Reconciliation: intensity denominator uses unique vessel-call count when call-link rows are available."
            ),
            "Unit standard: absolute greenhouse-gas values are expressed in tCO2e.",
        ]
        caveats = [
            "Confidence expresses evidence/assumption strength, not certainty.",
            "Carbon estimates are deterministic but proxy-based inventory outputs, not direct stack measurements.",
            "Dataset-relative thresholds are used unless external regulatory thresholds are configured.",
        ]
        for w in diagnostics.get("warnings", [])[:4]:
            caveats.append(f"Sanity warning: {w}")

        payload = {
            "boundary": boundary,
            "pollutants": pollutants_list,
            "source_label": source_label,
            "result_state": result_state,
            "confidence_label": conf,
            "confidence_reason": conf_reason,
            "uncertainty_interval": uncertainty,
            "params_version": str(self.params_version.get("version", "unknown")),
            "evidence_ids": evidence_ids,
            "segment_ids": segment_ids,
            "diagnostics": diagnostics,
            "units": {
                "absolute_emissions": "tCO2e",
                "intensity_examples": ["kgCO2e/vessel-call", "tCO2e/day", "kgCO2e/hour"],
                "time": "UTC (24-hour format)",
                "distance": "nautical miles (nm)",
                "speed": "knots (kn)",
            },
            "rows": table.to_dict(orient="records"),
        }
        export_csv, export_json = self._write_exports("carbon_port_emissions", table, payload)
        if not export_csv or not export_json:
            caveats.append("Export files were skipped because runtime storage is read-only.")

        return CarbonResult(
            status="ok",
            answer=answer,
            table=table,
            chart=chart,
            coverage_notes=coverage,
            caveats=caveats,
            boundary=boundary,
            pollutants=pollutants_list,
            source_label=source_label,
            confidence_label=conf,
            confidence_reason=conf_reason,
            uncertainty_interval=uncertainty,
            params_version=str(self.params_version.get("version", "unknown")),
            evidence_ids=evidence_ids,
            segment_ids=segment_ids,
            result_state=result_state,
            diagnostics=diagnostics,
            export_csv_path=export_csv,
            export_json_path=export_json,
        )

    def query_vessel_call(
        self,
        mmsi: str,
        call_id: str,
        boundary: str = "TTW",
        pollutants: Optional[Sequence[str]] = None,
        include_uncertainty: bool = True,
        include_evidence: bool = True,
    ) -> CarbonResult:
        boundary = _norm_boundary(boundary)
        pollutants_list = _norm_pollutants(pollutants)
        target_mmsi = str(mmsi).strip()
        target_call_id = _normalize_call_id(call_id)
        target_call_canon = _canonical_call_id(target_call_id)
        if not self.available:
            return self._no_data(
                "Carbon outputs are not available.",
                boundary=boundary,
                pollutants=pollutants_list,
                result_state=CARBON_STATE_NOT_COMPUTABLE,
            )

        calls = self.calls.copy()
        if calls.empty:
            return self._no_data(
                "No call-level carbon table available.",
                boundary=boundary,
                pollutants=pollutants_list,
                result_state=CARBON_STATE_NOT_COMPUTABLE,
            )
        calls["mmsi"] = calls["mmsi"].fillna("").astype(str)
        calls["call_id"] = calls["call_id"].fillna("").astype(str)
        calls["call_id_norm"] = calls["call_id"].map(_normalize_call_id)
        calls["call_id_canon"] = calls["call_id_norm"].map(_canonical_call_id)
        mmsi_scope = calls[calls["mmsi"] == target_mmsi]
        exact = mmsi_scope[mmsi_scope["call_id_norm"] == target_call_id]
        if not exact.empty:
            work = exact.copy()
        else:
            canon_matches = mmsi_scope[mmsi_scope["call_id_canon"] == target_call_canon].copy()
            if canon_matches.empty:
                work = pd.DataFrame()
            else:
                unique_norm = canon_matches["call_id_norm"].dropna().astype(str).unique().tolist()
                if len(unique_norm) > 1:
                    return self._no_data(
                        "Ambiguous call_id match after normalization; provide full call_id as stored in carbon_emissions_call.",
                        boundary=boundary,
                        pollutants=pollutants_list,
                        result_state=CARBON_STATE_NOT_COMPUTABLE,
                        diagnostics={
                            "result_state": CARBON_STATE_NOT_COMPUTABLE,
                            "sanity_status": "warning",
                            "warnings": [f"Ambiguous canonical call_id match: {len(unique_norm)} candidates."],
                            "call_id_input": target_call_id,
                            "canonical_call_id": target_call_canon,
                            "candidate_call_ids": unique_norm[:10],
                        },
                    )
                work = canon_matches.copy()
        if work.empty:
            return self._no_data(
                "No matching call_id/mmsi carbon rows found.",
                boundary=boundary,
                pollutants=pollutants_list,
                result_state=CARBON_STATE_NOT_COMPUTABLE,
            )
        matched_call_id = str(work["call_id"].iloc[0]).strip()

        metric_cols = [_metric_column(pol, boundary) for pol in pollutants_list]
        metric_primary = metric_cols[0] if metric_cols else ("wtw_co2e_t" if boundary == "WTW" else "ttw_co2e_t")
        cols = ["call_id", "mmsi", "port_key", "port_label", "locode_norm", "confidence_label", "confidence_reason"] + metric_cols
        if include_uncertainty:
            for metric in metric_cols:
                cols.extend([f"{metric}_lower", f"{metric}_upper"])
        cols = [c for c in cols if c in work.columns]
        table = work[cols].copy()
        chart = None

        uncertainty = self._build_uncertainty_summary(work, pollutants_list, boundary) if include_uncertainty else {}
        ci_width_rel = float(pd.to_numeric(work.get("ci_width_rel"), errors="coerce").fillna(1.0).mean())
        fallback_ratio = float(pd.to_numeric(work.get("fallback_usage_ratio"), errors="coerce").fillna(0.0).mean())
        source_label = (
            "Computed with fallback defaults"
            if fallback_ratio > 0.15
            else "Computed from AIS + port-call segmentation"
        )
        conf = (
            "high"
            if ci_width_rel <= 0.20 and fallback_ratio <= 0.05
            else "medium"
            if ci_width_rel <= 0.40 or fallback_ratio <= 0.20
            else "low"
        )
        conf_reason = f"CI width={ci_width_rel:.2f}, fallback_ratio={fallback_ratio:.2f}."
        total_co2e_key = "CO2e" if "CO2e" in uncertainty else (pollutants_list[0] if pollutants_list else "CO2e")
        total_point = float(uncertainty.get(total_co2e_key, {}).get("point", _sum_col(work, metric_primary)))
        result_state = CARBON_STATE_COMPUTED_ZERO if abs(total_point) < 1e-12 else CARBON_STATE_COMPUTED

        evidence_ids: List[str] = []
        segment_ids: List[str] = []
        if include_evidence and not self.evidence.empty:
            ev = self.evidence[
                (self.evidence["mmsi"].astype(str) == target_mmsi)
                & (
                    self.evidence["call_id"].fillna("").astype(str).map(_normalize_call_id)
                    == _normalize_call_id(matched_call_id)
                )
            ]
            if not ev.empty:
                evidence_ids = ev["evidence_id"].astype(str).head(30).tolist()
                if "segment_id" in ev.columns:
                    segment_ids = ev["segment_id"].astype(str).head(30).tolist()

        seg_scope = self.segments.copy()
        if not seg_scope.empty:
            seg_scope["mmsi"] = seg_scope["mmsi"].fillna("").astype(str)
            seg_scope["call_id"] = seg_scope["call_id"].fillna("").astype(str)
            seg_scope = seg_scope[
                (seg_scope["mmsi"] == target_mmsi)
                & (seg_scope["call_id"].map(_normalize_call_id) == _normalize_call_id(matched_call_id))
            ]
            if "segment_id" in seg_scope.columns:
                seg_scope = seg_scope.drop_duplicates(subset=["segment_id"], keep="first")

        baseline_scope = self.daily_port.copy()
        if not baseline_scope.empty and "port_key" in baseline_scope.columns and "port_key" in work.columns:
            port_key = str(work["port_key"].iloc[0]) if pd.notna(work["port_key"].iloc[0]) else ""
            if port_key:
                baseline_scope = baseline_scope[baseline_scope["port_key"].astype(str) == port_key]
        baseline_values = (
            pd.to_numeric(baseline_scope.get(metric_primary), errors="coerce").dropna().tolist()
            if metric_primary in baseline_scope.columns
            else []
        )
        diagnostics = self._build_scope_diagnostics(
            seg_scope=seg_scope,
            metric_col=metric_primary,
            baseline_values=baseline_values,
        )
        diagnostics["trace_single_call"] = self._build_call_trace_payload(
            call_id=matched_call_id,
            mmsi=target_mmsi,
            boundary=boundary,
        )
        diagnostics["result_state"] = result_state
        diagnostics["reconciliation_total_tco2e"] = total_point
        diagnostics["reconciliation_total_from_unique_calls_tco2e"] = _sum_col(work, metric_primary)
        diagnostics["reconciliation_unique_call_count"] = 1
        diagnostics["call_id_input"] = target_call_id
        diagnostics["call_id_matched"] = matched_call_id

        answer = (
            f"{boundary} emissions for call `{matched_call_id}` (MMSI {target_mmsi}) were computed deterministically "
            "with greenhouse-gas values reported as tCO2e."
        )
        payload = {
            "boundary": boundary,
            "pollutants": pollutants_list,
            "source_label": source_label,
            "result_state": result_state,
            "confidence_label": conf,
            "confidence_reason": conf_reason,
            "uncertainty_interval": uncertainty,
            "params_version": str(self.params_version.get("version", "unknown")),
            "evidence_ids": evidence_ids,
            "segment_ids": segment_ids,
            "diagnostics": diagnostics,
            "units": {
                "absolute_emissions": "tCO2e",
                "intensity_examples": ["kgCO2e/vessel-call", "tCO2e/day", "kgCO2e/hour"],
                "time": "UTC (24-hour format)",
                "distance": "nautical miles (nm)",
                "speed": "knots (kn)",
            },
            "rows": table.to_dict(orient="records"),
        }
        export_csv, export_json = self._write_exports("carbon_vessel_call", table, payload)
        caveats = [
            "Confidence reflects inventory evidence quality and assumption strength.",
            "Operational recommendations should be combined with local fuel and engine records when available.",
        ]
        if not export_csv or not export_json:
            caveats.append("Export files were skipped because runtime storage is read-only.")
        return CarbonResult(
            status="ok",
            answer=answer,
            table=table,
            chart=chart,
            coverage_notes=[
                f"Rows used: {len(work)}",
                "Unique vessel-calls: 1",
                f"Boundary: {boundary}",
                f"Fallback usage ratio: {fallback_ratio:.2f}",
                "Unit standard: absolute greenhouse-gas values are expressed in tCO2e.",
            ],
            caveats=caveats,
            boundary=boundary,
            pollutants=pollutants_list,
            source_label=source_label,
            confidence_label=conf,
            confidence_reason=conf_reason,
            uncertainty_interval=uncertainty,
            params_version=str(self.params_version.get("version", "unknown")),
            evidence_ids=evidence_ids,
            segment_ids=segment_ids,
            result_state=result_state,
            diagnostics=diagnostics,
            export_csv_path=export_csv,
            export_json_path=export_json,
        )

    def estimate_with_assumptions(self, payload: Dict[str, Any]) -> CarbonResult:
        registry = load_factor_registry(self.factor_registry_path)
        vessel_type = str(payload.get("vessel_type", "unknown")).strip().lower()
        vessel_class = registry.resolve_vessel_class(vessel_type)
        mode = str(payload.get("mode", "transit")).strip().lower()
        duration_h = max(0.0, float(payload.get("duration_h", 1.0)))
        speed_kn = max(0.0, float(payload.get("speed_kn", 10.0)))
        boundary = _norm_boundary(str(payload.get("boundary", "TTW")))
        pollutants = _norm_pollutants(payload.get("pollutants"))

        defaults = registry.vessel_defaults(vessel_class)
        mcr_kw = float(payload.get("mcr_kw") or defaults.get("mcr_kw", 8000))
        ref_speed = float(payload.get("ref_speed_kn") or defaults.get("ref_speed_kn", 12.0))
        fuel = str(payload.get("fuel_type", defaults.get("fuel", "MGO"))).upper()
        engine = str(payload.get("engine_family", defaults.get("engine_family", "medium_speed_diesel")))
        aux_kw = float(payload.get("aux_power_kw") or registry.mode_aux_power_kw(mode, vessel_class))

        lf = np.clip((speed_kn / ref_speed) ** 3, 0.2, 1.0) if mode in {"transit", "manoeuvring"} else 0.0
        p_main = mcr_kw * lf
        fuel_t = (
            (
                p_main * registry.mode_sfc_main(mode)
                + aux_kw * registry.mode_sfc_aux(mode)
            )
            * duration_h
            / 1_000_000.0
        )
        fuel_f = registry.fuel_factors(fuel)
        co2_t = fuel_t * fuel_f["co2_t_per_t_fuel"]
        nox_kg = fuel_t * registry.nox_factor(engine, mode)
        pm_kg = fuel_t * registry.pm_factor(engine, mode)
        sox_kg = fuel_t * registry.mode_sulfur_fraction(mode) * 1000.0 * float(registry.assumptions.get("sox_multiplier", 2.0))
        ttw_co2e_t = co2_t
        wtt_co2e_t = fuel_t * fuel_f["wtt_co2e_t_per_t_fuel"]
        wtw_co2e_t = ttw_co2e_t + wtt_co2e_t

        row = pd.DataFrame(
            [
                {
                    "scenario": "explicit_assumptions",
                    "fuel_t": fuel_t,
                    "co2_t": co2_t,
                    "nox_kg": nox_kg,
                    "sox_kg": sox_kg,
                    "pm_kg": pm_kg,
                    "ttw_co2e_t": ttw_co2e_t,
                    "wtt_co2e_t": wtt_co2e_t,
                    "wtw_co2e_t": wtw_co2e_t,
                }
            ]
        )
        uncertainty: Dict[str, Dict[str, float]] = {}
        rel = 0.30
        for pol in pollutants:
            metric = _metric_column(pol, boundary)
            point = float(row[metric].iloc[0])
            uncertainty[pol] = {
                "point": point,
                "lower": max(0.0, point * (1.0 - 1.96 * rel)),
                "upper": point * (1.0 + 1.96 * rel),
            }

        point_tco2e = float(wtw_co2e_t if boundary == "WTW" else ttw_co2e_t)
        answer = (
            f"Carbon estimate computed for {mode} mode ({boundary}) using explicit assumptions. "
            f"Fuel={fuel_t:.3f} t, CO2e={point_tco2e:.3f} tCO2e."
        )
        payload_out = {
            "boundary": boundary,
            "pollutants": pollutants,
            "source_label": (
                "Computed from explicit scenario assumptions (estimated, with fallback vessel defaults)"
                if vessel_class == "unknown"
                else "Computed from explicit scenario assumptions (estimated)"
            ),
            "confidence_label": "medium",
            "confidence_reason": "Scenario estimate with user-specified assumptions and default uncertainty envelope.",
            "uncertainty_interval": uncertainty,
            "params_version": str(self.params_version.get("version", "unknown")),
            "evidence_ids": [],
            "units": {
                "absolute_emissions": "tCO2e",
                "intensity_examples": ["kgCO2e/vessel-call", "tCO2e/day", "kgCO2e/hour"],
                "time": "UTC (24-hour format)",
                "distance": "nautical miles (nm)",
                "speed": "knots (kn)",
            },
            "rows": row.to_dict(orient="records"),
            "assumptions": payload,
        }
        export_csv, export_json = self._write_exports("carbon_estimate", row, payload_out)
        caveats = ["Estimate is scenario-based and not a direct measured inventory."]
        if not export_csv or not export_json:
            caveats.append("Export files were skipped because runtime storage is read-only.")
        computed_state = (
            CARBON_STATE_COMPUTED_ZERO
            if abs(float(row["wtw_co2e_t"].iloc[0] if boundary == "WTW" else row["ttw_co2e_t"].iloc[0])) < 1e-12
            else CARBON_STATE_COMPUTED
        )
        return CarbonResult(
            status="ok",
            answer=answer,
            table=row,
            chart=None,
            coverage_notes=[
                "Mode: " + mode,
                "Boundary: " + boundary,
                "Assumptions provided explicitly.",
                "Unit standard: absolute greenhouse-gas values are expressed in tCO2e.",
            ],
            caveats=caveats,
            boundary=boundary,
            pollutants=pollutants,
            source_label=payload_out["source_label"],
            confidence_label="medium",
            confidence_reason=payload_out["confidence_reason"],
            uncertainty_interval=uncertainty,
            params_version=str(self.params_version.get("version", "unknown")),
            evidence_ids=[],
            segment_ids=[],
            result_state=computed_state,
            diagnostics={
                "result_state": computed_state,
                "sanity_status": "checked",
                "warnings": [],
                "trace_assumptions": payload,
            },
            export_csv_path=export_csv,
            export_json_path=export_json,
        )

    def get_evidence(self, evidence_id: str) -> Dict[str, Any]:
        if self.evidence.empty:
            return {"status": "no_data", "reason": "carbon_evidence.parquet is missing."}
        match = self.evidence[self.evidence["evidence_id"].astype(str) == str(evidence_id)]
        if match.empty:
            return {"status": "no_data", "reason": f"No evidence row found for evidence_id={evidence_id}."}
        row = match.iloc[0].to_dict()
        for key, value in list(row.items()):
            if isinstance(value, pd.Timestamp):
                row[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
            elif isinstance(value, (np.integer,)):
                row[key] = int(value)
            elif isinstance(value, (np.floating,)):
                row[key] = float(value)
        return {"status": "ok", "evidence": row}

    def from_question_entities(
        self,
        question: str,
        entities: Dict[str, Any],
        user_filters: Dict[str, Any],
        resolved_scope: Optional[Dict[str, Any]] = None,
    ) -> CarbonResult:
        q = question.lower()
        boundary = _norm_boundary(str(entities.get("boundary", "TTW")))
        pollutants = _norm_pollutants(entities.get("pollutants"))
        resolved_scope = dict(resolved_scope or {})
        port = resolved_scope.get("port") or user_filters.get("port") or entities.get("port")
        date_from = resolved_scope.get("date_from") or user_filters.get("date_from") or entities.get("date_from")
        date_to = resolved_scope.get("date_to") or user_filters.get("date_to") or entities.get("date_to")

        call_id = _normalize_call_id(str(entities.get("call_id") or ""))
        mmsi = str(entities.get("mmsi") or "").strip()
        if call_id and mmsi:
            return self.query_vessel_call(
                mmsi=mmsi,
                call_id=call_id,
                boundary=boundary,
                pollutants=pollutants,
                include_uncertainty=True,
                include_evidence=True,
            )

        estimate_payload = _extract_estimate_payload(question, entities)
        if estimate_payload is not None:
            return self.estimate_with_assumptions(estimate_payload)

        if any(token in q for token in ("forecast", "predict", "expected", "future", "next", "coming", "will")) and not (
            date_from or date_to
        ):
            return self._no_data(
                "Carbon forecast was requested, but deterministic carbon forecast outputs are not available in this runtime.",
                boundary=boundary,
                pollutants=pollutants,
                result_state=CARBON_STATE_FORECAST_ONLY,
                diagnostics={
                    "result_state": CARBON_STATE_FORECAST_ONLY,
                    "sanity_status": "unstable baseline",
                    "warnings": [
                        "Forecast-only carbon request: no deterministic carbon forecast model available.",
                    ],
                },
            )

        group_by = "month" if ("monthly" in q or "per month" in q) else "day"
        return self.query_port_emissions(
            port_id=str(port) if port else None,
            date_from=date_from,
            date_to=date_to,
            group_by=group_by,
            boundary=boundary,
            pollutants=pollutants,
            include_uncertainty=True,
            include_evidence=True,
        )


def extract_carbon_call_id(question: str) -> Optional[str]:
    m = re.search(r"\bcall[_\-\s]?id[\s:=_\-]*([A-Za-z0-9_\-:.]+)\b", question, flags=re.IGNORECASE)
    if not m:
        return None
    return _normalize_call_id(m.group(1))
