"""Shared carbon/emissions presentation helpers (units, thresholds, findings, suggestions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_THRESHOLD_PERCENTILES = (0.25, 0.50, 0.75)


@dataclass
class ScaledValue:
    value: float
    unit: str


@dataclass
class ThresholdBands:
    p25: float
    p50: float
    p75: float
    source_label: str


@dataclass
class ChartFinding:
    timestamp: pd.Timestamp
    value: float
    finding: str
    kind: str


def sanitize_threshold_percentiles(raw: Any) -> Tuple[float, float, float]:
    try:
        vals = tuple(float(x) for x in list(raw))
        if len(vals) != 3:
            return DEFAULT_THRESHOLD_PERCENTILES
        p25, p50, p75 = vals
        if not (0.0 < p25 < p50 < p75 < 1.0):
            return DEFAULT_THRESHOLD_PERCENTILES
        return vals
    except Exception:
        return DEFAULT_THRESHOLD_PERCENTILES


def scale_tco2e(value_tco2e: float) -> ScaledValue:
    value = float(value_tco2e)
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return ScaledValue(value=value / 1_000_000.0, unit="MtCO2e")
    if abs_val >= 1_000:
        return ScaledValue(value=value / 1_000.0, unit="ktCO2e")
    return ScaledValue(value=value, unit="tCO2e")


def format_tco2e(value_tco2e: float, decimals: int = 2) -> str:
    scaled = scale_tco2e(value_tco2e)
    return f"{scaled.value:,.{decimals}f} {scaled.unit}"


def format_kgco2e(value_kgco2e: float, decimals: int = 1) -> str:
    return f"{float(value_kgco2e):,.{decimals}f} kgCO2e"


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}%"


def safe_percent_delta(
    current_value: Optional[float],
    baseline_value: Optional[float],
    min_denominator: float = 1.0,
) -> Optional[float]:
    if current_value is None or baseline_value is None:
        return None
    try:
        curr = float(current_value)
        base = float(baseline_value)
    except Exception:
        return None
    if abs(base) < float(min_denominator):
        return None
    return ((curr - base) / base) * 100.0


def format_hours(value_hours: float, decimals: int = 1) -> str:
    return f"{float(value_hours):,.{decimals}f} h"


def format_knots(value_knots: float, decimals: int = 1) -> str:
    return f"{float(value_knots):,.{decimals}f} kn"


def format_nautical_miles(value_nm: float, decimals: int = 1) -> str:
    return f"{float(value_nm):,.{decimals}f} nm"


def format_utc_timestamp(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return "n/a"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


def emissions_measurement_note(unit: str) -> str:
    if unit in {"tCO2e", "ktCO2e", "MtCO2e"}:
        return "Measured in tonnes of carbon dioxide equivalent (tCO2e)."
    if unit == "kgCO2e/vessel-call":
        return "Measured as kilograms of carbon dioxide equivalent per vessel-call (kgCO2e/vessel-call)."
    if unit == "tCO2e/day":
        return "Measured as tonnes of carbon dioxide equivalent per day (tCO2e/day)."
    if unit == "kgCO2e/hour":
        return "Measured as kilograms of carbon dioxide equivalent per hour (kgCO2e/hour)."
    if unit == "tCO2e/forecast-window":
        return "Measured as tonnes of carbon dioxide equivalent per forecast window (tCO2e/forecast-window)."
    return "Measured in standardized carbon/emissions units."


def derive_threshold_bands(
    values: Sequence[float],
    percentiles: Tuple[float, float, float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> ThresholdBands:
    clean = pd.to_numeric(pd.Series(list(values), dtype="float64"), errors="coerce").dropna()
    if clean.empty:
        return ThresholdBands(p25=0.0, p50=0.0, p75=0.0, source_label="relative to this dataset")

    p25, p50, p75 = np.quantile(clean.to_numpy(), list(percentiles))
    return ThresholdBands(
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        source_label="relative to this dataset",
    )


def classify_level(value: float, bands: ThresholdBands) -> str:
    val = float(value)
    if val <= bands.p25:
        return "Low"
    if val <= bands.p50:
        return "Moderate"
    if val <= bands.p75:
        return "High"
    return "Very High"


def build_comparison_bar_table(
    current_value: float,
    bands: ThresholdBands,
) -> pd.DataFrame:
    max_ref = max(float(current_value), bands.p75, 1.0)
    very_high_end = max_ref * 1.15
    return pd.DataFrame(
        [
            {"level": "Low", "start": 0.0, "end": bands.p25, "color": "#22c55e"},
            {"level": "Moderate", "start": bands.p25, "end": bands.p50, "color": "#84cc16"},
            {"level": "High", "start": bands.p50, "end": bands.p75, "color": "#f59e0b"},
            {"level": "Very High", "start": bands.p75, "end": very_high_end, "color": "#ef4444"},
        ]
    )


def _pick_value_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["wtw_co2e_t", "ttw_co2e_t", "co2_t"]:
        if col in df.columns:
            return col
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def extract_chart_findings(
    chart_df: pd.DataFrame,
    target_ts: Optional[pd.Timestamp] = None,
    max_findings: int = 5,
) -> List[ChartFinding]:
    if chart_df is None or chart_df.empty:
        return []

    work = chart_df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "date" in work.columns:
            work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
            work = work.dropna(subset=["date"]).set_index("date")
        else:
            return []

    value_col = _pick_value_col(work.reset_index())
    if not value_col or value_col not in work.columns:
        return []

    values = pd.to_numeric(work[value_col], errors="coerce").dropna()
    if values.empty:
        return []

    items: List[ChartFinding] = []

    ts_high = values.idxmax()
    items.append(
        ChartFinding(
            timestamp=pd.Timestamp(ts_high),
            value=float(values.loc[ts_high]),
            finding="Finding: Highest emissions in this window.",
            kind="highest",
        )
    )

    ts_low = values.idxmin()
    if ts_low != ts_high:
        items.append(
            ChartFinding(
                timestamp=pd.Timestamp(ts_low),
                value=float(values.loc[ts_low]),
                finding="Finding: Lowest emissions in this window.",
                kind="lowest",
            )
        )

    delta = values.diff().dropna()
    if not delta.empty:
        ts_spike = delta.idxmax()
        if float(delta.loc[ts_spike]) > 0:
            items.append(
                ChartFinding(
                    timestamp=pd.Timestamp(ts_spike),
                    value=float(values.loc[ts_spike]),
                    finding="Finding: Sharp increase versus previous period.",
                    kind="spike",
                )
            )
        ts_drop = delta.idxmin()
        if float(delta.loc[ts_drop]) < 0 and ts_drop != ts_spike:
            items.append(
                ChartFinding(
                    timestamp=pd.Timestamp(ts_drop),
                    value=float(values.loc[ts_drop]),
                    finding="Finding: Largest drop versus previous period.",
                    kind="drop",
                )
            )

    if target_ts is not None:
        target = pd.Timestamp(target_ts).tz_convert("UTC") if pd.Timestamp(target_ts).tzinfo else pd.Timestamp(target_ts, tz="UTC")
        nearest_idx = (values.index.to_series() - target).abs().idxmin()
        items.append(
            ChartFinding(
                timestamp=pd.Timestamp(nearest_idx),
                value=float(values.loc[nearest_idx]),
                finding="Finding: Current selected period.",
                kind="selected",
            )
        )

    dedup: List[ChartFinding] = []
    seen: set[tuple[str, pd.Timestamp]] = set()
    for item in items:
        key = (item.kind, pd.Timestamp(item.timestamp))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= max_findings:
            break
    return dedup


def build_emissions_findings(
    current_tco2e: float,
    level: str,
    change_vs_median_pct: Optional[float],
    source_label: str,
    ci_width_rel: Optional[float],
    chart_findings: Sequence[ChartFinding],
) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    findings.append(
        {
            "type": "deterministic",
            "text": f"Emissions level is {level.lower()} relative to this dataset baseline.",
        }
    )
    if change_vs_median_pct is not None:
        direction = "above" if change_vs_median_pct >= 0 else "below"
        findings.append(
            {
                "type": "deterministic",
                "text": f"Current emissions are {abs(change_vs_median_pct):.1f}% {direction} the historical median for this scope.",
            }
        )
    if ci_width_rel is not None and ci_width_rel > 0.40:
        findings.append(
            {
                "type": "inferred",
                "text": "Uncertainty interval is wide; treat this as an estimated inventory rather than a precise measurement.",
            }
        )
    if "fallback" in source_label.lower():
        findings.append(
            {
                "type": "inferred",
                "text": "Computation used fallback defaults; interpret absolute values conservatively.",
            }
        )
    if chart_findings:
        findings.append({"type": "deterministic", "text": chart_findings[0].finding.replace("Finding: ", "")})
    return findings[:5]


def build_reduction_suggestions(
    level: str,
    change_vs_median_pct: Optional[float],
    ci_width_rel: Optional[float],
    source_label: str,
) -> List[str]:
    suggestions: List[str] = []

    if level in {"High", "Very High"}:
        suggestions.append("Reduce anchorage waiting by using staggered arrival windows and tighter ETA coordination.")
        suggestions.append("Apply berth-allocation smoothing on peak days to avoid simultaneous vessel clustering.")
        suggestions.append("Prioritize shore-power or idle-engine reduction for long berth stays where available.")
    else:
        suggestions.append("Keep baseline operating plan and monitor daily emissions drift against historical median.")
        suggestions.append("Use vessel-type specific windows to prevent localized traffic bunching.")
        suggestions.append("Review berth stay patterns weekly and target long idle windows for operational adjustment.")

    if change_vs_median_pct is not None and change_vs_median_pct > 20:
        suggestions.append("Pre-activate mitigation measures before high-pressure windows to prevent escalation.")
    if "fallback" in source_label.lower():
        suggestions.append("Prioritize data-quality improvements (fuel/engine metadata) to reduce dependency on fallback factors.")
    if ci_width_rel is not None and ci_width_rel > 0.40:
        suggestions.append("Validate assumptions with latest AIS coverage before acting on aggressive mitigation changes.")

    deduped: List[str] = []
    for item in suggestions:
        if item not in deduped:
            deduped.append(item)
    if len(deduped) < 3:
        for filler in [
            "Track emissions weekly against baseline and escalate only when sustained drift is observed.",
            "Coordinate vessel arrival sequencing with pilots to reduce simultaneous queue pressure.",
            "Re-run this assessment with tighter filters (port/date/vessel type) before high-impact interventions.",
        ]:
            if filler not in deduped:
                deduped.append(filler)
            if len(deduped) >= 3:
                break
    return deduped[:5]


def to_emissions_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    table = df.copy()

    rename_map = {
        "co2_t": "CO2 (tCO2e)",
        "ttw_co2e_t": "TTW CO2e (tCO2e)",
        "wtt_co2e_t": "WTT CO2e (tCO2e)",
        "wtw_co2e_t": "WTW CO2e (tCO2e)",
        "nox_kg": "NOx (kg)",
        "sox_kg": "SOx (kg)",
        "pm_kg": "PM (kg)",
        "duration_h": "Duration (h)",
        "date": "Date (UTC)",
    }
    table = table.rename(columns={k: v for k, v in rename_map.items() if k in table.columns})
    if "Date (UTC)" in table.columns:
        table["Date (UTC)"] = pd.to_datetime(table["Date (UTC)"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    return table


def compute_emissions_metrics(
    result_table: Optional[pd.DataFrame],
    boundary: str,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "total_tco2e": None,
        "intensity_kg_per_call": None,
        "tco2e_per_day": None,
        "kgco2e_per_hour": None,
        "calls_count": None,
        "days_count": None,
        "hours_total": None,
    }
    if result_table is None or result_table.empty:
        return out

    metric_col = "wtw_co2e_t" if boundary == "WTW" else "ttw_co2e_t"
    if metric_col not in result_table.columns:
        metric_col = "co2_t" if "co2_t" in result_table.columns else metric_col
    if metric_col not in result_table.columns:
        return out

    total_tco2e = float(pd.to_numeric(result_table[metric_col], errors="coerce").fillna(0).sum())
    out["total_tco2e"] = total_tco2e

    calls_count = len(result_table["call_id"].dropna().unique()) if "call_id" in result_table.columns else len(result_table)
    out["calls_count"] = float(calls_count) if calls_count else None
    if calls_count:
        out["intensity_kg_per_call"] = (total_tco2e * 1000.0) / float(calls_count)

    days_count = len(result_table["date"].dropna().unique()) if "date" in result_table.columns else None
    out["days_count"] = float(days_count) if days_count else None
    if days_count:
        out["tco2e_per_day"] = total_tco2e / float(days_count)

    if "duration_h" in result_table.columns:
        hours_total = float(pd.to_numeric(result_table["duration_h"], errors="coerce").fillna(0).sum())
        out["hours_total"] = hours_total
        if hours_total > 0:
            out["kgco2e_per_hour"] = (total_tco2e * 1000.0) / hours_total

    return out
