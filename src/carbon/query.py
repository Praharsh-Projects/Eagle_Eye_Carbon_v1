"""Carbon query engine for TTW/WTW analytics, uncertainty, and provenance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.carbon.build import build_carbon_layer
from src.carbon.factors import load_factor_registry


SUPPORTED_POLLUTANTS = ["CO2", "CO2e", "NOx", "SOx", "PM"]


def _norm_boundary(value: str) -> str:
    token = (value or "TTW").strip().upper()
    if token in {"WTW", "WELL_TO_WAKE", "WELL-TO-WAKE"}:
        return "WTW"
    return "TTW"


def _norm_pollutants(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return ["CO2", "NOx", "SOx", "PM"]
    out: List[str] = []
    for item in values:
        token = str(item).strip().upper()
        if token in {"CO2", "NOX", "SOX", "PM", "CO2E"}:
            fixed = token.replace("NOX", "NOx").replace("SOX", "SOx").replace("CO2E", "CO2e")
            if fixed not in out:
                out.append(fixed)
    return out or ["CO2", "NOx", "SOx", "PM"]


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


def _port_filter(df: pd.DataFrame, port_token: Optional[str]) -> pd.DataFrame:
    if df.empty or not port_token:
        return df
    token = str(port_token).strip()
    code = token.upper().replace(" ", "")
    low = token.lower()
    mask = pd.Series(False, index=df.index)
    if "port_key" in df.columns:
        mask |= df["port_key"].fillna("").astype(str).str.upper() == code
    if "locode_norm" in df.columns:
        mask |= df["locode_norm"].fillna("").astype(str).str.upper() == code
    if "port_label" in df.columns:
        mask |= df["port_label"].fillna("").astype(str).str.lower().str.contains(low, regex=False)
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
        work = work[dates <= pd.Timestamp(date_to, tz="UTC")]
    return work


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
    export_csv_path: Optional[str] = None
    export_json_path: Optional[str] = None


class CarbonQueryEngine:
    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        factor_registry_path: str | Path = "config/carbon_factors.v1.json",
        monte_carlo_draws: int = 500,
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
        self.export_dir = self.processed_dir / "carbon_exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
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

    def _no_data(self, message: str, boundary: str = "TTW", pollutants: Optional[List[str]] = None) -> CarbonResult:
        return CarbonResult(
            status="no_data",
            answer="I don't have evidence in the dataset to answer that carbon query.",
            table=None,
            chart=None,
            coverage_notes=[message],
            caveats=["Run `python -m src.carbon.build --processed_dir data/processed` to materialize carbon outputs."],
            boundary=boundary,
            pollutants=pollutants or ["CO2"],
            source_label="Computed with fallback defaults",
            confidence_label="low",
            confidence_reason="Missing carbon artifacts for deterministic computation.",
            uncertainty_interval={},
            params_version=str(self.params_version.get("version", "unknown")),
            evidence_ids=[],
            segment_ids=[],
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
            point = float(pd.to_numeric(df.get(metric, 0), errors="coerce").fillna(0).sum())
            lower = float(pd.to_numeric(df.get(lower_col, 0), errors="coerce").fillna(0).sum())
            upper = float(pd.to_numeric(df.get(upper_col, 0), errors="coerce").fillna(0).sum())
            summary[pol] = {"point": point, "lower": lower, "upper": upper}
        return summary

    def _write_exports(self, prefix: str, table: pd.DataFrame, payload: Dict[str, Any]) -> tuple[str, str]:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        csv_path = self.export_dir / f"{prefix}_{stamp}.csv"
        json_path = self.export_dir / f"{prefix}_{stamp}.json"
        table.to_csv(csv_path, index=False)

        def _json_default(value: Any) -> Any:
            if isinstance(value, pd.Timestamp):
                return value.strftime("%Y-%m-%dT%H:%M:%SZ")
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            return str(value)

        json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return str(csv_path), str(json_path)

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
            return self._no_data("Carbon outputs are not available.", boundary=boundary, pollutants=pollutants_list)

        work = _port_filter(self.daily_port, port_id)
        work = _date_filter(work, "date", date_from, date_to)
        if work.empty:
            return self._no_data("No carbon rows matched the requested port/date filters.", boundary, pollutants_list)

        metric_cols = [_metric_column(pol, boundary) for pol in pollutants_list]
        table_cols = ["date", "port_key", "port_label", "locode_norm", "confidence_label", "confidence_reason"] + metric_cols
        if include_uncertainty:
            for metric in metric_cols:
                table_cols.extend([f"{metric}_lower", f"{metric}_upper"])
        table_cols = [c for c in table_cols if c in work.columns]
        table = work[table_cols].copy().sort_values("date")

        if group_by.lower() in {"month", "monthly"}:
            month = pd.to_datetime(table["date"], errors="coerce", utc=True).dt.to_period("M").astype(str)
            agg_map = {m: "sum" for m in metric_cols}
            if include_uncertainty:
                for m in metric_cols:
                    agg_map[f"{m}_lower"] = "sum"
                    agg_map[f"{m}_upper"] = "sum"
            agg = table.assign(month=month).groupby("month", dropna=False).agg(agg_map).reset_index()
            chart = agg.set_index("month")[[metric_cols[0]]] if metric_cols else None
            table = agg
        else:
            chart = table.set_index("date")[[metric_cols[0]]] if metric_cols and "date" in table.columns else None

        uncertainty = self._build_uncertainty_summary(work, pollutants_list, boundary) if include_uncertainty else {}
        total_co2e_key = "CO2e" if "CO2e" in uncertainty else pollutants_list[0]
        total_point = float(uncertainty.get(total_co2e_key, {}).get("point", 0.0))
        total_low = float(uncertainty.get(total_co2e_key, {}).get("lower", 0.0))
        total_up = float(uncertainty.get(total_co2e_key, {}).get("upper", 0.0))
        ci_width_rel = (total_up - total_low) / total_point if total_point > 0 else 1.0
        fallback_ratio = float(pd.to_numeric(work.get("fallback_usage_ratio"), errors="coerce").fillna(0).mean())
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
        conf_reason = (
            f"CI width={ci_width_rel:.2f}, fallback_ratio={fallback_ratio:.2f}, "
            f"rows={len(work):,}, group_by={group_by}."
        )

        evidence_ids: List[str] = []
        segment_ids: List[str] = []
        if include_evidence and not self.evidence.empty:
            edf = _port_filter(self.evidence, port_id)
            edf = _date_filter(edf, "timestamp_start", date_from, date_to)
            if not edf.empty and "evidence_id" in edf.columns:
                evidence_ids = edf["evidence_id"].astype(str).head(25).tolist()
                if "segment_id" in edf.columns:
                    segment_ids = edf["segment_id"].astype(str).head(25).tolist()

        port_label = port_id or "the selected scope"
        answer = (
            f"{boundary} emissions for {port_label} were computed from deterministic segmentation. "
            f"Total {total_co2e_key}={total_point:.2f} ({total_low:.2f}-{total_up:.2f})."
        )
        coverage = [
            f"Coverage rows: {len(work):,}",
            f"Boundary: {boundary}",
            f"Pollutants: {', '.join(pollutants_list)}",
            f"Group by: {group_by}",
            "Source label: " + source_label,
            f"Fallback usage ratio: {fallback_ratio:.2f}",
        ]
        caveats = [
            "Confidence expresses evidence/assumption strength, not certainty.",
            "Carbon estimates use deterministic heuristics with mode segmentation and local factor pack.",
        ]

        payload = {
            "boundary": boundary,
            "pollutants": pollutants_list,
            "source_label": source_label,
            "confidence_label": conf,
            "confidence_reason": conf_reason,
            "uncertainty_interval": uncertainty,
            "params_version": str(self.params_version.get("version", "unknown")),
            "evidence_ids": evidence_ids,
            "segment_ids": segment_ids,
            "rows": table.to_dict(orient="records"),
        }
        export_csv, export_json = self._write_exports("carbon_port_emissions", table, payload)

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
        if not self.available:
            return self._no_data("Carbon outputs are not available.", boundary=boundary, pollutants=pollutants_list)

        calls = self.calls.copy()
        if calls.empty:
            return self._no_data("No call-level carbon table available.", boundary, pollutants_list)
        calls["mmsi"] = calls["mmsi"].fillna("").astype(str)
        calls["call_id"] = calls["call_id"].fillna("").astype(str)
        work = calls[(calls["mmsi"] == str(mmsi)) & (calls["call_id"] == str(call_id))]
        if work.empty:
            return self._no_data("No matching call_id/mmsi carbon rows found.", boundary, pollutants_list)

        metric_cols = [_metric_column(pol, boundary) for pol in pollutants_list]
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

        evidence_ids: List[str] = []
        segment_ids: List[str] = []
        if include_evidence and not self.evidence.empty:
            ev = self.evidence[
                (self.evidence["mmsi"].astype(str) == str(mmsi))
                & (self.evidence["call_id"].fillna("").astype(str) == str(call_id))
            ]
            if not ev.empty:
                evidence_ids = ev["evidence_id"].astype(str).head(30).tolist()
                if "segment_id" in ev.columns:
                    segment_ids = ev["segment_id"].astype(str).head(30).tolist()

        answer = f"{boundary} emissions for call `{call_id}` (MMSI {mmsi}) were computed deterministically."
        payload = {
            "boundary": boundary,
            "pollutants": pollutants_list,
            "source_label": source_label,
            "confidence_label": conf,
            "confidence_reason": conf_reason,
            "uncertainty_interval": uncertainty,
            "params_version": str(self.params_version.get("version", "unknown")),
            "evidence_ids": evidence_ids,
            "segment_ids": segment_ids,
            "rows": table.to_dict(orient="records"),
        }
        export_csv, export_json = self._write_exports("carbon_vessel_call", table, payload)
        return CarbonResult(
            status="ok",
            answer=answer,
            table=table,
            chart=chart,
            coverage_notes=[
                f"Rows used: {len(work)}",
                f"Boundary: {boundary}",
                f"Fallback usage ratio: {fallback_ratio:.2f}",
            ],
            caveats=[
                "Confidence reflects inventory evidence quality and assumption strength.",
                "Operational recommendations should be combined with local fuel and engine records when available.",
            ],
            boundary=boundary,
            pollutants=pollutants_list,
            source_label=source_label,
            confidence_label=conf,
            confidence_reason=conf_reason,
            uncertainty_interval=uncertainty,
            params_version=str(self.params_version.get("version", "unknown")),
            evidence_ids=evidence_ids,
            segment_ids=segment_ids,
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

        answer = (
            f"Carbon estimate computed for {mode} mode ({boundary}) using explicit assumptions. "
            f"Fuel={fuel_t:.3f} t, CO2={co2_t:.3f} t."
        )
        payload_out = {
            "boundary": boundary,
            "pollutants": pollutants,
            "source_label": "Computed with fallback defaults" if vessel_class == "unknown" else "Computed from AIS + port-call segmentation",
            "confidence_label": "medium",
            "confidence_reason": "Scenario estimate with user-specified assumptions and default uncertainty envelope.",
            "uncertainty_interval": uncertainty,
            "params_version": str(self.params_version.get("version", "unknown")),
            "evidence_ids": [],
            "rows": row.to_dict(orient="records"),
            "assumptions": payload,
        }
        export_csv, export_json = self._write_exports("carbon_estimate", row, payload_out)
        return CarbonResult(
            status="ok",
            answer=answer,
            table=row,
            chart=None,
            coverage_notes=["Mode: " + mode, "Boundary: " + boundary, "Assumptions provided explicitly."],
            caveats=["Estimate is scenario-based and not a direct measured inventory."],
            boundary=boundary,
            pollutants=pollutants,
            source_label=payload_out["source_label"],
            confidence_label="medium",
            confidence_reason=payload_out["confidence_reason"],
            uncertainty_interval=uncertainty,
            params_version=str(self.params_version.get("version", "unknown")),
            evidence_ids=[],
            segment_ids=[],
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
    ) -> CarbonResult:
        q = question.lower()
        boundary = _norm_boundary(str(entities.get("boundary", "TTW")))
        pollutants = _norm_pollutants(entities.get("pollutants"))
        port = user_filters.get("port") or entities.get("port")
        date_from = user_filters.get("date_from") or entities.get("date_from")
        date_to = user_filters.get("date_to") or entities.get("date_to")

        call_id = str(entities.get("call_id") or "").strip()
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
    m = re.search(r"\bcall[_\-\s]?id\s*[:=]?\s*([A-Za-z0-9_\-:.]+)\b", question, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()
