"""FastAPI deployment path for Eagle Eye."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.carbon.query import (
    CARBON_STATE_COMPUTED,
    CARBON_STATE_COMPUTED_ZERO,
    CARBON_STATE_FORECAST_ONLY,
    CARBON_STATE_NOT_COMPUTABLE,
    CARBON_STATE_RETRIEVAL_ONLY,
    CARBON_STATE_UNSUPPORTED,
    CarbonQueryEngine,
    CarbonResult,
)
from src.carbon.presentation import (
    build_emissions_findings,
    build_reduction_suggestions,
    classify_level,
    compute_emissions_metrics,
    derive_threshold_bands,
    extract_chart_findings,
    format_percent,
    format_tco2e,
    safe_percent_delta,
    sanitize_threshold_percentiles,
)
from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.qa.intent import IntentResult, classify_question
from src.rag.retriever import QueryFilters, RAGRetriever
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet
from src.utils.cloud_bootstrap import ensure_bundle, ensure_file_manifest
from src.utils.config import load_config
from src.utils.runtime import chroma_remote_settings, force_local_vector_env
from src.utils.serialization import compact_traffic_evidence


class AskFiltersPayload(BaseModel):
    port: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    vessel_type: Optional[str] = None
    anomaly: Optional[bool] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k_evidence: int = Field(default=5, ge=1, le=10)
    filters: AskFiltersPayload = Field(default_factory=AskFiltersPayload)


class CarbonEstimateRequest(BaseModel):
    vessel_type: Optional[str] = None
    mode: str = "transit"
    duration_h: float = Field(default=1.0, gt=0.0, le=240.0)
    speed_kn: float = Field(default=10.0, ge=0.0, le=60.0)
    mcr_kw: Optional[float] = Field(default=None, gt=0.0)
    ref_speed_kn: Optional[float] = Field(default=None, gt=0.0)
    aux_power_kw: Optional[float] = Field(default=None, ge=0.0)
    fuel_type: Optional[str] = None
    engine_family: Optional[str] = None
    boundary: str = "TTW"
    pollutants: Optional[List[str]] = None


@dataclass
class EvidenceBundle:
    lines: List[str]
    rows: List[Dict[str, Any]]
    trace: Dict[str, Any]


def _resolve_processed_dir(preferred_dir: Path) -> tuple[Path, bool]:
    if (preferred_dir / "arrivals_daily.parquet").exists():
        return preferred_dir, False
    fallback = Path("demo_data/processed")
    if (fallback / "arrivals_daily.parquet").exists():
        return fallback, True
    return preferred_dir, False


def _resolve_persist_dir(preferred_dir: Path) -> tuple[Path, bool]:
    if (preferred_dir / "chroma.sqlite3").exists():
        return preferred_dir, False
    fallback = Path("demo_data/chroma")
    if (fallback / "chroma.sqlite3").exists():
        return fallback, True
    return preferred_dir, False


def _pick_filter(override: Optional[str], extracted: Optional[str]) -> Optional[str]:
    if override is not None and str(override).strip():
        return str(override).strip()
    if extracted is not None and str(extracted).strip():
        return str(extracted).strip()
    return None


def _load_runtime_setting(name: str) -> str:
    return str(os.getenv(name, "")).strip()


def _maybe_bootstrap_bundle(
    env_name: str,
    target_dir: Path,
    required_files: List[str],
) -> tuple[bool, str]:
    if all((target_dir / rel_path).exists() for rel_path in required_files):
        return False, f"{env_name} assets already exist in {target_dir}."
    bundle_url = _load_runtime_setting(env_name)
    if not bundle_url:
        return False, f"No {env_name} configured."
    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=target_dir,
        required_files=required_files,
    )
    return changed, message


def _maybe_bootstrap_chroma_runtime(target_dir: Path) -> tuple[bool, str]:
    required_files = ["chroma.sqlite3", "traffic_metadata_index.csv"]
    if all((target_dir / name).exists() for name in required_files):
        return False, f"Chroma runtime assets already exist in {target_dir}."

    manifest_url = _load_runtime_setting("APP_CHROMA_MANIFEST_URL")
    if manifest_url:
        return ensure_file_manifest(
            url=manifest_url,
            target_dir=target_dir,
            required_files=required_files,
            timeout_seconds=3600,
        )

    bundle_url = _load_runtime_setting("APP_CHROMA_BUNDLE_URL")
    if bundle_url:
        return ensure_bundle(
            url=bundle_url,
            target_dir=target_dir,
            required_files=required_files,
            timeout_seconds=3600,
        )

    return False, "No APP_CHROMA_MANIFEST_URL or APP_CHROMA_BUNDLE_URL configured."


def _init_retriever(
    persist_dir: Path,
    config_path: str,
    force_local_vector: bool = False,
) -> RAGRetriever:
    if force_local_vector:
        with force_local_vector_env():
            return RAGRetriever(persist_dir=persist_dir, config_path=config_path)
    return RAGRetriever(persist_dir=persist_dir, config_path=config_path)


def _make_rag_filters(
    entities: Dict[str, Any],
    overrides: Dict[str, Any],
    include_dates: bool = True,
) -> QueryFilters:
    port_token = _pick_filter(overrides.get("port"), entities.get("port"))
    locode = None
    destination = None
    port_name = None
    if port_token:
        import re

        if re.fullmatch(r"[A-Za-z]{2}\s?[A-Za-z]{3}", port_token):
            locode = port_token
        else:
            destination = port_token
            port_name = port_token

    date_from = _pick_filter(overrides.get("date_from"), entities.get("date_from")) if include_dates else None
    date_to = _pick_filter(overrides.get("date_to"), entities.get("date_to")) if include_dates else None

    return QueryFilters(
        mmsi=entities.get("mmsi"),
        imo=entities.get("imo"),
        locode=locode,
        port_name=port_name,
        destination=destination,
        vessel_type=_pick_filter(overrides.get("vessel_type"), entities.get("vessel_type")),
        date_from=date_from,
        date_to=date_to,
    )


def _retrieve_evidence_api(
    retriever: Optional[RAGRetriever],
    retriever_reason: str,
    question: str,
    entities: Dict[str, Any],
    overrides: Dict[str, Any],
    top_k: int,
    include_dates: bool,
) -> EvidenceBundle:
    if retriever is None:
        return EvidenceBundle(
            lines=[],
            rows=[],
            trace={
                "retrieval_status": "disabled",
                "reason": retriever_reason or "Retriever unavailable.",
            },
        )

    import time

    try:
        filters = _make_rag_filters(entities=entities, overrides=overrides, include_dates=include_dates)
        started = time.perf_counter()
        result = retriever.query_traffic(question=question, filters=filters, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        return EvidenceBundle(
            lines=[],
            rows=[],
            trace={"retrieval_status": "error", "reason": f"Vector retrieval failed: {exc}"},
        )

    lines: List[str] = []
    rows: List[Dict[str, Any]] = []
    for item in result.evidence[:top_k]:
        chunk_id = str(item.metadata.get("chunk_id") or item.metadata.get("stable_id") or item.metadata.get("id") or "")
        dist_txt = f"{float(item.distance):.4f}" if item.distance is not None else "n/a"
        lines.append(
            f"vector_id={item.id} | chunk_id={chunk_id or 'n/a'} | dist={dist_txt} | "
            f"{compact_traffic_evidence(item.metadata, item.text)}"
        )
        rows.append(
            {
                "vector_id": item.id,
                "chunk_id": chunk_id or None,
                "distance": item.distance,
                "timestamp": item.metadata.get("timestamp_full") or item.metadata.get("date"),
                "port": item.metadata.get("locode_norm")
                or item.metadata.get("locode")
                or item.metadata.get("port_name")
                or item.metadata.get("destination_norm"),
                "vessel_type": item.metadata.get("vessel_type_norm") or item.metadata.get("vessel_type"),
                "mmsi": item.metadata.get("mmsi"),
            }
        )

    trace = {
        "retrieval_status": "ok" if rows else "no_hits",
        "reason": "Vector rows retrieved successfully." if rows else "No vector rows matched the query and filters.",
        "collection": retriever.config["index"]["traffic_collection"],
        "mode": result.mode,
        "vector_backend": getattr(retriever, "vector_backend", "unknown"),
        "query_latency_ms": round(latency_ms, 2),
        "returned_items": len(result.evidence),
        "top_k_requested": top_k,
        "where_filter": result.where_filter,
    }
    return EvidenceBundle(lines=lines, rows=rows, trace=trace)


def _fallback_evidence_from_result(
    value: Union[AnalyticsResult, ForecastResult, CarbonResult],
    max_items: int = 5,
) -> List[str]:
    lines: List[str] = []
    if isinstance(value, CarbonResult):
        for eid in (value.evidence_ids or [])[:max_items]:
            lines.append(f"carbon_evidence_id={eid}")
        if value.table is not None and not value.table.empty:
            head = value.table.head(min(3, max_items))
            for _, row in head.iterrows():
                tokens = []
                for col in head.columns[:4]:
                    cell = row[col]
                    if pd.isna(cell):
                        continue
                    tokens.append(f"{col}={cell}")
                if tokens:
                    lines.append(" | ".join(tokens))
        return lines[:max_items]

    if isinstance(value, ForecastResult):
        anchor_values_note = next((n for n in value.coverage_notes if n.startswith("Analog values used:")), None)
        anchor_dates_note = next((n for n in value.coverage_notes if n.startswith("Analog dates used:")), None)
        if anchor_values_note:
            lines.append(anchor_values_note)
        elif anchor_dates_note:
            lines.append(anchor_dates_note)
        if value.history is not None and not value.history.empty and {"date", "actual"}.issubset(value.history.columns):
            hist = value.history.copy()
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=True).dt.floor("D")
            hist = hist.dropna(subset=["date", "actual"]).sort_values("date")
            for _, row in hist.tail(max_items).iterrows():
                lines.append(f"Historical point | {row['date'].strftime('%Y-%m-%d')} | value={float(row['actual']):.2f}")
        return lines[:max_items]

    if value.table is not None and not value.table.empty:
        table = value.table.head(max_items).copy()
        for _, row in table.iterrows():
            parts: List[str] = []
            for col in table.columns[:4]:
                cell = row[col]
                if pd.isna(cell):
                    continue
                if isinstance(cell, pd.Timestamp):
                    rendered = cell.strftime("%Y-%m-%d")
                else:
                    rendered = str(cell)
                parts.append(f"{col}={rendered}")
            if parts:
                lines.append(" | ".join(parts))
    return lines[:max_items]


def _extract_confidence_label(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> str:
    if isinstance(value, CarbonResult):
        if value.result_state in {CARBON_STATE_NOT_COMPUTABLE, CARBON_STATE_UNSUPPORTED}:
            return "low / unavailable (deterministic carbon computation unavailable for this scope)"
        if value.result_state == CARBON_STATE_RETRIEVAL_ONLY:
            return "retrieval-only (supporting traffic evidence found, not numeric carbon source-of-truth)"
        if value.result_state == CARBON_STATE_FORECAST_ONLY:
            return "unavailable (carbon forecast requested but deterministic carbon forecast is not configured)"
        return f"{value.confidence_label} ({value.confidence_reason})"
    if value.status != "ok":
        return "low (insufficient matched data/evidence)"
    for note in value.caveats:
        if note.lower().startswith("confidence:"):
            return note.split(":", 1)[1].strip()
    if isinstance(value, ForecastResult):
        return "medium (forecast based on available historical patterns)"
    return "medium (deterministic aggregation over filtered rows)"


def _build_method_steps(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
    steps: List[str] = []
    if isinstance(value, CarbonResult):
        steps.append(f"Result state: {value.result_state}.")
        if value.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
            steps.append("Deterministic carbon computation: unavailable for this scope.")
            if value.result_state == CARBON_STATE_RETRIEVAL_ONLY:
                steps.append("Retrieved evidence is traffic-only context and not numeric carbon source-of-truth.")
            elif value.result_state == CARBON_STATE_FORECAST_ONLY:
                steps.append("Forecast-only carbon query detected; no deterministic carbon forecast model configured.")
            for note in value.coverage_notes[:4]:
                steps.append(note)
            return steps
        steps.append("Applied deterministic AIS + port-call mode segmentation (transit/manoeuvring/berth/anchorage).")
        steps.append(f"Boundary: {value.boundary}; Pollutants: {', '.join(value.pollutants)}.")
        steps.append(f"Source label: {value.source_label}.")
        steps.append(f"Confidence: {value.confidence_label} ({value.confidence_reason})")
        if value.params_version:
            steps.append(f"Factor registry version: {value.params_version}.")
        for note in value.coverage_notes[:6]:
            steps.append(note)
        return steps

    if isinstance(value, ForecastResult):
        steps.append("Applied active filters (port/date/vessel-type) to congestion history.")
        for note in value.coverage_notes:
            if (
                note.startswith("Coverage window:")
                or note.startswith("Rows used:")
                or note.startswith("Target date:")
                or note.startswith("Forecast target weekday:")
                or note.startswith("Analog dates used:")
                or note.startswith("Analog values used:")
                or note.startswith("Meaning:")
                or note.startswith("Method:")
            ):
                steps.append(note)
        steps.append("Computed point estimate and uncertainty interval for the requested target.")
    else:
        steps.append("Applied active filters to KPI tables.")
        for note in value.coverage_notes:
            if note.startswith("Coverage window:") or note.startswith("Rows used:") or note.startswith("Data sources used:"):
                steps.append(note)
        if value.table is not None and not value.table.empty:
            steps.append(f"Aggregated filtered rows into {len(value.table):,} output row(s).")
        else:
            steps.append("Computed deterministic metric directly from filtered subset.")

    for assumption in [c for c in value.caveats if not c.lower().startswith("confidence:")][:2]:
        steps.append(f"Assumption: {assumption}")

    out: List[str] = []
    for step in steps:
        if step and step not in out:
            out.append(step)
    return out


def _build_port_actions(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
    actions: List[str] = []
    if isinstance(value, CarbonResult):
        if value.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
            return [
                "Improve carbon data coverage for the selected scope before interpreting emissions numerically.",
                "Add validated vessel fuel/engine/activity factors for periods with missing deterministic carbon rows.",
                "Use retrieved traffic evidence as context only until deterministic carbon inventory is available.",
            ]
        metric = value.uncertainty_interval.get("CO2e") or value.uncertainty_interval.get("CO2") or {}
        point = float(metric.get("point", 0.0))
        upper = float(metric.get("upper", point))
        if point >= 50:
            actions.append("Prioritize shore-power and berth energy optimization on the highest-emitting call windows.")
            actions.append("Coordinate speed and arrival windows with pilots to reduce manoeuvring fuel burn.")
        elif point >= 15:
            actions.append("Apply targeted slow-steaming and auxiliary-load management for vessels in this corridor.")
            actions.append("Flag high-intensity days for emissions-aware berth allocation.")
        else:
            actions.append("Keep baseline operating plan; monitor for drift against this emissions baseline.")
        if upper > point * 1.4:
            actions.append("Uncertainty is wide; refresh with updated AIS coverage before operational decisions.")
        actions.append("Use evidence IDs in technical audit mode to validate factor/fallback assumptions.")
        return actions

    if isinstance(value, ForecastResult) and value.forecast is not None and not value.forecast.empty:
        pred = float(value.forecast["predicted"].mean())
        upper = float(value.forecast["upper"].mean()) if "upper" in value.forecast.columns else pred
        lower = float(value.forecast["lower"].mean()) if "lower" in value.forecast.columns else pred
        spread = max(0.0, upper - pred)
        confidence = _extract_confidence_label(value).lower()
        if pred >= 1.8:
            actions.append("Activate high-traffic playbook: reserve extra berth windows and pre-book pilot/tug shifts.")
            actions.append("Advance-notify terminal and gate teams to smooth truck and yard peaks.")
        elif pred >= 1.3:
            actions.append("Pre-allocate buffer berth slots and increase watchstanding in VTS for the target window.")
            actions.append("Coordinate with agents to stagger ETAs for vessels with flexible arrival windows.")
        else:
            actions.append("Run normal berth plan but keep one fallback slot for late-arrival clustering.")
        actions.append(f"Use predicted range {lower:.2f}-{upper:.2f} to set staffing floors/ceilings instead of a single-point plan.")
        if spread >= 0.6:
            actions.append("Maintain operational contingency: uncertainty is wide, so add tug/pilot standby margin.")
        if "low" in confidence:
            actions.append("Refresh this forecast 24-48 hours before execution because confidence is currently low.")
        return actions

    answer_text = value.answer.lower()
    if "jump" in answer_text or "anomaly" in answer_text:
        actions.append("Open AIS integrity checks for listed MMSI and validate with external tracking feeds.")
        actions.append("Flag suspicious tracks for VTS review before acting on route deviations.")
    else:
        actions.append("Use the daily/weekly pattern in the chart to plan shift staffing and pilot windows.")
        actions.append("Re-run this query with tighter vessel-type filters for targeted operational planning.")
    return actions


def _serialize_chart(chart: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
    if chart is None or chart.empty:
        return None
    df = chart.reset_index()
    records = []
    for row in df.to_dict(orient="records"):
        item: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                item[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                item[key] = value
        records.append(item)
    return records


def _pick_chart(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> Optional[pd.DataFrame]:
    if isinstance(value, AnalyticsResult):
        return value.chart
    if isinstance(value, CarbonResult):
        return value.chart
    if value.forecast is not None and not value.forecast.empty:
        return value.forecast
    if value.history is not None and not value.history.empty:
        return value.history
    return None


def _handle_ask_question_api(
    question: str,
    intent_result: IntentResult,
    kpi: KPIQueryEngine,
    forecaster: ForecastEngine,
    carbon: CarbonQueryEngine,
    retriever: Optional[RAGRetriever],
    retriever_reason: str,
    top_k_evidence: int,
    user_filters: Dict[str, Any],
    events_path: Optional[Path],
) -> tuple[Union[AnalyticsResult, ForecastResult, CarbonResult], EvidenceBundle]:
    entities = intent_result.entities
    q_lower = question.lower()

    port = _pick_filter(user_filters.get("port"), entities.get("port"))
    start = _pick_filter(user_filters.get("date_from"), entities.get("date_from"))
    end = _pick_filter(user_filters.get("date_to"), entities.get("date_to"))
    vessel_type = _pick_filter(user_filters.get("vessel_type"), entities.get("vessel_type"))
    dow = entities.get("dow")
    target_date = entities.get("target_date")
    window = entities.get("window")
    metric = entities.get("metric", "arrivals_vessels")
    aggregation = entities.get("aggregation")
    mmsi = entities.get("mmsi")
    horizon_weeks = int(entities.get("horizon_weeks") or 4)
    ports: List[str] = [str(p).strip() for p in entities.get("ports") or [] if str(p).strip()]
    if port and port not in ports:
        ports.insert(0, port)

    if intent_result.intent == "G":
        return KPIQueryEngine.unsupported(
            "This question needs terminal operations data (berth/crane/TEU/gate), which is not in PRJ912/PRJ896."
        ), EvidenceBundle(lines=[], rows=[], trace={})

    if intent_result.intent == "H":
        result = carbon.from_question_entities(question=question, entities=entities, user_filters=user_filters)
        evidence = _retrieve_evidence_api(
            retriever,
            retriever_reason,
            question,
            entities,
            user_filters,
            top_k_evidence,
            True,
        )
        if isinstance(result, CarbonResult):
            if result.result_state == CARBON_STATE_NOT_COMPUTABLE and evidence.rows:
                result.result_state = CARBON_STATE_RETRIEVAL_ONLY
                result.source_label = "Retrieved supporting traffic evidence only (no deterministic carbon computation)"
                result.confidence_label = "low"
                result.confidence_reason = (
                    "Retrieval-only evidence is available; deterministic carbon inventory is not computable for this scope."
                )
                result.coverage_notes.append(
                    "Traffic evidence was retrieved, but numeric carbon emissions could not be computed reliably."
                )
                result.diagnostics = dict(result.diagnostics or {})
                result.diagnostics["result_state"] = CARBON_STATE_RETRIEVAL_ONLY
                result.diagnostics["sanity_status"] = result.diagnostics.get("sanity_status", "warning")
            elif result.status == "ok" and evidence.rows and result.result_state in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                result.source_label = "Hybrid (computed + retrieved supporting evidence)"
        return result, evidence

    if intent_result.intent == "A":
        if aggregation == "peak_day":
            result = kpi.get_peak_arrival_day(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
        elif mmsi and any(token in q_lower for token in ("how long", "dwell", "in port", "port stay", "stayed")):
            result = kpi.get_mmsi_port_stays(mmsi=str(mmsi), start=start, end=end, port=port)
        elif "top" in q_lower and "port" in q_lower:
            result = kpi.top_ports_by_arrivals(start=start, end=end, vessel_type=vessel_type, dow=dow)
        elif "dwell" in q_lower:
            result = kpi.get_avg_dwell_time(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow)
        elif "congestion" in q_lower:
            result = kpi.get_congestion(port=port, start=start, end=end, dow=dow, window=window)
        else:
            result = kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window)
        evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, True)
        return result, evidence

    if intent_result.intent == "B":
        if aggregation == "peak_day":
            result = kpi.get_peak_arrival_day(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
        elif entities.get("dow") and entities.get("dow_compare"):
            result = kpi.compare_weekdays(
                port=port,
                start=start,
                end=end,
                day_a=entities["dow"],
                day_b=entities["dow_compare"],
                vessel_type=vessel_type,
            )
        elif "hour" in q_lower:
            result = kpi.get_busiest_hour(port=port, start=start, end=end, vessel_type=vessel_type)
        else:
            result = kpi.get_busiest_dow(port=port, start=start, end=end, vessel_type=vessel_type)
        evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, True)
        return result, evidence

    if intent_result.intent == "C":
        if target_date:
            result = forecaster.forecast_congestion_for_date(port=port or "", target_date=target_date, horizon_weeks=horizon_weeks)
        else:
            result = forecaster.forecast_congestion(port=port or "", target_dow=dow or "Friday", horizon_weeks=horizon_weeks)
        evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, False)
        return result, evidence

    if intent_result.intent == "D":
        result = kpi.compare_ports(ports=ports, metric=metric, start=start, end=end, vessel_type=vessel_type, dow=dow)
        evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, True)
        return result, evidence

    if intent_result.intent == "E":
        target = start or end
        result = kpi.diagnose_congestion(port=port, target_date=target)
        evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, True)
        return result, evidence

    if intent_result.intent == "F":
        if any(token in q_lower for token in ("jump", "spoof", "teleport", "impossible")):
            filters = _make_rag_filters(entities=entities, overrides=user_filters, include_dates=True)
            jump_result: Dict[str, Any]
            if retriever is not None:
                jump_result = retriever.detect_sudden_jumps(filters=filters)
            elif events_path and events_path.exists():
                jump_result = detect_sudden_jump_events_from_parquet(
                    events_path=events_path,
                    mmsi=filters.mmsi,
                    date_from=filters.date_from,
                    date_to=filters.date_to,
                )
            else:
                return AnalyticsResult(
                    status="no_data",
                    answer="I don't have row-level AIS evidence in the current runtime to verify jump anomalies.",
                    table=None,
                    chart=None,
                    coverage_notes=[],
                    caveats=[
                        "This query needs either a working vector retriever or events.parquet in the runtime.",
                        "Configure remote Chroma or APP_EVENTS_BUNDLE_URL for event-level anomaly detection.",
                    ],
                ), EvidenceBundle(lines=[], rows=[], trace={})

            count = int(jump_result.get("count", 0))
            events = pd.DataFrame(jump_result.get("events") or [])
            chart = None
            table = None
            if not events.empty:
                table_cols = [
                    c for c in [
                        "mmsi", "timestamp_full", "distance_km", "dt_minutes",
                        "latitude", "longitude", "prev_latitude", "prev_longitude", "port", "stable_id"
                    ] if c in events.columns
                ]
                table = events[table_cols].copy()
                if {"timestamp_full", "distance_km"}.issubset(events.columns):
                    chart = (
                        events.assign(timestamp_dt=pd.to_datetime(events["timestamp_full"], errors="coerce", utc=True))
                        .dropna(subset=["timestamp_dt"])
                        .sort_values("timestamp_dt")
                        .set_index("timestamp_dt")[["distance_km"]]
                    )
            result = AnalyticsResult(
                status="ok",
                answer=f"Detected {count} potential sudden AIS coordinate jumps in the filtered range.",
                table=table,
                chart=chart,
                coverage_notes=[
                    f"Rows used: {count}",
                    "Data sources used: AIS metadata index" if retriever is not None else "Data sources used: row-level AIS events parquet",
                ],
                caveats=[
                    "Jump rule: coordinate displacement above threshold within 30 minutes.",
                    "This is a heuristic anomaly indicator, not proof of spoofing.",
                ],
            )
            evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, True)
            return result, evidence

        result = kpi.detect_arrival_spikes(port=port, start=start, end=end)
        evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, True)
        return result, evidence

    result = kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window)
    evidence = _retrieve_evidence_api(retriever, retriever_reason, question, entities, user_filters, top_k_evidence, True)
    return result, evidence


def _serialize_result(
    result: Union[AnalyticsResult, ForecastResult, CarbonResult],
    evidence: EvidenceBundle,
    threshold_percentiles: Tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": result.status,
        "answer": result.answer,
        "confidence": _extract_confidence_label(result),
        "coverage_notes": result.coverage_notes,
        "caveats": result.caveats,
        "method_steps": _build_method_steps(result),
        "recommendations": _build_port_actions(result),
        "evidence": {
            "computed": _fallback_evidence_from_result(result),
            "retrieved_lines": evidence.lines,
            "retrieved_rows": evidence.rows,
        },
        "chart": _serialize_chart(_pick_chart(result)),
        "retrieval_provenance": evidence.trace,
    }
    if isinstance(result, CarbonResult):
        computed_states = {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}
        is_computed = result.result_state in computed_states
        diagnostics = dict(result.diagnostics or {})
        min_baseline_denominator = 1.0
        try:
            min_baseline_denominator = float(diagnostics.get("min_baseline_denominator_tco2e", 1.0))
        except Exception:
            min_baseline_denominator = 1.0
        payload["carbon"] = {
            "result_state": result.result_state,
            "boundary": result.boundary,
            "pollutants": result.pollutants,
            "source_label": result.source_label,
            "confidence_label": result.confidence_label,
            "confidence_reason": result.confidence_reason,
            "uncertainty_interval": result.uncertainty_interval,
            "params_version": result.params_version,
            "evidence_ids": result.evidence_ids,
            "segment_ids": result.segment_ids,
            "diagnostics": diagnostics,
            "export_csv_path": result.export_csv_path,
            "export_json_path": result.export_json_path,
            "units": {
                "absolute_emissions": "tCO2e (auto-scales to ktCO2e/MtCO2e in UI)",
                "intensity_per_call": "kgCO2e/vessel-call",
                "per_day": "tCO2e/day",
                "per_hour": "kgCO2e/hour",
                "threshold_basis": "relative to this dataset",
            },
            "deterministic_carbon_evidence": _fallback_evidence_from_result(result),
            "retrieved_supporting_traffic_evidence": evidence.lines,
        }
        if not is_computed:
            payload["chart"] = None
            state_reason = {
                CARBON_STATE_NOT_COMPUTABLE: "No deterministic carbon inventory matched the requested scope.",
                CARBON_STATE_RETRIEVAL_ONLY: "Traffic evidence was retrieved, but numeric carbon emissions could not be computed reliably.",
                CARBON_STATE_FORECAST_ONLY: "Carbon forecast was requested, but deterministic carbon forecast outputs are unavailable.",
                CARBON_STATE_UNSUPPORTED: "Carbon query is outside the supported deterministic scope.",
            }.get(result.result_state, "No deterministic carbon output is available for this scope.")
            suggestions = [
                "Improve carbon data coverage for this scope before interpreting emissions numerically.",
                "Add validated fuel/engine/activity factors before interpreting carbon totals.",
                "Use retrieved traffic evidence as context only, not as numeric carbon truth.",
            ]
            payload["carbon"]["availability"] = {
                "computable": False,
                "message": state_reason,
            }
            payload["carbon"]["relative_scale"] = None
            payload["carbon"]["metrics"] = {
                "total_emissions": None,
                "intensity_kgco2e_per_vessel_call": None,
                "tco2e_per_day": None,
                "kgco2e_per_hour": None,
                "relative_level": None,
                "change_vs_baseline": None,
                "change_vs_historical_median": None,
            }
            payload["carbon"]["findings"] = [
                {"type": "status", "text": state_reason},
                {
                    "type": "status",
                    "text": "Retrieved traffic evidence is contextual and does not provide deterministic numeric carbon accounting.",
                }
                if result.result_state == CARBON_STATE_RETRIEVAL_ONLY
                else {"type": "status", "text": "Interpret this response as unavailable rather than low-emission."},
            ]
            payload["carbon"]["emissions_reduction_suggestions"] = suggestions
            payload["recommendations"] = suggestions
            return payload

        metrics = compute_emissions_metrics(result.table, result.boundary)
        total = float(metrics.get("total_tco2e") or 0.0)
        table_metric_col = "wtw_co2e_t" if result.boundary == "WTW" else "ttw_co2e_t"
        if result.table is not None and table_metric_col not in result.table.columns:
            table_metric_col = "co2_t" if "co2_t" in result.table.columns else table_metric_col
        hist_values = (
            pd.to_numeric(result.table.get(table_metric_col), errors="coerce").dropna().tolist()
            if result.table is not None and table_metric_col in result.table.columns
            else []
        )
        bands = derive_threshold_bands(hist_values, percentiles=threshold_percentiles)
        level = classify_level(total, bands)
        hist_median = float(np.median(hist_values)) if hist_values else None
        hist_mean = float(np.mean(hist_values)) if hist_values else None
        change_vs_median_pct = safe_percent_delta(
            current_value=total,
            baseline_value=hist_median,
            min_denominator=min_baseline_denominator,
        )
        change_vs_baseline_pct = safe_percent_delta(
            current_value=total,
            baseline_value=hist_mean,
            min_denominator=min_baseline_denominator,
        )
        ci_item = result.uncertainty_interval.get("CO2e") or result.uncertainty_interval.get("CO2") or {}
        point = float(ci_item.get("point", 0.0))
        lower = float(ci_item.get("lower", 0.0))
        upper = float(ci_item.get("upper", 0.0))
        ci_width_rel = ((upper - lower) / point) if point > 0 else None
        chart_df = _pick_chart(result)
        chart_findings = extract_chart_findings(chart_df if chart_df is not None else pd.DataFrame(), target_ts=None, max_findings=5)
        findings = build_emissions_findings(
            current_tco2e=total,
            level=level,
            change_vs_median_pct=change_vs_median_pct,
            source_label=result.source_label,
            ci_width_rel=ci_width_rel,
            chart_findings=chart_findings,
        )
        if change_vs_baseline_pct is None or change_vs_median_pct is None:
            findings.append(
                {
                    "type": "inferred",
                    "text": "Baseline denominator is too small for meaningful percentage comparison in this scope.",
                }
            )
        suggestions = build_reduction_suggestions(
            level=level,
            change_vs_median_pct=change_vs_median_pct,
            ci_width_rel=ci_width_rel,
            source_label=result.source_label,
        )
        payload["carbon"]["availability"] = {"computable": True, "message": "Deterministic carbon inventory computed."}
        payload["carbon"]["relative_scale"] = {
            "classification": level,
            "basis": bands.source_label,
            "thresholds_tco2e": {"p25": bands.p25, "p50": bands.p50, "p75": bands.p75},
            "current_tco2e": total,
            "current_display": format_tco2e(total),
        }
        payload["carbon"]["metrics"] = {
            "total_emissions": format_tco2e(total),
            "intensity_kgco2e_per_vessel_call": (
                f"{float(metrics['intensity_kg_per_call']):.1f} kgCO2e/vessel-call"
                if metrics.get("intensity_kg_per_call") is not None
                else None
            ),
            "tco2e_per_day": (
                f"{float(metrics['tco2e_per_day']):.2f} tCO2e/day"
                if metrics.get("tco2e_per_day") is not None
                else None
            ),
            "kgco2e_per_hour": (
                f"{float(metrics['kgco2e_per_hour']):.2f} kgCO2e/hour"
                if metrics.get("kgco2e_per_hour") is not None
                else None
            ),
            "relative_level": level,
            "change_vs_baseline": format_percent(change_vs_baseline_pct) if change_vs_baseline_pct is not None else None,
            "change_vs_historical_median": (
                format_percent(change_vs_median_pct) if change_vs_median_pct is not None else None
            ),
        }
        payload["carbon"]["findings"] = findings
        payload["carbon"]["emissions_reduction_suggestions"] = suggestions
        payload["recommendations"] = suggestions
    return payload


def _build_state() -> Dict[str, Any]:
    config_path = "config/config.yaml"
    config = load_config(config_path)
    configured_processed_dir = Path(config.get("predict", {}).get("processed_dir", "data/processed"))
    carbon_cfg = config.get("carbon", {})
    threshold_percentiles = sanitize_threshold_percentiles(
        carbon_cfg.get("relative_level_percentiles", [0.25, 0.50, 0.75])
    )
    _maybe_bootstrap_bundle(
        "APP_PROCESSED_BUNDLE_URL",
        configured_processed_dir,
        [
            "arrivals_daily.parquet",
            "arrivals_hourly.parquet",
            "congestion_daily.parquet",
            "dwell_time.parquet",
            "occupancy_hourly.parquet",
            "port_catalog.parquet",
            "kpi_capabilities.json",
        ],
    )
    _maybe_bootstrap_bundle(
        "APP_EVENTS_BUNDLE_URL",
        configured_processed_dir,
        ["events.parquet"],
    )

    requested_vector_mode = str(os.getenv("VECTOR_DB_MODE", config.get("vector_db", {}).get("mode", "local"))).strip().lower()
    try:
        using_remote_vector = chroma_remote_settings(config=config) is not None
    except Exception:
        using_remote_vector = False
    configured_persist_dir = Path(config["paths"].get("persist_dir", "data/chroma"))
    chroma_bootstrap_changed = False
    chroma_bootstrap_message = ""
    if not using_remote_vector:
        chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_runtime(
            configured_persist_dir
        )

    processed_dir, using_demo_processed = _resolve_processed_dir(configured_processed_dir)
    if using_remote_vector:
        persist_dir = configured_persist_dir
        using_demo_chroma = False
    else:
        persist_dir, using_demo_chroma = _resolve_persist_dir(configured_persist_dir)

    kpi_engine = KPIQueryEngine(processed_dir=processed_dir)
    forecast_engine = ForecastEngine(processed_dir=processed_dir)
    carbon_engine = CarbonQueryEngine(
        processed_dir=processed_dir,
        factor_registry_path=carbon_cfg.get("factor_registry_path", "config/carbon_factors.v1.json"),
        monte_carlo_draws=int(carbon_cfg.get("monte_carlo_draws", 500)),
        auto_build=True,
    )

    retriever = None
    retriever_reason = ""
    api_key = _load_runtime_setting("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            retriever = _init_retriever(persist_dir=persist_dir, config_path=config_path)
            retriever_reason = f"Retriever active (backend: {retriever.vector_backend})."
        except Exception as exc:
            retriever_reason = f"Retriever init failed: {exc}"
            if using_remote_vector:
                chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_runtime(
                    configured_persist_dir
                )
                fallback_persist_dir, fallback_using_demo_chroma = _resolve_persist_dir(configured_persist_dir)
                if (fallback_persist_dir / "chroma.sqlite3").exists():
                    try:
                        retriever = _init_retriever(
                            persist_dir=fallback_persist_dir,
                            config_path=config_path,
                            force_local_vector=True,
                        )
                        persist_dir = fallback_persist_dir
                        using_demo_chroma = fallback_using_demo_chroma
                        retriever_reason = (
                            f"Remote retriever failed ({exc}). "
                            f"Fell back to local vector store at {fallback_persist_dir} "
                            f"(backend: {retriever.vector_backend})."
                        )
                    except Exception as local_exc:
                        retriever_reason = (
                            f"Remote retriever failed ({exc}); local fallback failed ({local_exc})."
                        )
    else:
        retriever_reason = "Retriever unavailable: OPENAI_API_KEY not set."

    events_path = configured_processed_dir / "events.parquet"
    if not events_path.exists():
        events_path = processed_dir / "events.parquet"

    return {
        "config_path": config_path,
        "threshold_percentiles": threshold_percentiles,
        "processed_dir": str(processed_dir),
        "persist_dir": str(persist_dir),
        "using_demo_processed": using_demo_processed,
        "using_demo_chroma": using_demo_chroma,
        "using_remote_vector": using_remote_vector,
        "requested_vector_mode": requested_vector_mode,
        "chroma_bootstrap_changed": chroma_bootstrap_changed,
        "chroma_bootstrap_message": chroma_bootstrap_message,
        "kpi": kpi_engine,
        "forecast": forecast_engine,
        "carbon": carbon_engine,
        "retriever": retriever,
        "retriever_reason": retriever_reason,
        "events_path": str(events_path),
    }


app = FastAPI(title="Eagle Eye API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    app.state.runtime = _build_state()


def _runtime_state() -> Dict[str, Any]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        runtime = _build_state()
        app.state.runtime = runtime
    return runtime


@app.get("/health")
def health() -> Dict[str, Any]:
    state = _runtime_state()
    return {
        "status": "ok",
        "processed_dir": state["processed_dir"],
        "persist_dir": state["persist_dir"],
        "using_demo_processed": state["using_demo_processed"],
        "using_demo_chroma": state["using_demo_chroma"],
        "using_remote_vector": state["using_remote_vector"],
        "requested_vector_mode": state["requested_vector_mode"],
        "chroma_bootstrap_changed": state["chroma_bootstrap_changed"],
        "chroma_bootstrap_message": state["chroma_bootstrap_message"],
        "retriever_reason": state["retriever_reason"],
        "events_available": bool(Path(state["events_path"]).exists()),
        "carbon_available": bool(state["carbon"].available),
        "carbon_params_version": state["carbon"].params_version.get("version"),
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Eagle Eye API",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
        "carbon_ports": "/api/v1/carbon/ports/{port_id}/emissions",
        "carbon_call": "/api/v1/carbon/vessels/{mmsi}/calls/{call_id}",
        "carbon_estimate": "/api/v1/carbon/estimate",
        "carbon_evidence": "/api/v1/carbon/evidence/{evidence_id}",
    }


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    state = _runtime_state()
    question = req.question.strip()
    intent_result = classify_question(question)
    user_filters = req.filters.model_dump()
    result, evidence = _handle_ask_question_api(
        question=question,
        intent_result=intent_result,
        kpi=state["kpi"],
        forecaster=state["forecast"],
        carbon=state["carbon"],
        retriever=state["retriever"],
        retriever_reason=state["retriever_reason"],
        top_k_evidence=req.top_k_evidence,
        user_filters=user_filters,
        events_path=Path(state["events_path"]) if Path(state["events_path"]).exists() else None,
    )
    return {
        "question": question,
        "intent": asdict(intent_result),
        "result": _serialize_result(result, evidence, threshold_percentiles=state["threshold_percentiles"]),
    }


def _parse_pollutants_query(value: Optional[str]) -> List[str]:
    if not value:
        return ["CO2e", "NOx", "SOx", "PM"]
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    return items or ["CO2e", "NOx", "SOx", "PM"]


@app.get("/api/v1/carbon/ports/{port_id}/emissions")
def carbon_port_emissions(
    port_id: str,
    from_date: Optional[str] = Query(default=None, alias="from"),
    to_date: Optional[str] = Query(default=None, alias="to"),
    group_by: str = Query(default="day"),
    boundary: str = Query(default="TTW"),
    pollutants: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    result = engine.query_port_emissions(
        port_id=port_id,
        date_from=from_date,
        date_to=to_date,
        group_by=group_by,
        boundary=boundary,
        pollutants=_parse_pollutants_query(pollutants),
        include_uncertainty=True,
        include_evidence=True,
    )
    return {
        "port_id": port_id,
        "result": _serialize_result(
            result,
            EvidenceBundle(lines=[], rows=[], trace={}),
            threshold_percentiles=state["threshold_percentiles"],
        ),
    }


@app.get("/api/v1/carbon/vessels/{mmsi}/calls/{call_id}")
def carbon_vessel_call(
    mmsi: str,
    call_id: str,
    boundary: str = Query(default="TTW"),
    pollutants: Optional[str] = Query(default=None),
    include_uncertainty: bool = Query(default=True),
    include_evidence: bool = Query(default=True),
) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    result = engine.query_vessel_call(
        mmsi=mmsi,
        call_id=call_id,
        boundary=boundary,
        pollutants=_parse_pollutants_query(pollutants),
        include_uncertainty=include_uncertainty,
        include_evidence=include_evidence,
    )
    return {
        "mmsi": mmsi,
        "call_id": call_id,
        "result": _serialize_result(
            result,
            EvidenceBundle(lines=[], rows=[], trace={}),
            threshold_percentiles=state["threshold_percentiles"],
        ),
    }


@app.post("/api/v1/carbon/estimate")
def carbon_estimate(req: CarbonEstimateRequest) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    result = engine.estimate_with_assumptions(req.model_dump())
    return {
        "result": _serialize_result(
            result,
            EvidenceBundle(lines=[], rows=[], trace={}),
            threshold_percentiles=state["threshold_percentiles"],
        )
    }


@app.get("/api/v1/carbon/evidence/{evidence_id}")
def carbon_evidence(evidence_id: str) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    payload = engine.get_evidence(evidence_id)
    if payload.get("status") != "ok":
        raise HTTPException(status_code=404, detail=payload.get("reason", "Evidence not found"))
    return payload
