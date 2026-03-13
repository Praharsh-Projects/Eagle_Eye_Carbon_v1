"""FastAPI deployment path for Eagle Eye."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.qa.intent import IntentResult, classify_question
from src.rag.retriever import QueryFilters, RAGRetriever
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet
from src.utils.cloud_bootstrap import ensure_bundle
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


def _fallback_evidence_from_result(value: Union[AnalyticsResult, ForecastResult], max_items: int = 5) -> List[str]:
    lines: List[str] = []
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


def _extract_confidence_label(value: Union[AnalyticsResult, ForecastResult]) -> str:
    if value.status != "ok":
        return "low (insufficient matched data/evidence)"
    for note in value.caveats:
        if note.lower().startswith("confidence:"):
            return note.split(":", 1)[1].strip()
    if isinstance(value, ForecastResult):
        return "medium (forecast based on available historical patterns)"
    return "medium (deterministic aggregation over filtered rows)"


def _build_method_steps(value: Union[AnalyticsResult, ForecastResult]) -> List[str]:
    steps: List[str] = []
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


def _build_port_actions(value: Union[AnalyticsResult, ForecastResult]) -> List[str]:
    actions: List[str] = []
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


def _pick_chart(value: Union[AnalyticsResult, ForecastResult]) -> Optional[pd.DataFrame]:
    if isinstance(value, AnalyticsResult):
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
    retriever: Optional[RAGRetriever],
    retriever_reason: str,
    top_k_evidence: int,
    user_filters: Dict[str, Any],
    events_path: Optional[Path],
) -> tuple[Union[AnalyticsResult, ForecastResult], EvidenceBundle]:
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
    horizon_weeks = int(entities.get("horizon_weeks") or 4)
    ports: List[str] = [str(p).strip() for p in entities.get("ports") or [] if str(p).strip()]
    if port and port not in ports:
        ports.insert(0, port)

    if intent_result.intent == "G":
        return KPIQueryEngine.unsupported(
            "This question needs terminal operations data (berth/crane/TEU/gate), which is not in PRJ912/PRJ896."
        ), EvidenceBundle(lines=[], rows=[], trace={})

    if intent_result.intent == "A":
        if "top" in q_lower and "port" in q_lower:
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
        if entities.get("dow") and entities.get("dow_compare"):
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
    result: Union[AnalyticsResult, ForecastResult],
    evidence: EvidenceBundle,
) -> Dict[str, Any]:
    return {
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


def _build_state() -> Dict[str, Any]:
    config_path = "config/config.yaml"
    config = load_config(config_path)
    configured_processed_dir = Path(config.get("predict", {}).get("processed_dir", "data/processed"))
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
        chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_bundle(
            "APP_CHROMA_BUNDLE_URL",
            configured_persist_dir,
            ["chroma.sqlite3", "traffic_metadata_index.csv"],
        )

    processed_dir, using_demo_processed = _resolve_processed_dir(configured_processed_dir)
    if using_remote_vector:
        persist_dir = configured_persist_dir
        using_demo_chroma = False
    else:
        persist_dir, using_demo_chroma = _resolve_persist_dir(configured_persist_dir)

    kpi_engine = KPIQueryEngine(processed_dir=processed_dir)
    forecast_engine = ForecastEngine(processed_dir=processed_dir)

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
                chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_bundle(
                    "APP_CHROMA_BUNDLE_URL",
                    configured_persist_dir,
                    ["chroma.sqlite3", "traffic_metadata_index.csv"],
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
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Eagle Eye API",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
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
        retriever=state["retriever"],
        retriever_reason=state["retriever_reason"],
        top_k_evidence=req.top_k_evidence,
        user_filters=user_filters,
        events_path=Path(state["events_path"]) if Path(state["events_path"]).exists() else None,
    )
    return {
        "question": question,
        "intent": asdict(intent_result),
        "result": _serialize_result(result, evidence),
    }
