"""Simplified Ask-only Streamlit app with integrated future prediction."""

from __future__ import annotations

import time
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
try:
    import altair as alt
except Exception:  # pragma: no cover - optional visualization dependency
    alt = None

from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.qa.intent import IntentResult, classify_question
from src.rag.retriever import QueryFilters, RAGRetriever
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet
from src.utils.cloud_bootstrap import ensure_bundle
from src.utils.config import load_config
from src.utils.runtime import chroma_remote_settings, force_local_vector_env
from src.utils.serialization import compact_traffic_evidence


SAMPLE_QUERIES = [
    "How many vessel arrivals were recorded at SEGOT in March 2022?",
    "Which weekday is usually busiest at LVVNT?",
    "Compare Friday and Monday arrivals at GDANSK in March 2022.",
    "Show suspicious AIS jumps for MMSI 212575000 on 2021-01-01.",
    "For MMSI 266232000, summarize movement and destination changes on 2021-01-01.",
    "What will congestion be at LVVNT on Friday, February 20, 2026?",
    "Predict congestion for SEGOT next Friday based on historical patterns.",
    "Expected congestion at GDANSK on 2026-03-06?",
    "Compare expected congestion next Friday between LVVNT and SEGOT.",
    "Predict whether LUBECK is likely high congestion on 2026-02-20.",
]


@dataclass
class EvidenceBundle:
    lines: List[str]
    rows: List[Dict[str, Any]]
    trace: Dict[str, Any]


@st.cache_resource
def _init_kpi_engine(processed_dir: str) -> KPIQueryEngine:
    return KPIQueryEngine(processed_dir=processed_dir)


@st.cache_resource
def _init_forecast_engine(processed_dir: str) -> ForecastEngine:
    return ForecastEngine(processed_dir=processed_dir)


@st.cache_resource
def _init_retriever(
    persist_dir: str,
    config_path: str,
    force_local_vector: bool = False,
) -> RAGRetriever:
    if force_local_vector:
        with force_local_vector_env():
            return RAGRetriever(persist_dir=persist_dir, config_path=config_path)
    return RAGRetriever(persist_dir=persist_dir, config_path=config_path)


def _resolve_processed_dir(preferred_dir: Path) -> tuple[Path, bool]:
    required = preferred_dir / "arrivals_daily.parquet"
    if required.exists():
        return preferred_dir, False
    fallback = Path("demo_data/processed")
    if (fallback / "arrivals_daily.parquet").exists():
        return fallback, True
    return preferred_dir, False


def _resolve_persist_dir(preferred_dir: Path) -> tuple[Path, bool]:
    required = preferred_dir / "chroma.sqlite3"
    if required.exists():
        return preferred_dir, False
    fallback = Path("demo_data/chroma")
    if (fallback / "chroma.sqlite3").exists():
        return fallback, True
    return preferred_dir, False


def _remote_vector_enabled(config: Dict[str, Any]) -> bool:
    try:
        return chroma_remote_settings(config=config) is not None
    except Exception:
        return False


def _parse_anomaly_filter(value: str) -> Optional[bool]:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _load_openai_api_key_from_runtime() -> tuple[Optional[str], str]:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key, "env"

    try:
        secret_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        secret_key = ""

    if secret_key:
        os.environ["OPENAI_API_KEY"] = secret_key
        return secret_key, "streamlit_secrets"
    return None, "missing"


def _load_runtime_setting(name: str) -> tuple[str, str]:
    value = str(os.getenv(name, "")).strip()
    if value:
        return value, "env"
    try:
        secret_value = str(st.secrets.get(name, "")).strip()
    except Exception:
        secret_value = ""
    if secret_value:
        os.environ[name] = secret_value
        return secret_value, "streamlit_secrets"
    return "", "missing"


def _maybe_bootstrap_processed_bundle(preferred_dir: Path) -> tuple[bool, str]:
    required_files = [
        "arrivals_daily.parquet",
        "arrivals_hourly.parquet",
        "congestion_daily.parquet",
        "dwell_time.parquet",
        "occupancy_hourly.parquet",
        "port_catalog.parquet",
        "kpi_capabilities.json",
    ]
    if all((preferred_dir / name).exists() for name in required_files):
        return False, f"Processed runtime assets already exist in {preferred_dir}."

    bundle_url, source = _load_runtime_setting("APP_PROCESSED_BUNDLE_URL")
    if not bundle_url:
        return False, "No APP_PROCESSED_BUNDLE_URL configured."

    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=preferred_dir,
        required_files=required_files,
    )
    if source != "missing":
        message = f"{message} Source: {source}."
    return changed, message


def _maybe_bootstrap_events_bundle(preferred_dir: Path) -> tuple[bool, str]:
    required_files = ["events.parquet"]
    if all((preferred_dir / name).exists() for name in required_files):
        return False, f"Events runtime asset already exists in {preferred_dir}."

    bundle_url, source = _load_runtime_setting("APP_EVENTS_BUNDLE_URL")
    if not bundle_url:
        return False, "No APP_EVENTS_BUNDLE_URL configured."

    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=preferred_dir,
        required_files=required_files,
    )
    if source != "missing":
        message = f"{message} Source: {source}."
    return changed, message


def _maybe_bootstrap_chroma_bundle(preferred_dir: Path) -> tuple[bool, str]:
    required_files = ["chroma.sqlite3", "traffic_metadata_index.csv"]
    if all((preferred_dir / name).exists() for name in required_files):
        return False, f"Chroma runtime assets already exist in {preferred_dir}."

    bundle_url, source = _load_runtime_setting("APP_CHROMA_BUNDLE_URL")
    if not bundle_url:
        return False, "No APP_CHROMA_BUNDLE_URL configured."

    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=preferred_dir,
        required_files=required_files,
        timeout_seconds=1800,
    )
    if source != "missing":
        message = f"{message} Source: {source}."
    return changed, message


def _pick_filter(override: Optional[str], extracted: Optional[str]) -> Optional[str]:
    if override is not None and override.strip():
        return override.strip()
    if extracted is not None and str(extracted).strip():
        return str(extracted).strip()
    return None


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


def _retrieve_evidence(
    retriever: Optional[RAGRetriever],
    question: str,
    entities: Dict[str, Any],
    overrides: Dict[str, Any],
    top_k: int,
    include_dates: bool,
) -> EvidenceBundle:
    runtime_reason = str(st.session_state.get("retriever_reason", "")).strip()
    empty = EvidenceBundle(
        lines=[],
        rows=[],
        trace={
            "retrieval_status": "disabled",
            "reason": runtime_reason
            or "Retriever unavailable (missing OPENAI_API_KEY or retriever init failure).",
        },
    )
    if retriever is None:
        return empty

    try:
        filters = _make_rag_filters(entities=entities, overrides=overrides, include_dates=include_dates)
        started = time.perf_counter()
        result = retriever.query_traffic(question=question, filters=filters, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        return EvidenceBundle(
            lines=[],
            rows=[],
            trace={
                "retrieval_status": "error",
                "reason": f"Vector retrieval failed: {exc}",
            },
        )

    lines: List[str] = []
    rows: List[Dict[str, Any]] = []
    for item in result.evidence[:top_k]:
        chunk_id = str(
            item.metadata.get("chunk_id")
            or item.metadata.get("stable_id")
            or item.metadata.get("id")
            or ""
        )
        dist_txt = f"{float(item.distance):.4f}" if item.distance is not None else "n/a"
        lines.append(
            f"`vector_id={item.id}` | `chunk_id={chunk_id or 'n/a'}` | `dist={dist_txt}` | "
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
                "vessel_type": item.metadata.get("vessel_type_norm")
                or item.metadata.get("vessel_type"),
                "mmsi": item.metadata.get("mmsi"),
            }
        )

    active_filters = {
        key: value
        for key, value in {
            "mmsi": filters.mmsi,
            "imo": filters.imo,
            "locode": filters.locode,
            "port_name": filters.port_name,
            "destination": filters.destination,
            "vessel_type": filters.vessel_type,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "lat_min": filters.lat_min,
            "lat_max": filters.lat_max,
            "lon_min": filters.lon_min,
            "lon_max": filters.lon_max,
        }.items()
        if value not in (None, "", [])
    }

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
        "active_filters": active_filters,
    }
    return EvidenceBundle(lines=lines, rows=rows, trace=trace)


def _extract_prediction_triplet(
    result: ForecastResult,
    target_date: Optional[str],
    target_dow: Optional[str],
) -> Optional[Tuple[float, float, float]]:
    if result.forecast is None or result.forecast.empty:
        return None

    df = result.forecast.copy()
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
    rows = df.dropna(subset=["date"])

    if target_date:
        target_ts = pd.to_datetime(target_date, errors="coerce", utc=True)
        if pd.notna(target_ts):
            target_ts = pd.Timestamp(target_ts).floor("D")
            picked = rows[rows["date"] == target_ts]
            if picked.empty:
                picked = rows.tail(1)
            return (
                float(picked["predicted"].mean()),
                float(picked["lower"].mean()),
                float(picked["upper"].mean()),
            )

    if target_dow:
        dow_rows = rows[rows["date"].dt.day_name() == target_dow.title()]
        if dow_rows.empty:
            dow_rows = rows.tail(1)
        return (
            float(dow_rows["predicted"].mean()),
            float(dow_rows["lower"].mean()),
            float(dow_rows["upper"].mean()),
        )

    tail = rows.tail(1)
    return (
        float(tail["predicted"].mean()),
        float(tail["lower"].mean()),
        float(tail["upper"].mean()),
    )


def _compare_forecast_ports(
    forecaster: ForecastEngine,
    ports: List[str],
    target_date: Optional[str],
    target_dow: Optional[str],
    horizon_weeks: int,
) -> AnalyticsResult:
    unique_ports = []
    for port in ports:
        token = str(port).strip()
        if token and token not in unique_ports:
            unique_ports.append(token)

    if len(unique_ports) < 2:
        return KPIQueryEngine.no_data("Comparison forecast needs at least two distinct ports in the question.")

    p1, p2 = unique_ports[0], unique_ports[1]

    if target_date:
        r1 = forecaster.forecast_congestion_for_date(port=p1, target_date=target_date, horizon_weeks=horizon_weeks)
        r2 = forecaster.forecast_congestion_for_date(port=p2, target_date=target_date, horizon_weeks=horizon_weeks)
    else:
        r1 = forecaster.forecast_congestion(port=p1, target_dow=target_dow or "Friday", horizon_weeks=horizon_weeks)
        r2 = forecaster.forecast_congestion(port=p2, target_dow=target_dow or "Friday", horizon_weeks=horizon_weeks)

    t1 = _extract_prediction_triplet(r1, target_date=target_date, target_dow=target_dow)
    t2 = _extract_prediction_triplet(r2, target_date=target_date, target_dow=target_dow)
    if t1 is None or t2 is None:
        return KPIQueryEngine.no_data("Could not compute comparison forecast due to missing forecast outputs.")

    p1_pred, p1_low, p1_high = t1
    p2_pred, p2_low, p2_high = t2
    higher = p1 if p1_pred >= p2_pred else p2

    target_label = target_date or (target_dow or "selected period")
    answer = (
        f"Predicted congestion comparison for {target_label}: {p1}={p1_pred:.2f} ({p1_low:.2f}-{p1_high:.2f}) vs "
        f"{p2}={p2_pred:.2f} ({p2_low:.2f}-{p2_high:.2f}). {higher} is likely higher."
    )

    table = pd.DataFrame(
        [
            {"port": p1, "predicted": p1_pred, "lower": p1_low, "upper": p1_high},
            {"port": p2, "predicted": p2_pred, "lower": p2_low, "upper": p2_high},
        ]
    )

    coverage_notes = [f"Target: {target_label}", "Method: per-port congestion forecast followed by direct comparison."]
    for item in (r1.coverage_notes + r2.coverage_notes):
        if item not in coverage_notes:
            coverage_notes.append(item)

    caveats: List[str] = []
    for item in (r1.caveats + r2.caveats):
        if item not in caveats:
            caveats.append(item)

    return AnalyticsResult(
        status="ok",
        answer=answer,
        table=table,
        chart=table.set_index("port")[["predicted"]],
        coverage_notes=coverage_notes,
        caveats=caveats[:6],
    )


def _handle_ask_question(
    question: str,
    intent_result: IntentResult,
    kpi: KPIQueryEngine,
    forecaster: ForecastEngine,
    retriever: Optional[RAGRetriever],
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
        return (
            KPIQueryEngine.unsupported(
                "This question needs terminal operations data (berth/crane/TEU/gate), which is not in PRJ912/PRJ896."
            ),
            EvidenceBundle(lines=[], rows=[], trace={}),
        )

    if intent_result.intent == "A":
        if "top" in q_lower and "port" in q_lower:
            result = kpi.top_ports_by_arrivals(start=start, end=end, vessel_type=vessel_type, dow=dow)
        elif "dwell" in q_lower:
            result = kpi.get_avg_dwell_time(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow)
        elif "congestion" in q_lower:
            result = kpi.get_congestion(port=port, start=start, end=end, dow=dow, window=window)
        else:
            result = kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window)
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=user_filters,
            top_k=top_k_evidence,
            include_dates=True,
        )
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

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=user_filters,
            top_k=top_k_evidence,
            include_dates=True,
        )
        return result, evidence

    if intent_result.intent == "C":
        if len(ports) >= 2 and any(token in q_lower for token in ("compare", "vs", "versus", "more than", "less than")):
            result = _compare_forecast_ports(
                forecaster=forecaster,
                ports=ports,
                target_date=target_date,
                target_dow=dow,
                horizon_weeks=horizon_weeks,
            )
            evidence = _retrieve_evidence(
                retriever=retriever,
                question=question,
                entities=entities,
                overrides=user_filters,
                top_k=top_k_evidence,
                include_dates=False,
            )
            return result, evidence

        if target_date:
            result = forecaster.forecast_congestion_for_date(
                port=port or "",
                target_date=target_date,
                horizon_weeks=horizon_weeks,
            )
        else:
            result = forecaster.forecast_congestion(
                port=port or "",
                target_dow=dow or "Friday",
                horizon_weeks=horizon_weeks,
            )

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=user_filters,
            top_k=top_k_evidence,
            include_dates=False,
        )
        return result, evidence

    if intent_result.intent == "D":
        result = kpi.compare_ports(
            ports=ports,
            metric=metric,
            start=start,
            end=end,
            vessel_type=vessel_type,
            dow=dow,
        )
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=user_filters,
            top_k=top_k_evidence,
            include_dates=True,
        )
        return result, evidence

    if intent_result.intent == "E":
        if not start and not end and not kpi.congestion.empty:
            latest = pd.to_datetime(kpi.congestion["date"], errors="coerce", utc=True).max()
            if pd.notna(latest):
                start = end = latest.strftime("%Y-%m-%d")
        target = start or end
        result = kpi.diagnose_congestion(port=port, target_date=target)
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=user_filters,
            top_k=top_k_evidence,
            include_dates=True,
        )
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
                return (
                    AnalyticsResult(
                        status="no_data",
                        answer="I don't have row-level AIS evidence in the current runtime to verify jump anomalies.",
                        table=None,
                        chart=None,
                        coverage_notes=[],
                        caveats=[
                            "This query needs either a working vector retriever or events.parquet in the cloud runtime.",
                            "Configure remote Chroma or set APP_EVENTS_BUNDLE_URL for event-level anomaly detection.",
                        ],
                    ),
                    EvidenceBundle(lines=[], rows=[], trace={}),
                )

            count = int(jump_result.get("count", 0))
            events = pd.DataFrame(jump_result.get("events") or [])
            chart = None
            table = None
            if not events.empty:
                table_cols = [
                    c
                    for c in [
                        "mmsi",
                        "timestamp_full",
                        "distance_km",
                        "dt_minutes",
                        "latitude",
                        "longitude",
                        "prev_latitude",
                        "prev_longitude",
                        "port",
                        "stable_id",
                    ]
                    if c in events.columns
                ]
                table = events[table_cols].copy()
                if {"timestamp_full", "distance_km"}.issubset(events.columns):
                    chart = (
                        events.assign(
                            timestamp_dt=pd.to_datetime(events["timestamp_full"], errors="coerce", utc=True)
                        )
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
        else:
            result = kpi.detect_arrival_spikes(port=port, start=start, end=end)

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=user_filters,
            top_k=top_k_evidence,
            include_dates=True,
        )
        return result, evidence

    result = kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window)
    evidence = _retrieve_evidence(
        retriever=retriever,
        question=question,
        entities=entities,
        overrides=user_filters,
        top_k=top_k_evidence,
        include_dates=True,
    )
    return result, evidence


def _render_compact_result(
    result: Union[AnalyticsResult, ForecastResult],
    evidence: EvidenceBundle,
) -> None:
    def _fallback_evidence_from_result(
        value: Union[AnalyticsResult, ForecastResult],
        max_items: int = 5,
    ) -> List[str]:
        lines: List[str] = []

        if isinstance(value, ForecastResult):
            anchor_values_note = next(
                (n for n in value.coverage_notes if n.startswith("Analog values used:")),
                None,
            )
            anchor_dates_note = next(
                (n for n in value.coverage_notes if n.startswith("Analog dates used:")),
                None,
            )
            if anchor_values_note:
                lines.append(anchor_values_note)
            elif anchor_dates_note:
                lines.append(anchor_dates_note)

            if value.history is not None and not value.history.empty:
                hist = value.history.copy()
                if "date" in hist.columns and "actual" in hist.columns:
                    hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=True).dt.floor("D")
                    hist = hist.dropna(subset=["date", "actual"]).sort_values("date")
                    for _, row in hist.tail(max_items).iterrows():
                        lines.append(
                            f"Historical point | {row['date'].strftime('%Y-%m-%d')} | value={float(row['actual']):.2f}"
                        )

            if value.forecast is not None and not value.forecast.empty:
                fdf = value.forecast.copy()
                if "date" in fdf.columns and "predicted" in fdf.columns:
                    fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce", utc=True).dt.floor("D")
                    fdf = fdf.dropna(subset=["date", "predicted"]).sort_values("date")
                    for _, row in fdf.tail(min(2, max_items)).iterrows():
                        lower = float(row["lower"]) if "lower" in row and pd.notna(row["lower"]) else float("nan")
                        upper = float(row["upper"]) if "upper" in row and pd.notna(row["upper"]) else float("nan")
                        lines.append(
                            f"Forecast target | {row['date'].strftime('%Y-%m-%d')} | "
                            f"pred={float(row['predicted']):.2f}, range={lower:.2f}-{upper:.2f}"
                        )
            return lines[:max_items]

        if value.table is not None and not value.table.empty:
            tdf = value.table.head(max_items).copy()
            for _, row in tdf.iterrows():
                fragments: List[str] = []
                for col in tdf.columns[:4]:
                    cell = row[col]
                    if pd.isna(cell):
                        continue
                    if isinstance(cell, pd.Timestamp):
                        rendered = cell.strftime("%Y-%m-%d")
                    else:
                        rendered = str(cell)
                    fragments.append(f"{col}={rendered}")
                if fragments:
                    lines.append(" | ".join(fragments))
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
            steps.append("Applied active filters (port/date/vessel-type) to the congestion history for this query.")
            for note in value.coverage_notes:
                if (
                    note.startswith("Coverage window:")
                    or note.startswith("Rows used:")
                    or note.startswith("Target date:")
                    or note.startswith("Forecast target weekday:")
                    or note.startswith("Analog dates used:")
                    or note.startswith("Analog values used:")
                    or note.startswith("Meaning:")
                ):
                    steps.append(note)
            method = next((n for n in value.coverage_notes if n.startswith("Method:")), None)
            if method:
                steps.append(method)
            steps.append("Computed point estimate and uncertainty interval (lower-upper) for the requested target.")
        else:
            steps.append("Applied active filters (port/date/vessel-type/anomaly) to KPI tables.")
            for note in value.coverage_notes:
                if note.startswith("Coverage window:") or note.startswith("Rows used:") or note.startswith("Data sources used:"):
                    steps.append(note)
            if value.table is not None and not value.table.empty:
                steps.append(f"Aggregated filtered rows into {len(value.table):,} output row(s) using deterministic pandas operations.")
            else:
                steps.append("Computed deterministic metric directly from the filtered subset.")

        assumptions = [c for c in value.caveats if not c.lower().startswith("confidence:")]
        for assumption in assumptions[:2]:
            steps.append(f"Assumption: {assumption}")

        deduped: List[str] = []
        for step in steps:
            if step and step not in deduped:
                deduped.append(step)
        return deduped

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

            actions.append(
                f"Use predicted range {lower:.2f}-{upper:.2f} to set staffing floors/ceilings instead of a single-point plan."
            )
            if spread >= 0.6:
                actions.append("Maintain operational contingency: uncertainty is wide, so add tug/pilot standby margin.")
            if "low" in confidence:
                actions.append("Refresh this forecast 24-48 hours before execution because confidence is currently low.")
            actions.append("Use retrieved analog evidence rows to brief operations with concrete historical precedents.")
            return actions

        answer_text = value.answer.lower()
        if "jump" in answer_text or "anomaly" in answer_text:
            actions.append("Open AIS integrity checks for listed MMSI and validate with external tracking feeds.")
            actions.append("Flag suspicious tracks for VTS review before acting on route deviations.")
            actions.append("Prioritize vessels with repeated jump flags for manual watchlist monitoring.")
        else:
            actions.append("Use the daily/weekly pattern in the chart to plan shift staffing and pilot windows.")
            actions.append("Re-run this query with tighter vessel-type filters for targeted operational planning.")
            actions.append("Apply port-vs-port comparisons to rebalance pilot/tug resources across nearby ports.")
        return actions

    def _build_evidence_backed_answer(
        value: Union[AnalyticsResult, ForecastResult],
        bundle: EvidenceBundle,
    ) -> Optional[str]:
        if value.status == "ok" or not bundle.rows:
            return None
        df = pd.DataFrame(bundle.rows)
        if df.empty:
            return None
        port_text = "unknown port"
        if "port" in df.columns and df["port"].notna().any():
            top_port = (
                df["port"].dropna().astype(str).value_counts().index.tolist()[:1]
            )
            if top_port:
                port_text = top_port[0]
        date_span = ""
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
            if not ts.empty:
                date_span = f" between {ts.min().strftime('%Y-%m-%d')} and {ts.max().strftime('%Y-%m-%d')}"
        return (
            f"Direct KPI aggregation returned no exact match, but vector retrieval found {len(df)} relevant records "
            f"for {port_text}{date_span}. This is evidence-backed retrieval output, not a deterministic aggregate."
        )

    def _to_naive_datetime(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_convert(None)

    def _render_chart(value: Union[AnalyticsResult, ForecastResult]) -> None:
        st.subheader("Chart")

        if isinstance(value, ForecastResult):
            hist = pd.DataFrame()
            if value.history is not None and not value.history.empty and {"date", "actual"}.issubset(value.history.columns):
                hist = value.history[["date", "actual"]].copy()
                hist["date"] = _to_naive_datetime(hist["date"]).dt.floor("D")
                hist = hist.dropna(subset=["date"]).sort_values("date")

            fc = pd.DataFrame()
            if value.forecast is not None and not value.forecast.empty and {"date", "predicted"}.issubset(value.forecast.columns):
                cols = [c for c in ("date", "predicted", "lower", "upper") if c in value.forecast.columns]
                fc = value.forecast[cols].copy()
                fc["date"] = _to_naive_datetime(fc["date"]).dt.floor("D")
                fc = fc.dropna(subset=["date"]).sort_values("date")

            if hist.empty and fc.empty:
                st.info("No chartable series for this response.")
                return

            gap_days = 0
            if not hist.empty and not fc.empty:
                gap_days = int((fc["date"].min() - hist["date"].max()).days)

            if gap_days > 60:
                st.caption("Recent historical series used for baseline.")
                hist_tail = hist.tail(90).copy()
                if alt is not None and not hist_tail.empty:
                    c1 = (
                        alt.Chart(hist_tail)
                        .mark_line(color="#60a5fa", point=True)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("actual:Q", title="Observed value"),
                            tooltip=["date:T", alt.Tooltip("actual:Q", format=".2f")],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(c1, use_container_width=True)
                elif not hist_tail.empty:
                    st.line_chart(hist_tail.set_index("date")[["actual"]], use_container_width=True)

                if not fc.empty:
                    st.caption("Prediction interval for requested target date.")
                    fc_tail = fc.tail(1).copy()
                    if alt is not None:
                        band = (
                            alt.Chart(fc_tail)
                            .mark_rule(color="#f59e0b", strokeWidth=4)
                            .encode(
                                x=alt.X("date:T", title="Target date"),
                                y=alt.Y("lower:Q", title="Forecast"),
                                y2="upper:Q",
                                tooltip=[
                                    "date:T",
                                    alt.Tooltip("predicted:Q", format=".2f"),
                                    alt.Tooltip("lower:Q", format=".2f"),
                                    alt.Tooltip("upper:Q", format=".2f"),
                                ],
                            )
                        )
                        point = alt.Chart(fc_tail).mark_point(color="#ef4444", size=120, filled=True).encode(
                            x="date:T",
                            y="predicted:Q",
                        )
                        st.altair_chart((band + point).properties(height=220), use_container_width=True)
                    else:
                        st.dataframe(fc_tail, use_container_width=True, hide_index=True)
                return

            hist_tail = hist.tail(120).copy()
            frames: List[pd.DataFrame] = []
            if not hist_tail.empty:
                frames.append(hist_tail.assign(series="actual", value=hist_tail["actual"])[["date", "series", "value"]])
            if not fc.empty:
                frames.append(fc.assign(series="predicted", value=fc["predicted"])[["date", "series", "value"]])
            long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            if alt is not None and not long_df.empty:
                chart = (
                    alt.Chart(long_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("value:Q", title="Value"),
                        color=alt.Color("series:N", scale=alt.Scale(range=["#60a5fa", "#f59e0b"])),
                        tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                if not fc.empty and {"lower", "upper"}.issubset(fc.columns):
                    interval = (
                        alt.Chart(fc)
                        .mark_area(color="#f59e0b", opacity=0.15)
                        .encode(x="date:T", y="lower:Q", y2="upper:Q")
                    )
                    chart = interval + chart
                st.altair_chart(chart, use_container_width=True)
            elif not long_df.empty:
                wide = long_df.pivot_table(index="date", columns="series", values="value", aggfunc="mean").sort_index()
                st.line_chart(wide, use_container_width=True)
            return

        if value.chart is None or value.chart.empty:
            st.info("No chartable series for this response.")
            return

        chart_df = value.chart.copy()
        if isinstance(chart_df.index, pd.DatetimeIndex):
            plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "x"})
            plot_df["x"] = _to_naive_datetime(plot_df["x"]).dt.floor("D")
            value_col = [c for c in plot_df.columns if c != "x"][0]
            if alt is not None:
                chart = (
                    alt.Chart(plot_df.dropna(subset=["x"]))
                    .mark_line(color="#60a5fa", point=True)
                    .encode(
                        x=alt.X("x:T", title="Date"),
                        y=alt.Y(f"{value_col}:Q", title=value_col.replace("_", " ").title()),
                        tooltip=["x:T", alt.Tooltip(f"{value_col}:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.line_chart(chart_df, use_container_width=True)
        else:
            plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "x"})
            value_col = [c for c in plot_df.columns if c != "x"][0]
            if alt is not None:
                chart = (
                    alt.Chart(plot_df)
                    .mark_bar(color="#60a5fa")
                    .encode(
                        x=alt.X("x:N", title=plot_df.columns[0], sort="-y"),
                        y=alt.Y(f"{value_col}:Q", title=value_col.replace("_", " ").title()),
                        tooltip=[plot_df.columns[0], alt.Tooltip(f"{value_col}:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(chart_df, use_container_width=True)

    st.subheader("Answer")
    st.write(result.answer)
    evidence_backed = _build_evidence_backed_answer(result, evidence)
    if evidence_backed:
        st.info(evidence_backed)

    if isinstance(result, ForecastResult):
        meaning_note = next((n for n in result.coverage_notes if n.startswith("Meaning:")), None)
        if meaning_note:
            st.subheader("Forecast Meaning")
            st.write(meaning_note.split(":", 1)[1].strip())

    st.subheader("Evidence")
    retrieved_lines = evidence.lines
    computed_lines = _fallback_evidence_from_result(result)

    if computed_lines:
        st.markdown("**Computed evidence used for this answer**")
        for line in computed_lines:
            st.markdown(f"- {line}")
    if retrieved_lines:
        st.markdown("**Retrieved supporting evidence**")
        for line in retrieved_lines:
            st.markdown(f"- {line}")
    if not retrieved_lines and not computed_lines:
        st.info("No evidence rows were available for this response.")

    st.subheader("Confidence")
    st.write(_extract_confidence_label(result))

    _render_chart(result)

    method_steps = _build_method_steps(result)
    if method_steps:
        st.subheader("How This Was Computed")
        for idx, step in enumerate(method_steps, start=1):
            st.markdown(f"{idx}. {step}")

    st.subheader("Port Operations Recommendations")
    for action in _build_port_actions(result):
        st.markdown(f"- {action}")

    st.subheader("Retrieval Provenance")
    trace = evidence.trace or {}
    status = str(trace.get("retrieval_status", "unknown")).upper()
    reason = str(trace.get("reason", "No retrieval status available."))
    st.write(f"Status: `{status}`")
    st.write(reason)
    if trace:
        st.write(
            f"Collection: `{trace.get('collection', 'n/a')}` | "
            f"Backend: `{trace.get('vector_backend', 'n/a')}` | "
            f"Mode: `{trace.get('mode', 'n/a')}` | "
            f"Latency: `{trace.get('query_latency_ms', 'n/a')} ms` | "
            f"Returned: `{trace.get('returned_items', 0)}`"
        )
        where_used = trace.get("where_filter")
        if where_used not in (None, "", {}):
            st.write(f"Where filter: `{where_used}`")
    if evidence.rows:
        trace_df = pd.DataFrame(evidence.rows)
        cols = [c for c in ["vector_id", "chunk_id", "distance", "timestamp", "port", "vessel_type", "mmsi"] if c in trace_df.columns]
        st.dataframe(trace_df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No vector rows available for this query. Evidence above may come from deterministic KPI computation.")
    with st.expander("Raw retrieval trace JSON", expanded=False):
        st.json(trace)


def main() -> None:
    st.set_page_config(page_title="Eagle Eye", layout="wide")
    st.title("Eagle Eye")
    st.caption("Eagle Eye for traffic patterns, anomalies, and future congestion predictions.")

    config_path = "config/config.yaml"
    config = load_config(config_path)
    configured_processed_dir = Path(config.get("predict", {}).get("processed_dir", "data/processed"))
    processed_bootstrap_changed, processed_bootstrap_message = _maybe_bootstrap_processed_bundle(
        configured_processed_dir
    )
    events_bootstrap_changed, events_bootstrap_message = _maybe_bootstrap_events_bundle(
        configured_processed_dir
    )
    default_processed_dir, using_demo_processed = _resolve_processed_dir(configured_processed_dir)
    configured_persist_dir = Path(config["paths"].get("persist_dir", "data/chroma"))
    chroma_bootstrap_changed = False
    chroma_bootstrap_message = ""
    requested_vector_mode = str(
        os.getenv("VECTOR_DB_MODE", config.get("vector_db", {}).get("mode", "local"))
    ).strip().lower()
    using_remote_vector = _remote_vector_enabled(config)
    if using_remote_vector:
        persist_dir = configured_persist_dir
        using_demo_chroma = False
    else:
        chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_bundle(
            configured_persist_dir
        )
        persist_dir, using_demo_chroma = _resolve_persist_dir(configured_persist_dir)

    with st.sidebar:
        st.subheader("Ask Settings")
        top_k_evidence = st.slider("Evidence top K", min_value=1, max_value=8, value=5)
        st.caption("Keep questions specific (port + date helps).")
        if using_demo_processed:
            st.info("Running with bundled demo processed data (`demo_data/processed`).")
            if "APP_PROCESSED_BUNDLE_URL" in processed_bootstrap_message or "Downloaded" in processed_bootstrap_message:
                st.warning(processed_bootstrap_message)
        elif processed_bootstrap_changed:
            st.info(processed_bootstrap_message)
        elif "No APP_PROCESSED_BUNDLE_URL configured." not in processed_bootstrap_message:
            st.warning(processed_bootstrap_message)
        if events_bootstrap_changed:
            st.info(events_bootstrap_message)
        if using_remote_vector:
            st.info("Using remote Chroma service (configured via CHROMA_* / VECTOR_DB_MODE).")
        elif chroma_bootstrap_changed:
            st.info(chroma_bootstrap_message)
        if using_demo_chroma:
            st.info("Running with bundled demo vector index (`demo_data/chroma`).")
            st.caption("Full retrieval parity with local requires a remote Chroma service because the full local vector store is too large for cloud packaging.")
        elif chroma_bootstrap_message and "No APP_CHROMA_BUNDLE_URL configured." not in chroma_bootstrap_message:
            st.caption(chroma_bootstrap_message)
        if requested_vector_mode in {"remote", "http"} and not using_remote_vector:
            st.warning("VECTOR_DB_MODE is remote but CHROMA_HOST is missing/invalid; using local/demo index.")
        if not using_demo_processed and not processed_bootstrap_changed:
            st.caption(f"Processed runtime path: {default_processed_dir}")

    try:
        kpi_engine = _init_kpi_engine(str(default_processed_dir))
        forecast_engine = _init_forecast_engine(str(default_processed_dir))
    except Exception as exc:
        st.error(f"Could not initialize data engines: {exc}")
        st.info("Run `./run_demo_pipeline.sh` first.")
        st.stop()

    retriever: Optional[RAGRetriever] = None
    retriever_reason = ""
    api_key, key_source = _load_openai_api_key_from_runtime()
    if api_key:
        try:
            retriever = _init_retriever(persist_dir=str(persist_dir), config_path=config_path)
            retriever_reason = (
                f"Retriever active (API key source: {key_source}, backend: {retriever.vector_backend})."
            )
        except Exception as exc:
            retriever = None
            retriever_reason = f"Retriever init failed: {exc}"
            if using_remote_vector:
                chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_bundle(
                    configured_persist_dir
                )
                fallback_persist_dir, fallback_using_demo_chroma = _resolve_persist_dir(configured_persist_dir)
                if (fallback_persist_dir / "chroma.sqlite3").exists():
                    try:
                        retriever = _init_retriever(
                            persist_dir=str(fallback_persist_dir),
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
                        retriever = None
                        retriever_reason = (
                            f"Remote retriever failed ({exc}); local fallback failed ({local_exc})."
                        )
    else:
        retriever_reason = "Retriever unavailable: `OPENAI_API_KEY` not found in environment or Streamlit secrets."
        with st.sidebar:
            st.warning("Set `OPENAI_API_KEY` in app secrets to enable vector retrieval evidence.")

    with st.sidebar:
        if "Fell back to local vector store" in retriever_reason:
            st.warning(retriever_reason)

    st.session_state["retriever_reason"] = retriever_reason
    events_path = configured_processed_dir / "events.parquet"
    if not events_path.exists():
        events_path = default_processed_dir / "events.parquet"

    st.subheader("Sample Queries")
    if "ask_question" not in st.session_state:
        st.session_state["ask_question"] = SAMPLE_QUERIES[0]

    selected = st.selectbox("Try a sample query", options=SAMPLE_QUERIES, index=0)
    if st.button("Load sample query"):
        st.session_state["ask_question"] = selected

    st.subheader("Ask")
    st.text_area("Question", key="ask_question", height=90)

    with st.expander("Optional filters", expanded=False):
        ui_port = st.text_input("Port / LOCODE", value="")
        ui_date_from = st.text_input("From date (YYYY-MM-DD)", value="")
        ui_date_to = st.text_input("To date (YYYY-MM-DD)", value="")
        ui_vessel_type = st.text_input("Vessel type", value="")
        ui_anomaly = st.selectbox("Anomaly flag", options=["any", "true", "false"], index=0)

    ask = st.button("Ask", type="primary")
    if not ask:
        return

    question = st.session_state.get("ask_question", "").strip()
    if not question:
        st.warning("Enter a question first.")
        return

    intent_result = classify_question(question)

    user_filters: Dict[str, Any] = {
        "port": ui_port or None,
        "date_from": ui_date_from or None,
        "date_to": ui_date_to or None,
        "vessel_type": ui_vessel_type or None,
        "anomaly": _parse_anomaly_filter(ui_anomaly),
    }

    result, evidence = _handle_ask_question(
        question=question,
        intent_result=intent_result,
        kpi=kpi_engine,
        forecaster=forecast_engine,
        retriever=retriever,
        top_k_evidence=top_k_evidence,
        user_filters=user_filters,
        events_path=events_path if events_path.exists() else None,
    )

    _render_compact_result(result=result, evidence=evidence)


if __name__ == "__main__":
    main()
