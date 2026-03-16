"""Simplified Ask-only Streamlit app with integrated future prediction."""

from __future__ import annotations

import time
import os
import re
import unicodedata
from difflib import SequenceMatcher
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
from src.carbon.query import CarbonQueryEngine, CarbonResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.qa.intent import IntentResult, classify_question
from src.rag.retriever import QueryFilters, RAGRetriever
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet
from src.utils.cloud_bootstrap import ensure_bundle, ensure_file_manifest
from src.utils.config import load_config
from src.utils.runtime import chroma_remote_settings, force_local_vector_env
from src.utils.serialization import compact_traffic_evidence


SAMPLE_QUERIES_BY_CATEGORY: Dict[str, List[str]] = {
    "Traffic Monitoring": [
        "How many vessel arrivals were recorded at SEGOT in March 2022?",
        "Which weekday is usually busiest at LVVNT?",
        "Compare Friday and Monday arrivals at GDANSK in March 2022.",
    ],
    "Vessel Investigation": [
        "For MMSI 266232000, how long was the vessel in port on 2021-03-01?",
        "Show suspicious AIS jumps for MMSI 246521000 on 2022-03-10.",
    ],
    "Forecast Planning": [
        "What will congestion be at LVVNT on Friday, February 20, 2026?",
        "Predict congestion for SEGOT next Friday based on historical patterns.",
        "Expected congestion at GDANSK on 2026-03-06?",
        "Compare expected congestion next Friday between LVVNT and SEGOT.",
        "Predict whether LUBECK is likely high congestion on 2026-02-20.",
    ],
    "Carbon & Emissions": [
        "What are TTW emissions at SEGOT in March 2022 for CO2, NOx, SOx, and PM?",
        "Show WTW CO2e emissions at LVVNT between 2022-02-01 and 2022-02-28.",
        "Carbon emissions for SEGOT by month in 2022.",
    ],
    "Unsupported Scope": [
        "What is crane utilization at berth 3 in SEGOT today?",
        "What is gate queue length at Port of Gdansk right now?",
    ],
}

PORT_ALIAS_TO_CODE: Dict[str, str] = {
    "gothenburg": "SEGOT",
    "goteborg": "SEGOT",
    "goteborgs": "SEGOT",
    "gdansk": "PLGDN",
    "gdynia": "PLGDY",
    "klaipeda": "LTKLJ",
    "riga": "LVRIX",
    "kotka": "FIKTK",
    "swinoujscie": "PLSWI",
    "szczecin": "PLSZZ",
    "sodertalje": "SESOE",
}


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
def _init_carbon_engine(processed_dir: str, factor_registry_path: str, monte_carlo_draws: int) -> CarbonQueryEngine:
    return CarbonQueryEngine(
        processed_dir=processed_dir,
        factor_registry_path=factor_registry_path,
        monte_carlo_draws=monte_carlo_draws,
        auto_build=True,
    )


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

    manifest_url, manifest_source = _load_runtime_setting("APP_CHROMA_MANIFEST_URL")
    if manifest_url:
        changed, message = ensure_file_manifest(
            url=manifest_url,
            target_dir=preferred_dir,
            required_files=required_files,
            timeout_seconds=3600,
        )
        if manifest_source != "missing":
            message = f"{message} Source: {manifest_source}."
        return changed, message

    bundle_url, source = _load_runtime_setting("APP_CHROMA_BUNDLE_URL")
    if not bundle_url:
        return False, "No APP_CHROMA_MANIFEST_URL or APP_CHROMA_BUNDLE_URL configured."

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


def _normalize_text_token(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^port of\s+", "", text)
    return text


def _resolve_port_token(port_token: Optional[str], kpi: KPIQueryEngine) -> Optional[str]:
    token = (port_token or "").strip()
    if not token:
        return None
    if re.fullmatch(r"[A-Za-z]{2}\s?[A-Za-z]{3}", token):
        return token.upper().replace(" ", "")
    norm = _normalize_text_token(token)
    alias_code = PORT_ALIAS_TO_CODE.get(norm)
    if alias_code:
        return alias_code

    catalog = kpi.port_catalog
    if catalog.empty:
        return token

    code = token.upper().replace(" ", "")

    work = catalog.copy()
    for col in ("port_key", "locode_norm", "port_label", "port_name_norm"):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    if "arrivals_total" not in work.columns:
        work["arrivals_total"] = 0
    work["arrivals_total"] = pd.to_numeric(work["arrivals_total"], errors="coerce").fillna(0)
    work["source_kind"] = work.get("source_kind", "").fillna("").astype(str).str.lower()
    work["locode_norm"] = work.get("locode_norm", "").fillna("").astype(str).str.upper()
    work["is_structured_port"] = (
        (work["source_kind"] == "port_call")
        & work["locode_norm"].str.fullmatch(r"[A-Z]{5}")
    )

    exact_code = work[
        (work["port_key"].str.upper() == code) | (work["locode_norm"].str.upper() == code)
    ]
    if not exact_code.empty:
        row = exact_code.sort_values("arrivals_total", ascending=False).iloc[0]
        return str(row.get("port_key") or row.get("locode_norm") or token).strip()

    work["port_label_norm"] = work["port_label"].map(_normalize_text_token)
    work["port_name_norm_clean"] = work["port_name_norm"].map(_normalize_text_token)
    contains = work[
        work["port_label_norm"].str.contains(norm, regex=False)
        | work["port_name_norm_clean"].str.contains(norm, regex=False)
    ]
    if not contains.empty:
        if contains["is_structured_port"].any():
            contains = contains[contains["is_structured_port"]]
        row = contains.sort_values("arrivals_total", ascending=False).iloc[0]
        return str(row.get("port_key") or row.get("locode_norm") or token).strip()

    def _best_similarity(row: pd.Series) -> float:
        cand_a = str(row.get("port_label_norm", ""))
        cand_b = str(row.get("port_name_norm_clean", ""))
        return max(
            SequenceMatcher(None, norm, cand_a).ratio() if cand_a else 0.0,
            SequenceMatcher(None, norm, cand_b).ratio() if cand_b else 0.0,
        )

    work["similarity"] = work.apply(_best_similarity, axis=1)
    fuzzy = work[work["similarity"] >= 0.80]
    if not fuzzy.empty:
        if fuzzy["is_structured_port"].any():
            fuzzy = fuzzy[fuzzy["is_structured_port"]]
        row = fuzzy.sort_values(["similarity", "arrivals_total"], ascending=[False, False]).iloc[0]
        return str(row.get("port_key") or row.get("locode_norm") or token).strip()

    return token


def _resolve_ports(port_tokens: List[str], kpi: KPIQueryEngine) -> List[str]:
    resolved: List[str] = []
    for token in port_tokens:
        mapped = _resolve_port_token(token, kpi) or token
        if mapped not in resolved:
            resolved.append(mapped)
    return resolved


def _derive_answer_source(
    result: Union[AnalyticsResult, ForecastResult, CarbonResult],
    evidence: EvidenceBundle,
) -> tuple[str, str]:
    if isinstance(result, CarbonResult):
        label = result.source_label
        if evidence.rows and label.startswith("Computed"):
            label = "Hybrid (computed + retrieved supporting evidence)"
        detail = (
            "Carbon metrics are deterministic inventory outputs from AIS + port-call segmentation. "
            "Vector retrieval is optional supporting evidence."
        )
        return label, detail

    data_source_note = next(
        (n for n in result.coverage_notes if n.startswith("Data sources used:")),
        "",
    )
    source_text = data_source_note.replace("Data sources used:", "").strip().lower()
    retrieval_status = str((evidence.trace or {}).get("retrieval_status", "")).lower()
    has_vector_rows = bool(evidence.rows)

    if isinstance(result, ForecastResult):
        label = "Computed from historical congestion proxy series"
    elif "port_call" in source_text and "ais_destination_proxy" in source_text:
        label = "Hybrid computed answer (port-call + AIS proxy)"
    elif "port_call" in source_text:
        label = "Computed from port-call records"
    elif "ais_destination_proxy" in source_text:
        label = "Computed from AIS-derived proxy logic"
    elif retrieval_status in {"ok", "no_hits", "computed_only"} or has_vector_rows:
        label = "Retrieved from evidence chunks"
    else:
        label = "Computed answer"

    detail = (
        "Numeric outputs come from deterministic KPI/forecast computation; "
        "vector retrieval is used for supporting evidence and traceability."
    )
    if retrieval_status == "computed_only":
        detail = (
            "Primary evidence came from deterministic row-level computation; "
            "vector retrieval had no matching rows for this query."
        )
    return label, detail


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
    carbon: CarbonQueryEngine,
    retriever: Optional[RAGRetriever],
    top_k_evidence: int,
    user_filters: Dict[str, Any],
    events_path: Optional[Path],
) -> tuple[Union[AnalyticsResult, ForecastResult, CarbonResult], EvidenceBundle]:
    entities = intent_result.entities
    q_lower = question.lower()

    raw_port = _pick_filter(user_filters.get("port"), entities.get("port"))
    port = _resolve_port_token(raw_port, kpi)
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
    ports = _resolve_ports(ports, kpi)

    if start and end:
        start_ts = pd.to_datetime(start, errors="coerce", utc=True)
        end_ts = pd.to_datetime(end, errors="coerce", utc=True)
        if pd.notna(start_ts) and pd.notna(end_ts) and start_ts > end_ts:
            return (
                KPIQueryEngine.no_data("Invalid date range: `From date` is after `To date`."),
                EvidenceBundle(lines=[], rows=[], trace={}),
            )

    evidence_overrides = dict(user_filters)
    if port:
        evidence_overrides["port"] = port

    if intent_result.intent == "G":
        return (
            KPIQueryEngine.unsupported(
                "This question needs terminal operations data (berth/crane/TEU/gate), which is not in PRJ912/PRJ896."
            ),
            EvidenceBundle(lines=[], rows=[], trace={}),
        )

    if intent_result.intent == "H":
        result = carbon.from_question_entities(question=question, entities=entities, user_filters=user_filters)
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if result.status == "ok" and evidence.rows:
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
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
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

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
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
                overrides=evidence_overrides,
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
            overrides=evidence_overrides,
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
            overrides=evidence_overrides,
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
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        return result, evidence

    if intent_result.intent == "F":
        if any(token in q_lower for token in ("jump", "spoof", "teleport", "impossible")):
            filters = _make_rag_filters(entities=entities, overrides=evidence_overrides, include_dates=True)
            jump_result: Dict[str, Any]
            jump_source = ""
            if events_path and events_path.exists():
                jump_result = detect_sudden_jump_events_from_parquet(
                    events_path=events_path,
                    mmsi=filters.mmsi,
                    date_from=filters.date_from,
                    date_to=filters.date_to,
                )
                jump_source = "row-level AIS events parquet"
            elif retriever is not None:
                jump_result = retriever.detect_sudden_jumps(filters=filters)
                jump_source = "AIS metadata index"
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
                        "implied_speed_kn",
                        "dt_minutes",
                        "trigger_rule",
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

            if count > 0:
                answer = f"Detected {count} potential sudden AIS coordinate jumps in the filtered range."
            else:
                answer = "No sudden AIS coordinate jumps were detected in the filtered range."

            result = AnalyticsResult(
                status="ok",
                answer=answer,
                table=table,
                chart=chart,
                coverage_notes=[
                    f"Rows used: {count}",
                    f"Data sources used: {jump_source}",
                ],
                caveats=[
                    "Jump rule: distance >= 20 km within 30 minutes, or implied speed >= 40 kn with >= 5 km displacement.",
                    "This is a heuristic anomaly indicator, not proof of spoofing.",
                ],
            )
        else:
            result = kpi.detect_arrival_spikes(port=port, start=start, end=end)

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if (
            any(token in q_lower for token in ("jump", "spoof", "teleport", "impossible"))
            and result.table is not None
            and not result.table.empty
            and not evidence.rows
        ):
            trace = dict(evidence.trace or {})
            trace["retrieval_status"] = "computed_only"
            trace["reason"] = (
                "Vector retrieval returned no hits; evidence is from row-level AIS jump computation."
            )
            evidence = EvidenceBundle(lines=evidence.lines, rows=evidence.rows, trace=trace)
        return result, evidence

    result = kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window)
    evidence = _retrieve_evidence(
        retriever=retriever,
        question=question,
        entities=entities,
        overrides=evidence_overrides,
        top_k=top_k_evidence,
        include_dates=True,
    )
    return result, evidence


def _render_compact_result(
    result: Union[AnalyticsResult, ForecastResult, CarbonResult],
    evidence: EvidenceBundle,
    show_technical: bool,
) -> None:
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

    def _extract_confidence_label(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> str:
        if isinstance(value, CarbonResult):
            return f"{value.confidence_label} ({value.confidence_reason})"
        if value.status != "ok":
            return "low (insufficient matched data/evidence)"
        for note in value.caveats:
            if note.lower().startswith("confidence:"):
                return note.split(":", 1)[1].strip()
        if isinstance(value, ForecastResult):
            return "medium (forecast based on available historical patterns)"
        return "medium (deterministic aggregation over filtered rows)"

    def _to_analyst_evidence_line(line: str) -> str:
        if "|" not in line:
            return line
        return line.split("|", maxsplit=3)[-1].strip()

    def _build_recommendation_triggers(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
        triggers: List[str] = []
        if isinstance(value, CarbonResult):
            co2e = value.uncertainty_interval.get("CO2e") or value.uncertainty_interval.get("CO2")
            if co2e:
                triggers.append(
                    f"Trigger: {value.boundary} {('CO2e' if 'CO2e' in value.uncertainty_interval else 'CO2')} point={co2e.get('point', 0):.2f}."
                )
                triggers.append(
                    f"Trigger: uncertainty interval {co2e.get('lower', 0):.2f}-{co2e.get('upper', 0):.2f}."
                )
            triggers.append(f"Trigger: source label = {value.source_label}.")
            return triggers

        if isinstance(value, ForecastResult) and value.forecast is not None and not value.forecast.empty:
            pred = float(value.forecast["predicted"].mean())
            upper = float(value.forecast["upper"].mean()) if "upper" in value.forecast.columns else pred
            lower = float(value.forecast["lower"].mean()) if "lower" in value.forecast.columns else pred
            if pred >= 1.8:
                triggers.append("Trigger: forecast congestion index >= 1.80 (high-pressure band).")
            elif pred >= 1.3:
                triggers.append("Trigger: forecast congestion index in 1.30-1.79 (elevated band).")
            else:
                triggers.append("Trigger: forecast congestion index < 1.30 (normal-to-low band).")
            triggers.append(f"Trigger: uncertainty interval {lower:.2f}-{upper:.2f} used to size staffing buffer.")
            return triggers

        answer_text = value.answer.lower()
        if "jump" in answer_text or "anomaly" in answer_text:
            if value.table is not None and not value.table.empty:
                max_dist = (
                    float(pd.to_numeric(value.table.get("distance_km"), errors="coerce").max())
                    if "distance_km" in value.table.columns
                    else None
                )
                triggers.append("Trigger: anomaly heuristic matched at least one AIS jump event.")
                if max_dist is not None and not pd.isna(max_dist):
                    triggers.append(f"Trigger: max detected displacement {max_dist:.2f} km in a short interval.")
            else:
                triggers.append("Trigger: anomaly heuristic did not find qualifying jump events.")
            return triggers

        if value.chart is not None and not value.chart.empty:
            first_col = [c for c in value.chart.columns if c != "date"]
            if first_col:
                metric_col = first_col[0]
                metric_values = pd.to_numeric(value.chart[metric_col], errors="coerce").dropna()
                if not metric_values.empty:
                    triggers.append(
                        f"Trigger: planning recommendation based on observed {metric_col} range "
                        f"{metric_values.min():.2f}-{metric_values.max():.2f}."
                    )
        if not triggers:
            triggers.append("Trigger: recommendation generated from filtered KPI summary for the selected window.")
        return triggers

    def _build_method_steps(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
        steps: List[str] = []
        if isinstance(value, CarbonResult):
            steps.append("Applied deterministic AIS + port-call mode segmentation (transit/manoeuvring/berth/anchorage).")
            steps.append(f"Boundary: {value.boundary}; Pollutants: {', '.join(value.pollutants)}.")
            steps.append(f"Source label: {value.source_label}.")
            steps.append(f"Factor params version: {value.params_version}.")
            steps.append(f"Confidence: {value.confidence_label} ({value.confidence_reason})")
            for note in value.coverage_notes[:6]:
                steps.append(note)
            return steps

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

    def _build_port_actions(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
        actions: List[str] = []
        if isinstance(value, CarbonResult):
            co2e = value.uncertainty_interval.get("CO2e") or value.uncertainty_interval.get("CO2")
            point = float((co2e or {}).get("point", 0.0))
            upper = float((co2e or {}).get("upper", point))
            if point >= 50:
                actions.append("Prioritize shore-power and berth energy optimization on high-emission windows.")
                actions.append("Coordinate pilot/tug sequencing to reduce manoeuvring fuel intensity.")
            elif point >= 15:
                actions.append("Apply targeted slow-steaming and auxiliary-load controls for inbound traffic.")
                actions.append("Use emissions view for berth allocation on high-intensity dates.")
            else:
                actions.append("Maintain baseline operations; monitor emissions drift against this baseline.")
            if upper > point * 1.4:
                actions.append("Uncertainty is wide: refresh with latest AIS coverage before final ops decisions.")
            actions.append("Audit supporting evidence IDs before publishing inventory outputs externally.")
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
        value: Union[AnalyticsResult, ForecastResult, CarbonResult],
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

    def _render_chart(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> None:
        st.subheader("Chart")

        if isinstance(value, CarbonResult):
            if value.chart is None or value.chart.empty:
                st.info("No chartable carbon series for this response.")
                return
            chart_df = value.chart.copy()
            if isinstance(chart_df.index, pd.DatetimeIndex):
                plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "x"})
                plot_df["x"] = _to_naive_datetime(plot_df["x"]).dt.floor("D")
                value_col = [c for c in plot_df.columns if c != "x"][0]
                if alt is not None:
                    line = (
                        alt.Chart(plot_df.dropna(subset=["x"]))
                        .mark_line(color="#22c55e", point=True)
                        .encode(
                            x=alt.X("x:T", title="Date"),
                            y=alt.Y(f"{value_col}:Q", title=value_col),
                            tooltip=["x:T", alt.Tooltip(f"{value_col}:Q", format=".3f")],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(line, use_container_width=True)
                else:
                    st.line_chart(chart_df, use_container_width=True)
            else:
                st.dataframe(chart_df, use_container_width=True, hide_index=True)
            return

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

    source_label, source_detail = _derive_answer_source(result, evidence)
    st.subheader("Answer Source")
    st.write(source_label)
    st.caption(source_detail)

    if isinstance(result, CarbonResult):
        st.subheader("Carbon Contract")
        st.write(
            f"Boundary: `{result.boundary}` | Pollutants: `{', '.join(result.pollutants)}` | "
            f"Params version: `{result.params_version}`"
        )
        if result.uncertainty_interval:
            rows = []
            for key, payload in result.uncertainty_interval.items():
                rows.append(
                    {
                        "metric": key,
                        "point": float(payload.get("point", 0.0)),
                        "lower": float(payload.get("lower", 0.0)),
                        "upper": float(payload.get("upper", 0.0)),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if isinstance(result, ForecastResult):
        meaning_note = next((n for n in result.coverage_notes if n.startswith("Meaning:")), None)
        if meaning_note:
            st.subheader("Forecast Meaning")
            st.write(meaning_note.split(":", 1)[1].strip())

    st.subheader("Evidence")
    retrieved_lines = evidence.lines
    computed_lines = _fallback_evidence_from_result(result)
    display_lines = retrieved_lines if show_technical else [_to_analyst_evidence_line(line) for line in retrieved_lines]

    if computed_lines:
        st.markdown("**Computed evidence used for this answer**")
        for line in computed_lines:
            st.markdown(f"- {line}")
    if display_lines:
        st.markdown("**Retrieved supporting evidence**")
        for line in display_lines:
            st.markdown(f"- {line}")
    if not display_lines and not computed_lines:
        st.info("No evidence rows were available for this response.")

    st.subheader("Confidence")
    st.write(_extract_confidence_label(result))
    st.caption("Confidence reflects evidence strength and filter consistency, not ground-truth certainty.")

    _render_chart(result)

    method_steps = _build_method_steps(result)
    if method_steps:
        st.subheader("How This Was Computed")
        for idx, step in enumerate(method_steps, start=1):
            st.markdown(f"{idx}. {step}")

    st.subheader("Port Operations Recommendations")
    for action in _build_port_actions(result):
        st.markdown(f"- {action}")
    st.markdown("**Recommendation Triggers**")
    for trigger in _build_recommendation_triggers(result):
        st.markdown(f"- {trigger}")

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
    if show_technical:
        if isinstance(result, CarbonResult):
            st.markdown("**Carbon technical audit**")
            st.write(
                f"params_version=`{result.params_version}` | "
                f"confidence=`{result.confidence_label}` | "
                f"reason=`{result.confidence_reason}`"
            )
            if result.evidence_ids:
                st.write("Evidence IDs:", ", ".join(result.evidence_ids[:20]))
            if result.segment_ids:
                st.write("Segment IDs:", ", ".join(result.segment_ids[:20]))
            if result.export_csv_path or result.export_json_path:
                st.write(
                    f"Exports: csv=`{result.export_csv_path or 'n/a'}`, "
                    f"json=`{result.export_json_path or 'n/a'}`"
                )
        if evidence.rows:
            trace_df = pd.DataFrame(evidence.rows)
            cols = [c for c in ["vector_id", "chunk_id", "distance", "timestamp", "port", "vessel_type", "mmsi"] if c in trace_df.columns]
            st.dataframe(trace_df[cols], width="stretch", hide_index=True)
        else:
            st.info("No vector rows available for this query. Evidence above may come from deterministic KPI computation.")
        with st.expander("Raw retrieval trace JSON", expanded=False):
            st.json(trace)
    else:
        st.caption("Enable `Technical audit mode` from the sidebar to view vector IDs, chunk IDs, and raw trace JSON.")


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
        show_technical = st.toggle(
            "Technical audit mode",
            value=False,
            help="Show vector IDs, chunk IDs, and raw retrieval trace JSON.",
        )
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
        carbon_cfg = config.get("carbon", {})
        carbon_engine = _init_carbon_engine(
            processed_dir=str(default_processed_dir),
            factor_registry_path=str(carbon_cfg.get("factor_registry_path", "config/carbon_factors.v1.json")),
            monte_carlo_draws=int(carbon_cfg.get("monte_carlo_draws", 500)),
        )
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
        if getattr(carbon_engine, "available", False):
            st.caption(f"Carbon layer active (params: {carbon_engine.params_version.get('version', 'unknown')}).")
        else:
            st.warning("Carbon layer artifacts not found. Build with `python -m src.carbon.build --processed_dir data/processed`.")

    st.session_state["retriever_reason"] = retriever_reason
    events_path = configured_processed_dir / "events.parquet"
    if not events_path.exists():
        events_path = default_processed_dir / "events.parquet"

    default_from_date = pd.Timestamp.now().floor("D") - pd.Timedelta(days=30)
    default_to_date = pd.Timestamp.now().floor("D")
    if not kpi_engine.arrivals_daily.empty and "date" in kpi_engine.arrivals_daily.columns:
        date_series = pd.to_datetime(kpi_engine.arrivals_daily["date"], errors="coerce", utc=True).dropna()
        if not date_series.empty:
            default_from_date = date_series.min().floor("D")
            default_to_date = date_series.max().floor("D")

    st.subheader("Sample Queries")
    if "ask_question" not in st.session_state:
        st.session_state["ask_question"] = SAMPLE_QUERIES_BY_CATEGORY["Traffic Monitoring"][0]

    categories = list(SAMPLE_QUERIES_BY_CATEGORY.keys())
    sample_category = st.selectbox("Query category", options=categories, index=0)
    if sample_category == "Carbon & Emissions":
        st.caption("Carbon layer query tip: include boundary keywords (`TTW`/`WTW`) and pollutants (`CO2`, `NOx`, `SOx`, `PM`).")
    selected = st.selectbox("Try a sample query", options=SAMPLE_QUERIES_BY_CATEGORY[sample_category], index=0)
    if st.button("Load sample query"):
        st.session_state["ask_question"] = selected

    st.subheader("Ask")
    st.text_area("Question", key="ask_question", height=90)
    st.caption(
        "Scope note: congestion is a port-level proxy from AIS/port-call history, not berth-level ground truth."
    )

    with st.expander("Optional filters", expanded=False):
        ui_port = st.text_input("Port / LOCODE / name", value="", help="Examples: SEGOT, Gothenburg, Port of Gothenburg")
        use_date_range = st.checkbox(
            "Apply date range filter",
            value=False,
            help="Use calendar inputs to avoid date formatting errors.",
        )
        if use_date_range:
            date_col_1, date_col_2 = st.columns(2)
            ui_date_from_obj = date_col_1.date_input(
                "From date",
                value=default_from_date.date(),
                format="YYYY-MM-DD",
            )
            ui_date_to_obj = date_col_2.date_input(
                "To date",
                value=default_to_date.date(),
                format="YYYY-MM-DD",
            )
            ui_date_from = pd.Timestamp(ui_date_from_obj).strftime("%Y-%m-%d")
            ui_date_to = pd.Timestamp(ui_date_to_obj).strftime("%Y-%m-%d")
        else:
            ui_date_from = ""
            ui_date_to = ""
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
        carbon=carbon_engine,
        retriever=retriever,
        top_k_evidence=top_k_evidence,
        user_filters=user_filters,
        events_path=events_path if events_path.exists() else None,
    )

    _render_compact_result(result=result, evidence=evidence, show_technical=show_technical)


if __name__ == "__main__":
    main()
