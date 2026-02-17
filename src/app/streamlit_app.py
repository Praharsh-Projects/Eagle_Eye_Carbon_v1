"""Streamlit app for deterministic analytics + forecast with optional RAG evidence."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.forecast.backtest import run_backtest
from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.qa.intent import IntentResult, classify_question, describe_intent, required_data_for_intent
from src.rag.retriever import QueryFilters, RAGRetriever
from src.utils.config import load_config
from src.utils.runtime import import_chromadb
from src.utils.serialization import compact_traffic_evidence


@st.cache_resource
def _init_kpi_engine(processed_dir: str) -> KPIQueryEngine:
    return KPIQueryEngine(processed_dir=processed_dir)


@st.cache_resource
def _init_forecast_engine(processed_dir: str) -> ForecastEngine:
    return ForecastEngine(processed_dir=processed_dir)


@st.cache_resource
def _init_retriever(persist_dir: str, config_path: str) -> RAGRetriever:
    return RAGRetriever(persist_dir=persist_dir, config_path=config_path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_diagnostics(
    config: Dict[str, Any],
    persist_dir: str,
    processed_dir: Path,
    model_dir: Path,
    kpi_engine: Optional[KPIQueryEngine],
) -> None:
    with st.expander("Diagnostics", expanded=True):
        st.write(f"Python: `{sys.executable}` ({sys.version.split()[0]})")
        st.write(f"CWD: `{os.getcwd()}`")
        st.write(f"OPENAI_API_KEY set: `{bool(os.getenv('OPENAI_API_KEY'))}`")
        st.write(f"Configured embedding model: `{config['models']['embedding_model']}`")

        st.write(f"Persist dir exists: `{Path(persist_dir).exists()}`")
        st.write(f"Processed dir exists: `{processed_dir.exists()}`")

        required_kpi = [
            "arrivals_daily.parquet",
            "arrivals_hourly.parquet",
            "dwell_time.parquet",
            "occupancy_hourly.parquet",
            "congestion_daily.parquet",
        ]
        for file_name in required_kpi:
            st.write(f"{file_name}: `{(processed_dir / file_name).exists()}`")

        st.write(
            "Prediction model files: "
            f"destination=`{(model_dir / 'destination_model.pkl').exists()}`, "
            f"eta=`{(model_dir / 'eta_model.pkl').exists()}`, "
            f"anomaly=`{(model_dir / 'anomaly_model.pkl').exists()}`"
        )

        if kpi_engine is not None:
            caps = kpi_engine.capabilities()
            st.write("KPI capabilities:")
            st.json(caps)

        if Path(persist_dir).exists():
            try:
                chromadb = import_chromadb()
                client = chromadb.PersistentClient(path=str(Path(persist_dir)))
                traffic = client.get_or_create_collection(name=config["index"]["traffic_collection"]).count()
                docs = client.get_or_create_collection(name=config["index"]["docs_collection"]).count()
                st.write(f"traffic_events.count() = `{traffic}`")
                st.write(f"docs_chunks.count() = `{docs}`")
            except Exception as exc:
                st.warning(f"Could not inspect Chroma collections: {exc}")


def _render_analytics_result(result: AnalyticsResult) -> None:
    st.subheader("Answer")
    st.write(result.answer)

    if result.coverage_notes:
        st.subheader("Coverage Notes")
        for note in result.coverage_notes:
            st.markdown(f"- {note}")

    if result.caveats:
        st.subheader("Caveats")
        for caveat in result.caveats:
            st.markdown(f"- {caveat}")

    if result.chart is not None and not result.chart.empty:
        st.subheader("Chart")
        chart_df = result.chart.copy()
        if isinstance(chart_df.index, pd.DatetimeIndex):
            chart_df.index = chart_df.index.tz_convert("UTC").tz_localize(None)
        st.line_chart(chart_df)

    if result.table is not None and not result.table.empty:
        st.subheader("Data")
        st.dataframe(result.table)


def _render_forecast_result(result: ForecastResult) -> None:
    st.subheader("Forecast Answer")
    st.write(result.answer)

    if result.coverage_notes:
        st.subheader("Coverage Notes")
        for note in result.coverage_notes:
            st.markdown(f"- {note}")

    if result.caveats:
        st.subheader("Caveats")
        for caveat in result.caveats:
            st.markdown(f"- {caveat}")

    if result.history is not None and not result.history.empty:
        history = result.history.copy()
        history = history.rename(columns={"actual": "history_actual"})
        if "date" in history.columns:
            history["date"] = pd.to_datetime(history["date"], errors="coerce", utc=True).dt.tz_localize(None)
            history = history.set_index("date")

        if result.forecast is not None and not result.forecast.empty:
            forecast = result.forecast.copy()
            forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce", utc=True).dt.tz_localize(None)
            forecast = forecast.set_index("date")
            merged = history.join(forecast[["predicted", "lower", "upper"]], how="outer")
            st.subheader("Forecast Chart")
            st.line_chart(merged)
            st.subheader("Forecast Points")
            st.dataframe(forecast.reset_index())
        else:
            st.subheader("History")
            st.line_chart(history)


def _make_rag_filters(entities: Dict[str, Any]) -> QueryFilters:
    port = entities.get("port")
    locode = None
    destination = None
    port_name = None
    if isinstance(port, str) and port.strip():
        token = port.strip()
        if re.fullmatch(r"[A-Za-z]{2}\s?[A-Za-z]{3}", token):
            locode = token
        else:
            destination = token
            port_name = token

    return QueryFilters(
        mmsi=entities.get("mmsi"),
        imo=entities.get("imo"),
        locode=locode,
        port_name=port_name,
        destination=destination,
        vessel_type=entities.get("vessel_type"),
        date_from=entities.get("date_from"),
        date_to=entities.get("date_to"),
    )


def _retrieve_diagnostic_evidence(
    retriever: Optional[RAGRetriever],
    question: str,
    entities: Dict[str, Any],
    top_k: int,
) -> List[str]:
    if retriever is None:
        return []
    try:
        filters = _make_rag_filters(entities)
        result = retriever.query_traffic(question=question, filters=filters, top_k=top_k)
    except Exception as exc:
        return [f"RAG evidence unavailable: {exc}"]

    lines: List[str] = []
    for item in result.evidence[:top_k]:
        lines.append(f"`{item.id}` | {compact_traffic_evidence(item.metadata, item.text)}")
    return lines


def _handle_ask_question(
    question: str,
    intent_result: IntentResult,
    kpi: KPIQueryEngine,
    forecaster: ForecastEngine,
    retriever: Optional[RAGRetriever],
    top_k_evidence: int,
) -> tuple[AnalyticsResult | ForecastResult, List[str]]:
    entities = intent_result.entities
    port = entities.get("port")
    ports = entities.get("ports") or []
    start = entities.get("date_from")
    end = entities.get("date_to")
    vessel_type = entities.get("vessel_type")
    dow = entities.get("dow")
    window = entities.get("window")
    metric = entities.get("metric", "arrivals_vessels")

    if intent_result.intent == "G":
        return (
            KPIQueryEngine.unsupported(
                "This question needs terminal operations data (berth/crane/TEU/gate), which is not in PRJ912/PRJ896."
            ),
            [],
        )

    if intent_result.intent == "A":
        q = question.lower()
        if "top" in q and "port" in q:
            return (
                kpi.top_ports_by_arrivals(start=start, end=end, vessel_type=vessel_type, dow=dow),
                [],
            )
        if "dwell" in q:
            return (
                kpi.get_avg_dwell_time(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow),
                [],
            )
        if "congestion" in q:
            return (
                kpi.get_congestion(port=port, start=start, end=end, dow=dow, window=window),
                [],
            )
        return (
            kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window),
            [],
        )

    if intent_result.intent == "B":
        if entities.get("dow") and entities.get("dow_compare"):
            return (
                kpi.compare_weekdays(
                    port=port,
                    start=start,
                    end=end,
                    day_a=entities["dow"],
                    day_b=entities["dow_compare"],
                    vessel_type=vessel_type,
                ),
                [],
            )
        if "hour" in question.lower():
            return (
                kpi.get_busiest_hour(port=port, start=start, end=end, vessel_type=vessel_type),
                [],
            )
        return (
            kpi.get_busiest_dow(port=port, start=start, end=end, vessel_type=vessel_type),
            [],
        )

    if intent_result.intent == "C":
        return (
            forecaster.forecast_congestion(
                port=port or "",
                target_dow=dow or "Friday",
                horizon_weeks=int(entities.get("horizon_weeks") or 4),
            ),
            [],
        )

    if intent_result.intent == "D":
        return (
            kpi.compare_ports(
                ports=ports,
                metric=metric,
                start=start,
                end=end,
                vessel_type=vessel_type,
                dow=dow,
            ),
            [],
        )

    if intent_result.intent == "E":
        if not start and not end:
            # If no explicit date, use last date from congestion table.
            if not kpi.congestion.empty:
                latest = pd.to_datetime(kpi.congestion["date"], errors="coerce", utc=True).max()
                if pd.notna(latest):
                    start = end = latest.strftime("%Y-%m-%d")
        target_date = start or end
        analytics = kpi.diagnose_congestion(port=port, target_date=target_date)
        evidence = _retrieve_diagnostic_evidence(retriever, question, entities, top_k=top_k_evidence)
        return analytics, evidence

    if intent_result.intent == "F":
        q = question.lower()
        if any(token in q for token in ("jump", "spoof", "teleport", "impossible")) and retriever is not None:
            filters = _make_rag_filters(entities)
            jumps = retriever.detect_sudden_jumps(filters=filters)
            count = int(jumps.get("count", 0))
            ids = jumps.get("rows", [])[:10]
            result = AnalyticsResult(
                status="ok",
                answer=f"Detected {count} potential sudden AIS coordinate jumps in the filtered range.",
                table=pd.DataFrame({"row_id": ids}) if ids else pd.DataFrame(),
                chart=None,
                coverage_notes=[
                    "Jump rule: coordinate displacement above threshold within 30 minutes.",
                    f"Rows flagged: {count}",
                ],
                caveats=["This is a heuristic anomaly indicator, not proof of spoofing."],
            )
            evidence = _retrieve_diagnostic_evidence(retriever, question, entities, top_k=top_k_evidence)
            return result, evidence

        return (
            kpi.detect_arrival_spikes(port=port, start=start, end=end),
            _retrieve_diagnostic_evidence(retriever, question, entities, top_k=top_k_evidence),
        )

    # Safe fallback.
    return (
        kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window),
        [],
    )


def _render_ask_tab(
    kpi: KPIQueryEngine,
    forecaster: ForecastEngine,
    retriever: Optional[RAGRetriever],
    top_k_evidence: int,
) -> None:
    st.subheader("Ask")
    question = st.text_area(
        "Question",
        value="What will congestion look like next Friday at LUBECK?",
        height=90,
    )

    ask = st.button("Ask", type="primary")
    if not ask:
        return

    intent_result = classify_question(question)
    st.markdown(f"**Intent:** `{intent_result.intent}` ({describe_intent(intent_result.intent)})")
    st.markdown(f"**Reason:** {intent_result.reason}")
    st.markdown(f"**Required data:** `{', '.join(required_data_for_intent(intent_result.intent)) or 'none'}`")
    with st.expander("Extracted entities"):
        st.json(intent_result.entities)

    result, evidence = _handle_ask_question(
        question=question,
        intent_result=intent_result,
        kpi=kpi,
        forecaster=forecaster,
        retriever=retriever,
        top_k_evidence=top_k_evidence,
    )

    if isinstance(result, ForecastResult):
        _render_forecast_result(result)
    else:
        _render_analytics_result(result)

    if evidence:
        st.subheader("Representative Evidence")
        for line in evidence:
            st.markdown(f"- {line}")


def _render_forecast_tab(forecaster: ForecastEngine) -> None:
    st.subheader("Forecast")

    c1, c2, c3 = st.columns(3)
    with c1:
        port = st.text_input("Port / LOCODE / destination", value="LUBECK", key="fc_port")
    with c2:
        target_dow = st.selectbox("Target weekday", options=[
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ], index=4)
    with c3:
        horizon_weeks = st.slider("Horizon weeks", min_value=1, max_value=12, value=4)

    metric = st.radio("Forecast metric", options=["Congestion index", "Arrivals"], horizontal=True)
    run = st.button("Run forecast", type="primary", key="run_forecast")
    if not run:
        return

    if metric == "Congestion index":
        result = forecaster.forecast_congestion(port=port, target_dow=target_dow, horizon_weeks=horizon_weeks)
    else:
        result = forecaster.forecast_arrivals(port=port, horizon_weeks=horizon_weeks)

    _render_forecast_result(result)


def _render_evaluate_tab(model_dir: Path, processed_dir: Path, kpi: KPIQueryEngine) -> None:
    st.subheader("Evaluate")

    dest_metrics = _read_json(model_dir / "destination_metrics.json")
    eta_metrics = _read_json(model_dir / "eta_metrics.json")
    anomaly_metrics = _read_json(model_dir / "anomaly_metrics.json")

    st.markdown("### Prediction Models")
    if dest_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Destination Top-1", f"{dest_metrics.get('top1_accuracy', 0):.3f}")
        c2.metric("Destination Top-5", f"{dest_metrics.get('top5_accuracy', 0):.3f}")
        c3.metric("Destination Classes", str(dest_metrics.get("num_classes", "n/a")))
    if eta_metrics and not eta_metrics.get("skipped"):
        c1, c2, c3 = st.columns(3)
        c1.metric("ETA MAE (min)", f"{eta_metrics.get('mae_minutes', 0):.1f}")
        c2.metric("ETA RMSE (min)", f"{eta_metrics.get('rmse_minutes', 0):.1f}")
        c3.metric("ETA MedAE (min)", f"{eta_metrics.get('median_absolute_error_minutes', 0):.1f}")
    if anomaly_metrics and not anomaly_metrics.get("skipped"):
        st.metric("Anomaly training rows", str(anomaly_metrics.get("rows", "n/a")))

    st.markdown("### Forecast Backtest")
    backtest_path = processed_dir / "forecast_backtest.json"
    payload: Dict[str, Any] = _read_json(backtest_path)

    run_bt = st.button("Run / Refresh Backtest")
    if run_bt:
        with st.spinner("Running forecast backtest..."):
            payload = run_backtest(processed_dir=processed_dir, out_path=backtest_path)

    if payload:
        arrivals = payload.get("arrivals", {})
        congestion = payload.get("congestion", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Arrivals Backtest**")
            if arrivals.get("skipped"):
                st.info(arrivals.get("reason", "skipped"))
            else:
                st.write(f"Ports: {arrivals.get('ports_evaluated')}")
                st.write(f"MAE: {arrivals.get('mae_mean', 0):.3f}")
                st.write(f"MAPE: {arrivals.get('mape_mean', 0):.2f}%")
                st.dataframe(pd.DataFrame(arrivals.get("per_port", [])))
        with c2:
            st.markdown("**Congestion Backtest**")
            if congestion.get("skipped"):
                st.info(congestion.get("reason", "skipped"))
            else:
                st.write(f"Ports: {congestion.get('ports_evaluated')}")
                st.write(f"MAE: {congestion.get('mae_mean', 0):.3f}")
                st.write(f"MAPE: {congestion.get('mape_mean', 0):.2f}%")
                st.dataframe(pd.DataFrame(congestion.get("per_port", [])))
    else:
        st.info("No forecast backtest file found yet. Click 'Run / Refresh Backtest'.")

    st.markdown("### KPI Capability Check")
    st.json(kpi.capabilities())


def main() -> None:
    st.set_page_config(page_title="Portathon Congestion Analytics + Forecast", layout="wide")
    st.title("Portathon Analytics + Forecast Demo")
    st.caption("Deterministic KPI analytics + historical-pattern forecasting. RAG is optional evidence, not numeric truth.")

    config_path = "config/config.yaml"
    config = load_config(config_path)
    predict_cfg = config.get("predict", {})

    default_persist = config["paths"]["persist_dir"]
    default_processed_dir = Path(predict_cfg.get("processed_dir", "data/processed"))
    default_model_dir = Path(predict_cfg.get("model_dir", "models"))

    with st.sidebar:
        st.subheader("Runtime")
        processed_dir = Path(st.text_input("Processed dir", value=str(default_processed_dir)))
        persist_dir = st.text_input("Chroma persist dir", value=default_persist)
        model_dir = Path(st.text_input("Model dir", value=str(default_model_dir)))
        top_k_evidence = st.slider("Evidence top K", min_value=1, max_value=10, value=5)

        st.markdown(
            "Build KPIs: `python -m src.kpi.build_kpis ...`\n\n"
            "Run backtest: `python -m src.forecast.backtest`"
        )

    kpi_engine: Optional[KPIQueryEngine] = None
    forecast_engine: Optional[ForecastEngine] = None
    try:
        kpi_engine = _init_kpi_engine(str(processed_dir))
        forecast_engine = _init_forecast_engine(str(processed_dir))
    except Exception as exc:
        st.error(f"Could not initialize KPI/Forecast engines: {exc}")

    retriever: Optional[RAGRetriever] = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            retriever = _init_retriever(persist_dir=persist_dir, config_path=config_path)
        except Exception as exc:
            st.warning(f"RAG evidence disabled: {exc}")

    _render_diagnostics(
        config=config,
        persist_dir=persist_dir,
        processed_dir=processed_dir,
        model_dir=model_dir,
        kpi_engine=kpi_engine,
    )

    if kpi_engine is None or forecast_engine is None:
        st.stop()

    tab_ask, tab_forecast, tab_eval = st.tabs(["Ask", "Forecast", "Evaluate"])

    with tab_ask:
        _render_ask_tab(
            kpi=kpi_engine,
            forecaster=forecast_engine,
            retriever=retriever,
            top_k_evidence=top_k_evidence,
        )

    with tab_forecast:
        _render_forecast_tab(forecaster=forecast_engine)

    with tab_eval:
        _render_evaluate_tab(model_dir=model_dir, processed_dir=processed_dir, kpi=kpi_engine)


if __name__ == "__main__":
    main()
