# Conference Report Guide (Update Path, Not Full Rewrite)

This guide is for your **existing app/workflow** with added retrieval provenance. It tells you exactly what to capture and where to place it in the report.

## 1) Run Setup

```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
./run_demo_pipeline.sh
./run_streamlit.sh
open "http://localhost:8501"
```

Important:
- Keep `OPENAI_API_KEY` set before launching if you want vector retrieval rows (`vector_id`, `chunk_id`, distance).
- If API key is missing, the app will still work, but provenance status will show retrieval disabled.

## 2) Screenshot Naming Convention

Store all screenshots in `report_assets/` with these names:

1. `fig_01_problem_scope.png`
2. `fig_02_data_to_answer_pipeline.png`
3. `fig_03_query_with_filters.png`
4. `fig_04_answer_evidence_confidence.png`
5. `fig_05_chart_and_computation_steps.png`
6. `fig_06_retrieval_provenance_table.png`
7. `fig_07_raw_retrieval_trace_json.png`
8. `fig_08_future_prediction_with_interval.png`
9. `fig_09_anomaly_query_output.png`
10. `fig_10_port_actions_recommendations.png`

## 3) Exact Query Set to Capture

Run these in app, in this order:

1. `How many vessel arrivals were recorded at SEGOT in March 2022?`
2. `Which weekday is usually busiest at LVVNT?`
3. `Show suspicious AIS jumps for MMSI 212575000 on 2021-01-01.`
4. `What will congestion be at LVVNT on Friday, February 20, 2026?`
5. `Compare expected congestion next Friday between LVVNT and SEGOT.`

For each query, capture:
- Answer
- Evidence
- Confidence
- Chart
- How This Was Computed
- Port Operations Recommendations
- Retrieval Provenance

## 4) Where to Put Each Figure in the Paper

### Section 1: Introduction
- Put `fig_01_problem_scope.png`
- Caption: "Problem framing: evidence-grounded maritime decision support from AIS + port-call data."

### Section 3: Methodology (System Architecture)
- Put `fig_02_data_to_answer_pipeline.png`
- Caption: "Pipeline from cleaned data to retrieval, analytics/forecasting, and explainable output."

### Section 4: Query Interface and Filtering
- Put `fig_03_query_with_filters.png`
- Caption: "Question entry and operational filters (port/date/vessel/anomaly)."

### Section 5: Results (Core Output)
- Put `fig_04_answer_evidence_confidence.png`
- Caption: "Model output with direct evidence and confidence label."
- Put `fig_05_chart_and_computation_steps.png`
- Caption: "Visual output and deterministic computation steps for reproducibility."

### Section 5: Results (Traceability)
- Put `fig_06_retrieval_provenance_table.png`
- Caption: "Retrieval provenance: vector IDs, chunk IDs, distance, and metadata."
- Put `fig_07_raw_retrieval_trace_json.png`
- Caption: "Raw retrieval trace showing filter, collection, latency, and retrieval status."

### Section 5: Results (Prediction Use-Case)
- Put `fig_08_future_prediction_with_interval.png`
- Caption: "Future congestion proxy prediction with uncertainty interval."

### Section 5: Results (Risk/Anomaly Use-Case)
- Put `fig_09_anomaly_query_output.png`
- Caption: "Suspicious movement detection result with supporting evidence."

### Section 6: Operational Impact for Ports
- Put `fig_10_port_actions_recommendations.png`
- Caption: "Actionable recommendations generated from analytics/forecast output."

## 5) Results Text You Should Write (Template)

Use this structure in your Results section:

1. Query success: state that the system answers descriptive, anomaly, and forecast questions.
2. Evidence grounding: explain that each answer includes retrieved evidence lines and provenance table.
3. Traceability: explicitly mention `vector_id`, `chunk_id`, distance, active filter, and retrieval status.
4. Operational value: explain staffing/berth planning support and risk triage support.
5. Limits: congestion is proxy-based (not berth/crane-level truth).

## 6) Port Benefit Claims You Can Defend

Keep claims limited to what data supports:

- Better situational awareness from filtered AIS + port-call evidence.
- Faster planning support via daily/weekday traffic pattern summaries.
- Early warning support using anomaly spikes/jump detection.
- Transparent auditability using retrieval provenance fields.

Do not claim:
- Berth crane productivity optimization
- TEU throughput prediction
- Gate queue optimization
(unless you have those datasets)

## 7) Minimal Additional Changes Needed in Report

Add a short subsection titled: **"Evidence Traceability and Auditability"** with:
- One paragraph describing provenance fields.
- `fig_06` and `fig_07`.
- One sentence on reproducibility: same query + same filters + same index snapshot yields same provenance rows.

## 8) Demo Script for Conference (3 minutes)

1. Ask one descriptive question (SEGOT arrivals March 2022).
2. Show answer/evidence/confidence/chart.
3. Scroll to Retrieval Provenance and point to `vector_id`, `chunk_id`, distance.
4. Ask one forecast question (LVVNT future date) and show interval + recommendations.
5. Ask one anomaly question and show operational risk actions.

