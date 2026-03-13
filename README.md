# Eagle Eye Congestion Analytics + Forecast + RAG Evidence

This project uses:
- `PRJ912.csv` (AIS telemetry)
- `PRJ896.csv` (port calls)
- Optional docs (NIS2 PDF + public ISPS pages)

It now has two layers:
1. Deterministic analytics/forecast (source of truth for counts, congestion, trends)
2. Optional RAG evidence (representative examples, not numeric truth)

## 1) Mac Setup

```bash
cd "/Users/praharshchintu/Documents/New project"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set API key (only needed for RAG index + evidence retrieval):

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## 2) Data Inputs

Put files in `/Users/praharshchintu/Documents/New project/data/`:
- `PRJ912.csv`
- `PRJ896.csv`
- `CELEX_32022L2555_EN_TXT.pdf` (recommended)
- `ILOIMOCodeOfPracticeEnglish.pdf` (optional)

For ISPS, prefer public official pages via URLs, not unofficial full-text PDFs.

## 3) One-Command Pipeline

```bash
./run_demo_pipeline.sh
```

This runs, in order:
1. `src.predict.data_prep`
2. `src.kpi.build_kpis`
3. destination / ETA / anomaly training
4. RAG indexing (if `OPENAI_API_KEY` is set)

## 4) Manual Commands

### Build prediction-ready datasets
```bash
python -m src.predict.data_prep \
  --traffic_csv data/PRJ912.csv \
  --traffic_csvs data/PRJ896.csv \
  --out_dir data/processed
```

### Build KPI tables (required for Ask/Forecast tabs)
```bash
python -m src.kpi.build_kpis \
  --traffic_csv data/PRJ912.csv \
  --traffic_csvs data/PRJ896.csv \
  --out_dir data/processed
```

Outputs include:
- `data/processed/arrivals_daily.parquet`
- `data/processed/arrivals_hourly.parquet`
- `data/processed/dwell_time.parquet`
- `data/processed/occupancy_hourly.parquet`
- `data/processed/congestion_daily.parquet`
- `data/processed/kpi_capabilities.json`

### Train prediction models
```bash
python -m src.predict.train_destination --training_rows data/processed/training_rows.parquet --model_dir models
python -m src.predict.train_eta --training_rows data/processed/training_rows.parquet --model_dir models
python -m src.predict.anomaly --training_rows data/processed/training_rows.parquet --model_dir models
```

### Build RAG index (optional but recommended for evidence)
```bash
python -m src.index.build_index \
  --traffic_csv data/PRJ912.csv \
  --traffic_csvs data/PRJ896.csv \
  --persist_dir data/chroma \
  --pdf_paths data/CELEX_32022L2555_EN_TXT.pdf data/ILOIMOCodeOfPracticeEnglish.pdf \
  --doc_urls https://www.imo.org/en/OurWork/Security/Pages/SOLAS-XI-2%20ISPS-Code.aspx https://www.mpa.gov.sg/web/portal/home/port-of-singapore/operations-and-services/maritime-security/isps-code
```

### Forecast backtest
```bash
python -m src.forecast.backtest --processed_dir data/processed
```

## 5) Run Streamlit

```bash
./run_streamlit.sh
```

UI:
- **Ask-only** interface with integrated analytics + forecasting + retrieval evidence trace.
- Includes answer, evidence, confidence, chart, method steps, and retrieval provenance.

## 5.1) Streamlit Cloud Deploy

Use these exact values in Streamlit Cloud:
- Repository: `Praharsh-Projects/Eagle_Eye`
- Branch: `main`
- Main file path: `app/streamlit_app.py`

For cloud environments where `data/processed` is not present, the app auto-falls back to bundled app runtime KPI data in `demo_data/processed`.
For cloud environments where `data/chroma` is not present, the app auto-falls back to bundled demo vector index in `demo_data/chroma`.

To enable vector retrieval evidence (`vector_id`, `chunk_id`, distance), set Streamlit secret:
- `OPENAI_API_KEY = "..."`.

To run full-scale retrieval on cloud (same behavior as local), connect a remote Chroma service:
- `VECTOR_DB_MODE = "remote"`
- `CHROMA_HOST = "<your-chroma-host>"`
- `CHROMA_PORT = "8000"` (or your service port)
- `CHROMA_SSL = "true"` (for HTTPS services)
- Optional: `CHROMA_TENANT`, `CHROMA_DATABASE`, `CHROMA_AUTH_TOKEN`, `CHROMA_AUTH_HEADER`

To bootstrap full processed runtime data on cloud from a hosted bundle, set:
- `APP_PROCESSED_BUNDLE_URL = "https://.../eagle_eye_processed_bundle.tar.gz"`
- Optional for anomaly/jump detection without retriever:
- `APP_EVENTS_BUNDLE_URL = "https://.../eagle_eye_events_bundle.tar.gz"`
- Optional for local-bundle retrieval fallback on hosts with enough disk:
- `APP_CHROMA_BUNDLE_URL = "https://.../eagle_eye_chroma_bundle.tar.gz"`

Create that bundle locally:
```bash
python -m src.utils.package_cloud_bundle \
  --processed_dir data/processed \
  --out dist/eagle_eye_processed_bundle.tar.gz \
  --events_out dist/eagle_eye_events_bundle.tar.gz \
  --chroma_dir data/chroma \
  --chroma_out dist/eagle_eye_chroma_bundle.tar.gz
```

Index directly to remote service:
```bash
export VECTOR_DB_MODE=remote
export CHROMA_HOST=<your-chroma-host>
export CHROMA_PORT=8000
export CHROMA_SSL=true
python -m src.index.build_index \
  --traffic_csv data/PRJ912.csv \
  --traffic_csvs data/PRJ896.csv \
  --persist_dir data/chroma
```

Cloud parity summary:
- Deterministic analytics/forecast parity: bundled in `demo_data/processed`, or bootstrap via `APP_PROCESSED_BUNDLE_URL`
- Retrieval parity: requires remote Chroma service because local `data/chroma` is too large for Streamlit Cloud
- AIS jump/spoof anomaly parity without retriever: requires `APP_EVENTS_BUNDLE_URL` because those queries need row-level AIS events
- On non-Streamlit hosts with enough disk, `APP_CHROMA_BUNDLE_URL` can bootstrap a local full vector store and avoid remote Chroma entirely

## 6) Congestion Definition (used in code)

Daily congestion proxy per port:
- `arrivals_ratio = arrivals_day / median(arrivals_port)`
- `dwell_ratio = median_dwell_day / median(dwell_port)`
- `congestion_index = arrivals_ratio + dwell_ratio`

If dwell is unavailable, congestion falls back to arrivals-only ratio.

## 7) Supported vs Unsupported

Supported well:
- arrivals volume, busiest day/hour, dwell proxy, congestion proxy, historical-pattern forecasts

Out of scope (clean refusal):
- berth crane utilization
- gate queue length
- TEU throughput
- yard occupancy from terminal ops systems

## 8) Demo Questions

- `How many vessels arrived at LUBECK in March 2022?`
- `Is Friday usually busier than Monday at LVVNT?`
- `What will congestion look like next Friday at LUBECK?`
- `Why was LVVNT congested on 2021-01-01?`
- `Any unusual spikes in arrivals at GDANSK in 2021-02?`

## 9) Troubleshooting

- If Chroma fails with Python 3.14, recreate `.venv` with Python 3.12.
- If RAG evidence is unavailable, ensure `OPENAI_API_KEY` is exported in the same terminal.
- If retrieval is disabled on cloud, check Streamlit secrets for `OPENAI_API_KEY` and `CHROMA_*` variables.
- If cloud is still on partial coverage, verify whether the sidebar shows `demo_data/processed`; if so, either upload the processed bundle or set `APP_PROCESSED_BUNDLE_URL`.
- If Ask has no deterministic output, run `python -m src.kpi.build_kpis ...` first.

## 10) Full Deployment Alternative (Recommended for Local-Parity Hosting)

Streamlit Cloud is not a good target for the full local model because the local Chroma store is several GB.
For full deployment, use the FastAPI service in this repo on a host with disk, such as Render, Railway, Fly.io, or a VPS.

### Run locally

```bash
./run_api.sh
```

API endpoints:
- `GET /health`
- `POST /ask`
- Swagger docs at `http://localhost:8000/docs`

### Docker run

```bash
docker build -t eagle-eye .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="..." \
  -e APP_PROCESSED_BUNDLE_URL="https://.../eagle_eye_processed_bundle.tar.gz" \
  -e APP_EVENTS_BUNDLE_URL="https://.../eagle_eye_events_bundle.tar.gz" \
  -e APP_CHROMA_BUNDLE_URL="https://.../eagle_eye_chroma_bundle.tar.gz" \
  eagle-eye
```

### Recommended production modes

1. `FastAPI + remote Chroma`
- Best when you already operate a Chroma service.
- Set `VECTOR_DB_MODE=remote` plus `CHROMA_*` variables.

2. `FastAPI + local bundle bootstrap`
- Best when you want one deployed service and can attach disk.
- Set:
  - `APP_PROCESSED_BUNDLE_URL`
  - `APP_EVENTS_BUNDLE_URL`
  - `APP_CHROMA_BUNDLE_URL`
- Do not set `VECTOR_DB_MODE=remote`.

This repo also includes `render.yaml` for a Render web-service deployment with an attached disk.

### Example request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What will congestion be at LVVNT on Friday, February 20, 2026?",
    "top_k_evidence": 5,
    "filters": {"port": "LVVNT"}
  }'
```
