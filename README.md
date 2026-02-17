# Portathon Congestion Analytics + Forecast + RAG Evidence

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

Tabs:
- **Ask**: natural-language routing (A–G taxonomy)
- **Forecast**: explicit port/day/horizon forecast controls
- **Evaluate**: model metrics + forecast backtest

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
- If Ask/Forecast has no output, run `python -m src.kpi.build_kpis ...` first.
