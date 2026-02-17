#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=.

TRAFFIC_A="${1:-data/PRJ912.csv}"
TRAFFIC_B="${2:-data/PRJ896.csv}"
PDF_NIS2="${3:-data/CELEX_32022L2555_EN_TXT.pdf}"
PDF_ILO="${4:-data/ILOIMOCodeOfPracticeEnglish.pdf}"
PERSIST_DIR="${PERSIST_DIR:-data/chroma}"

if [[ ! -f "$TRAFFIC_A" ]]; then
  echo "ERROR: Missing traffic CSV: $TRAFFIC_A"
  exit 2
fi

csv_args=(--traffic_csv "$TRAFFIC_A")
if [[ -f "$TRAFFIC_B" ]]; then
  csv_args+=(--traffic_csvs "$TRAFFIC_B")
fi

limit_args=()
if [[ -n "${LIMIT_ROWS:-}" ]]; then
  limit_args+=(--limit_rows "$LIMIT_ROWS")
fi

echo "[1/7] Building processed prediction datasets..."
python -m src.predict.data_prep \
  "${csv_args[@]}" \
  --out_dir data/processed \
  "${limit_args[@]}"

echo "[2/7] Building KPI tables for deterministic analytics..."
python -m src.kpi.build_kpis \
  "${csv_args[@]}" \
  --out_dir data/processed \
  "${limit_args[@]}"

echo "[3/7] Training destination model..."
python -m src.predict.train_destination \
  --training_rows data/processed/training_rows.parquet \
  --model_dir models

echo "[4/7] Training ETA model..."
python -m src.predict.train_eta \
  --training_rows data/processed/training_rows.parquet \
  --model_dir models

echo "[5/7] Training anomaly model..."
python -m src.predict.anomaly \
  --training_rows data/processed/training_rows.parquet \
  --model_dir models

echo "[6/7] Running forecast backtest..."
python -m src.forecast.backtest \
  --processed_dir data/processed \
  --out data/processed/forecast_backtest.json

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[7/7] OPENAI_API_KEY is not set. Skipping RAG index build."
  echo "Set OPENAI_API_KEY and rerun to index traffic/docs into Chroma."
  exit 0
fi

echo "[7/7] Building Chroma index (traffic + docs)..."
pdf_args=()
if [[ -f "$PDF_NIS2" ]]; then
  pdf_args+=("$PDF_NIS2")
fi
if [[ -f "$PDF_ILO" ]]; then
  pdf_args+=("$PDF_ILO")
fi

if [[ ${#pdf_args[@]} -gt 0 ]]; then
  python -m src.index.build_index \
    "${csv_args[@]}" \
    --persist_dir "$PERSIST_DIR" \
    --pdf_paths "${pdf_args[@]}" \
    --doc_urls \
      "https://www.imo.org/en/OurWork/Security/Pages/SOLAS-XI-2%20ISPS-Code.aspx" \
      "https://www.mpa.gov.sg/web/portal/home/port-of-singapore/operations-and-services/maritime-security/isps-code" \
    "${limit_args[@]}"
else
  python -m src.index.build_index \
    "${csv_args[@]}" \
    --persist_dir "$PERSIST_DIR" \
    --doc_urls \
      "https://www.imo.org/en/OurWork/Security/Pages/SOLAS-XI-2%20ISPS-Code.aspx" \
      "https://www.mpa.gov.sg/web/portal/home/port-of-singapore/operations-and-services/maritime-security/isps-code" \
    "${limit_args[@]}"
fi

echo "Done. Run ./run_streamlit.sh"
