#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
STREAMLIT_LOG="${STREAMLIT_LOG:-/tmp/eagle-eye-streamlit.log}"
TUNNEL_LOG="${TUNNEL_LOG:-/tmp/eagle-eye-lt.log}"
PUBLIC_URL="${PUBLIC_URL:-https://eagle-eye.loca.lt}"
LOCAL_HEALTH_URL="${LOCAL_HEALTH_URL:-http://127.0.0.1:8501/_stcore/health}"
PUBLIC_HEALTH_URL="${PUBLIC_HEALTH_URL:-${PUBLIC_URL}/_stcore/health}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-20}"

cd "${ROOT_DIR}"

start_streamlit() {
  if ! lsof -iTCP:8501 -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    pkill -f "streamlit run app/streamlit_app.py" >/dev/null 2>&1 || true
    nohup env PYTHONPATH=. ./.venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 >"${STREAMLIT_LOG}" 2>&1 &
    sleep 3
  fi
}

start_tunnel() {
  if ! pgrep -f "lt --port 8501 --subdomain eagle-eye" >/dev/null 2>&1; then
    pkill -f "lt --port 8501 --subdomain eagle-eye" >/dev/null 2>&1 || true
    nohup lt --port 8501 --subdomain eagle-eye >"${TUNNEL_LOG}" 2>&1 &
    sleep 3
  fi
}

health_code() {
  local url="$1"
  curl -s -o /dev/null -w "%{http_code}" "${url}" || true
}

echo "[watchdog] starting eagle-eye public watchdog"
echo "[watchdog] public url: ${PUBLIC_URL}"

while true; do
  start_streamlit
  start_tunnel

  local_code="$(health_code "${LOCAL_HEALTH_URL}")"
  public_code="$(health_code "${PUBLIC_HEALTH_URL}")"

  if [[ "${local_code}" != "200" ]]; then
    echo "[watchdog] local health=${local_code} -> restarting streamlit"
    pkill -f "streamlit run app/streamlit_app.py" >/dev/null 2>&1 || true
    sleep 1
    start_streamlit
  fi

  if [[ "${public_code}" != "200" ]]; then
    echo "[watchdog] public health=${public_code} -> restarting localtunnel"
    pkill -f "lt --port 8501 --subdomain eagle-eye" >/dev/null 2>&1 || true
    sleep 1
    start_tunnel
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done

