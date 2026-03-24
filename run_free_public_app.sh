#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="eagle-eye-streamlit:local"
CONTAINER_NAME="eagle-eye-streamlit"
LOCAL_URL="http://127.0.0.1:8501"
TUNNEL_LOG="$(mktemp -t eagle-eye-tunnel.XXXX.log)"
TUNNEL_PID=""
TUNNEL_URL=""
PREFERRED_TUNNEL="${PREFERRED_TUNNEL:-cloudflared}"
RUN_MODE="docker"
STREAMLIT_PID=""
STREAMLIT_LOG="${ROOT_DIR}/.streamlit_local.log"

cleanup() {
  local exit_code=$?
  if [[ -n "${TUNNEL_PID}" ]] && kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
    kill "${TUNNEL_PID}" >/dev/null 2>&1 || true
    wait "${TUNNEL_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${STREAMLIT_PID}" ]] && kill -0 "${STREAMLIT_PID}" >/dev/null 2>&1; then
    kill "${STREAMLIT_PID}" >/dev/null 2>&1 || true
    wait "${STREAMLIT_PID}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  rm -f "${TUNNEL_LOG}"
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

print_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}

require_cmd() {
  local cmd="$1"
  local help_text="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    printf 'Error: required command `%s` is missing. %s\n' "${cmd}" "${help_text}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'Error: required file is missing: %s\n' "${path}" >&2
    exit 1
  fi
}

load_env_file() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    return
  fi
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${ROOT_DIR}/.env"
    set +a
  fi
}

wait_for_local_streamlit() {
  local attempts=60
  local i
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "${LOCAL_URL}/_stcore/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ "${RUN_MODE}" == "docker" ]]; then
      if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
        printf 'Error: Streamlit container stopped unexpectedly.\n' >&2
        docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
        exit 1
      fi
    else
      if [[ -n "${STREAMLIT_PID}" ]] && ! kill -0 "${STREAMLIT_PID}" >/dev/null 2>&1; then
        printf 'Error: local Streamlit process stopped unexpectedly.\n' >&2
        tail -n 200 "${STREAMLIT_LOG}" >&2 || true
        exit 1
      fi
    fi
    sleep 2
  done
  printf 'Error: Streamlit did not become healthy at %s within timeout.\n' "${LOCAL_URL}" >&2
  if [[ "${RUN_MODE}" == "docker" ]]; then
    docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
  else
    tail -n 200 "${STREAMLIT_LOG}" >&2 || true
  fi
  exit 1
}

wait_for_tunnel_url() {
  local regex="$1"
  local attempts=60
  local i
  for ((i=1; i<=attempts; i++)); do
    if [[ -n "${TUNNEL_PID}" ]] && ! kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
      printf 'Error: tunnel process exited unexpectedly.\n' >&2
      cat "${TUNNEL_LOG}" >&2 || true
      exit 1
    fi
    TUNNEL_URL="$(grep -Eo "${regex}" "${TUNNEL_LOG}" | head -n 1 || true)"
    if [[ -n "${TUNNEL_URL}" ]]; then
      return 0
    fi
    sleep 2
  done
  printf 'Error: could not determine a public tunnel URL.\n' >&2
  cat "${TUNNEL_LOG}" >&2 || true
  exit 1
}

wait_for_public_ready() {
  local url="$1"
  local attempts=90
  local i
  for ((i=1; i<=attempts; i++)); do
    local code
    code="$(curl -o /dev/null -s -w '%{http_code}' "${url}" || true)"
    if [[ "${code}" == "200" ]]; then
      return 0
    fi
    sleep 2
  done
  printf 'Warning: public tunnel URL is not returning HTTP 200 yet: %s\n' "${url}" >&2
  printf 'The tunnel is still running; retry the URL in a browser after 15-30 seconds.\n' >&2
  return 0
}

start_tunnel() {
  if [[ -n "${NGROK_DOMAIN:-}" ]] && command -v ngrok >/dev/null 2>&1; then
    print_step "5/6" "Starting ngrok tunnel with custom domain"
    ngrok http 8501 --domain="${NGROK_DOMAIN}" --log=stdout >"${TUNNEL_LOG}" 2>&1 &
    TUNNEL_PID=$!
    TUNNEL_URL="https://${NGROK_DOMAIN}"
    return 0
  fi

  if [[ "${PREFERRED_TUNNEL}" == "ngrok" ]] && command -v ngrok >/dev/null 2>&1; then
    print_step "5/6" "Starting free ngrok tunnel"
    ngrok http 8501 --log=stdout >"${TUNNEL_LOG}" 2>&1 &
    TUNNEL_PID=$!
    wait_for_tunnel_url 'https://[-a-zA-Z0-9.]+\.ngrok[-a-zA-Z0-9.]*'
    return 0
  fi

  if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]] && command -v cloudflared >/dev/null 2>&1; then
    print_step "5/6" "Starting Cloudflare named tunnel"
    cloudflared tunnel run --token "${CLOUDFLARE_TUNNEL_TOKEN}" >"${TUNNEL_LOG}" 2>&1 &
    TUNNEL_PID=$!
    if [[ -n "${CLOUDFLARE_TUNNEL_HOSTNAME:-}" ]]; then
      TUNNEL_URL="https://${CLOUDFLARE_TUNNEL_HOSTNAME}"
      return 0
    fi
    printf 'Error: CLOUDFLARE_TUNNEL_HOSTNAME is required with CLOUDFLARE_TUNNEL_TOKEN for a stable URL.\n' >&2
    exit 1
  fi

  if command -v cloudflared >/dev/null 2>&1; then
    print_step "5/6" "Starting free Cloudflare tunnel"
    cloudflared tunnel --protocol http2 --url "${LOCAL_URL}" >"${TUNNEL_LOG}" 2>&1 &
    TUNNEL_PID=$!
    wait_for_tunnel_url 'https://[-a-zA-Z0-9.]+\.trycloudflare\.com'
    wait_for_public_ready "${TUNNEL_URL}"
    return 0
  fi

  if command -v ngrok >/dev/null 2>&1; then
    print_step "5/6" "Starting free ngrok tunnel"
    ngrok http 8501 --log=stdout >"${TUNNEL_LOG}" 2>&1 &
    TUNNEL_PID=$!
    wait_for_tunnel_url 'https://[-a-zA-Z0-9.]+\.ngrok[-a-zA-Z0-9.]*'
    wait_for_public_ready "${TUNNEL_URL}"
    return 0
  fi

  printf 'Error: neither `cloudflared` nor `ngrok` is installed.\n' >&2
  printf 'Install `cloudflared` (recommended) or `ngrok` and rerun.\n' >&2
  exit 1
}

print_step "1/6" "Checking prerequisites"
require_cmd curl "curl is needed for health checks."
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  RUN_MODE="docker"
else
  RUN_MODE="local_process"
  printf 'Warning: Docker daemon is not reachable. Falling back to local Streamlit process mode.\n' >&2
  if [[ ! -x "${ROOT_DIR}/.venv/bin/streamlit" ]]; then
    printf 'Error: local fallback needs %s/.venv/bin/streamlit (venv not found).\n' "${ROOT_DIR}" >&2
    exit 1
  fi
fi

load_env_file
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'Error: OPENAI_API_KEY is not set. Export it in your shell or place it in %s/.env.\n' "${ROOT_DIR}" >&2
  exit 1
fi

require_file "${ROOT_DIR}/data/processed/arrivals_daily.parquet"
require_file "${ROOT_DIR}/data/processed/events.parquet"
require_file "${ROOT_DIR}/data/processed/carbon_emissions_daily_port.parquet"
require_file "${ROOT_DIR}/data/processed/carbon_emissions_call.parquet"
require_file "${ROOT_DIR}/data/processed/carbon_evidence.parquet"
require_file "${ROOT_DIR}/data/processed/carbon_params_version.json"
require_file "${ROOT_DIR}/data/chroma/chroma.sqlite3"
require_file "${ROOT_DIR}/data/chroma/traffic_metadata_index.csv"

if lsof -nP -iTCP:8501 -sTCP:LISTEN >/dev/null 2>&1; then
  printf 'Error: port 8501 is already in use. Stop the existing service first.\n' >&2
  exit 1
fi

print_step "2/6" "Building local Streamlit image"
if [[ "${RUN_MODE}" == "docker" ]]; then
  docker build -t "${IMAGE_NAME}" -f "${ROOT_DIR}/Dockerfile" "${ROOT_DIR}" >/dev/null 2>&1

  print_step "3/6" "Starting Streamlit container with full local data"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -p 8501:8501 \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    -e VECTOR_DB_MODE="local" \
    -v "${ROOT_DIR}/app:/app/app:ro" \
    -v "${ROOT_DIR}/src:/app/src:ro" \
    -v "${ROOT_DIR}/config:/app/config:ro" \
    -v "${ROOT_DIR}/data/processed:/app/data/processed:ro" \
    -v "${ROOT_DIR}/data/chroma:/app/data/chroma" \
    "${IMAGE_NAME}" >/dev/null
else
  print_step "2/6" "Starting local Streamlit process with full local data"
  rm -f "${STREAMLIT_LOG}" || true
  PYTHONPATH="${ROOT_DIR}" "${ROOT_DIR}/.venv/bin/streamlit" run "${ROOT_DIR}/app/streamlit_app.py" \
    --server.address 0.0.0.0 --server.port 8501 >"${STREAMLIT_LOG}" 2>&1 &
  STREAMLIT_PID=$!
  print_step "3/6" "Local Streamlit PID: ${STREAMLIT_PID}"
fi

print_step "4/6" "Waiting for Streamlit to become healthy"
wait_for_local_streamlit

start_tunnel

print_step "6/6" "Eagle Eye is live"
printf 'OPENAI_API_KEY: loaded\n'
printf 'Processed data: %s\n' "${ROOT_DIR}/data/processed"
printf 'Vector store: %s\n' "${ROOT_DIR}/data/chroma"
printf 'Local URL: %s\n' "${LOCAL_URL}"
printf 'Public URL: %s\n' "${TUNNEL_URL}"
if [[ "${TUNNEL_URL}" == *"trycloudflare.com"* ]]; then
  printf 'Note: trycloudflare.com URLs are random per run. Set NGROK_DOMAIN or CLOUDFLARE_TUNNEL_TOKEN+CLOUDFLARE_TUNNEL_HOSTNAME for a stable URL.\n'
fi
if [[ "${RUN_MODE}" == "docker" ]]; then
  printf 'Reminder: this stays live only while this Mac, Docker, and the tunnel process remain running.\n'
else
  printf 'Reminder: this stays live only while this Mac, local Streamlit process, and the tunnel process remain running.\n'
fi
printf 'Press Ctrl+C to stop the public app and clean up.\n\n'

while true; do
  if [[ "${RUN_MODE}" == "docker" ]]; then
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
      printf 'Error: Streamlit container stopped unexpectedly.\n' >&2
      docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
      exit 1
    fi
  else
    if [[ -n "${STREAMLIT_PID}" ]] && ! kill -0 "${STREAMLIT_PID}" >/dev/null 2>&1; then
      printf 'Error: local Streamlit process stopped unexpectedly.\n' >&2
      tail -n 200 "${STREAMLIT_LOG}" >&2 || true
      exit 1
    fi
  fi
  if [[ -n "${TUNNEL_PID}" ]] && ! kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
    printf 'Error: tunnel process stopped unexpectedly.\n' >&2
    cat "${TUNNEL_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done
