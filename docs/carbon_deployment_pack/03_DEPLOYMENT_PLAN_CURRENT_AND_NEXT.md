# Eagle Eye Deployment Plan (Current State + Next Steps)

## 1) Current deployment modes and parity expectations

### Mode A: Local native Streamlit (most stable for development)
- Command path: `run_streamlit.sh` or direct `.venv/bin/streamlit run ...`
- Uses full local assets directly:
  - `data/processed/*`
  - `data/chroma/*`
- Best for debugging and functional verification.

### Mode B: Docker + free tunnel (recommended free public demo)
- Command path: `run_free_public_app.sh`
- Starts container on `127.0.0.1:8501` then opens tunnel (`cloudflared` default, `ngrok` optional).
- Full parity if local full datasets and chroma store are mounted and Docker daemon is healthy.
- Public URL is temporary unless named tunnel/domain is configured.

### Mode C: Hosted cloud path (limited on free tiers)
- Free hosted full parity is not realistic with full local vector store size (`data/chroma` ~4.8GB).
- Free hosted modes are typically partial-parity (demo bundles, reduced retrieval depth, slower startup).

---

## 2) What has already been done
1. Deployment scripts exist:
   - `/Users/praharshchintu/Documents/New project/run_streamlit.sh`
   - `/Users/praharshchintu/Documents/New project/run_free_public_app.sh`
2. Docker image/runtime path exists (`Dockerfile`, `Dockerfile.api`).
3. Free public tunnel integration exists (`cloudflared`/`ngrok` support in launcher).
4. Cloud bootstrap utilities exist for processed/events/chroma bundles.
5. Remote Chroma runtime mode exists via env vars (`VECTOR_DB_MODE=remote`, `CHROMA_*`).

---

## 3) Exact runbooks (copy-paste)

## 3.1 Clean stop (always run first when things look stuck)
```bash
cd "/Users/praharshchintu/Documents/New project"
pkill -f "streamlit run app/streamlit_app.py" || true
pkill -f "cloudflared tunnel" || true
pkill -f "ngrok http" || true
docker rm -f eagle-eye-streamlit 2>/dev/null || true
```

## 3.2 Port 8501 conflict recovery
```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
# kill PID(s) from output
kill -9 <PID>
# verify free
lsof -nP -iTCP:8501 -sTCP:LISTEN || true
```

## 3.3 Docker daemon recovery (common failure)
Symptoms:
- `Cannot connect to the Docker daemon ...`
- `Docker is installed but the daemon is not reachable`

Fix:
```bash
open -a Docker
# wait until Docker Desktop says "Engine running"
docker info
```
Expected:
- `docker info` returns without daemon errors.

## 3.4 Local-only run (no public link)
```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
set -a; source .env; set +a
PYTHONPATH=. ./.venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
Open:
- `http://127.0.0.1:8501`

## 3.5 Free public run (full local parity while Mac stays on)
```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
set -a; source .env; set +a
./run_free_public_app.sh
```
Expected output:
- `Eagle Eye is live`
- `Local URL: http://127.0.0.1:8501`
- `Public URL: https://<random>.trycloudflare.com` (or ngrok domain)

## 3.6 Health checks (local + public)
```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -I "<PUBLIC_URL>"
```
Expected:
- local health returns quickly
- public URL returns HTTP 200 after tunnel warm-up

---

## 4) Public URL reality (`eagle-eye...` requirement)

### Free random URL
- trycloudflare gives random URL each run.
- cannot guarantee branded hostname (`eagle-eye...`) for free anonymous tunnel.

### Stable branded URL options
To get `eagle-eye...` style stable host, you need one of:
1. Cloudflare named tunnel with your domain DNS:
   - set `CLOUDFLARE_TUNNEL_TOKEN`
   - set `CLOUDFLARE_TUNNEL_HOSTNAME=eagle-eye.<your-domain>`
2. Ngrok reserved domain:
   - set `NGROK_DOMAIN=eagle-eye.<your-ngrok-domain>`

Without one of these, URL will remain random.

---

## 5) Troubleshooting table for recurring issues

| Symptom | Root cause | Fix |
|---|---|---|
| `Port 8501 is not available` | previous Streamlit still running | clean stop + kill PID on 8501 |
| `Docker daemon not reachable` | Docker Desktop engine not running | `open -a Docker`, wait, run `docker info` |
| Tunnel page shows `503 Tunnel Unavailable` | tunnel process died or target app down | restart launcher; verify local health first |
| Public page loads forever (skeleton) | local app unhealthy, heavy cold start, or tunnel not yet forwarding | verify `/_stcore/health`, check logs, wait 15–60 sec, retry |
| Retrieval provenance says unavailable | retriever init failed (missing key/vector backend issue) | verify `OPENAI_API_KEY`, vector settings, and chroma availability |
| Carbon query shows `NOT_COMPUTABLE` | scope has no deterministic carbon rows | verify port/date/call_id and carbon tables coverage |

---

## 6) Operational reproducibility checklist

Before any demo:
1. `source .venv/bin/activate`
2. `set -a; source .env; set +a`
3. `test -f data/processed/carbon_emissions_daily_port.parquet`
4. `test -f data/chroma/chroma.sqlite3`
5. `docker info` (if using free public launcher)
6. start app (local or free public script)
7. run smoke queries from each category

Smoke checks (minimum):
- one traffic descriptive query
- one forecast query
- one anomaly query
- one carbon deterministic query
- one unsupported query

---

## 7) Deployment hardening roadmap

### Immediate (high priority)
1. Add preflight in launcher to auto-kill stale 8501 listeners (with explicit confirmation log).
2. Add single command for local no-docker mode to avoid daemon dependency for urgent demos.
3. Persist run logs into timestamped files for quick post-failure diagnosis.

### Medium-term reliability
1. Add watchdog script to auto-restart tunnel when health checks fail.
2. Add startup self-test endpoint returning data asset and retriever readiness summary.
3. Add query-level latency and failure counters to lightweight metrics file.

### Full-production option
1. Move to stable host with persistent storage and managed TLS/custom domain.
2. Use remote vector DB (managed Chroma/Qdrant) with strict auth and uptime monitoring.
3. Introduce deployment environments (dev/stage/prod) and CI smoke tests per release.

---

## 8) What can be done next (practical options)

### Option 1: Keep fully free (recommended now)
- Use local full data + `run_free_public_app.sh`.
- Accept that app is live only while your Mac and tunnel are running.

### Option 2: Keep URL stable with minimal ops overhead
- Configure Cloudflare named tunnel or ngrok reserved domain.
- Continue serving from local machine during demos.

### Option 3: Move to stable hosted parity (not fully free)
- Host app + storage + remote vector DB.
- Gains: stable URL, uptime, less demo fragility.
- Cost: paid infra + maintenance.

---

## 9) Strict statement for reviewers
- Current free deployment is a **session-hosted public demo**, not an always-on SaaS service.
- Numeric source-of-truth remains deterministic KPI/carbon computation.
- Retrieval layer provides supporting evidence and provenance, not numeric truth.
