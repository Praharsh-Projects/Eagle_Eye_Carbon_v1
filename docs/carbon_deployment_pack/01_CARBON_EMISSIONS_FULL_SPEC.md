# Eagle Eye Carbon Emissions Full Spec (TTW + WTW)

## 1) Plain-language overview
Eagle Eye carbon is a deterministic inventory layer built on top of AIS + port-call data.
It computes emissions by first segmenting vessel activity into maritime operation modes, then estimating fuel and pollutants per segment, then aggregating by call/port/day with uncertainty and provenance.

Numeric carbon output is **only treated as valid** when deterministic carbon rows are available for the requested scope. Otherwise the app explicitly reports unavailability/retrieval-only context.

---

## 2) Source-of-truth implementation map
Primary source files:
- `/Users/praharshchintu/Documents/New project/src/carbon/build.py`
- `/Users/praharshchintu/Documents/New project/src/carbon/factors.py`
- `/Users/praharshchintu/Documents/New project/src/carbon/query.py`
- `/Users/praharshchintu/Documents/New project/src/carbon/presentation.py`
- `/Users/praharshchintu/Documents/New project/config/carbon_factors.v1.json`
- `/Users/praharshchintu/Documents/New project/data/processed/carbon_params_version.json`

Current factor/version metadata (from runtime artifact):
- `version`: `eagleeye-carbon-v1.0.0`
- `factor_checksum_sha256`: `850903e508fc3b7137ea2377fc688453181eb653d83e2a89f7fab87ca49bda32`
- `params_hash`: `44c3560aeb08942e84e1e4afc3ce3655ebfc586e9a0bc5a655bcdd325977c736`
- `monte_carlo_draws`: `500`

---

## 3) Data flow and architecture
### 3.1 Inputs
- AIS events: `data/processed/events.parquet`
- Dwell/port-call windows: `data/processed/dwell_time.parquet`
- Carbon factors: `config/carbon_factors.v1.json`

### 3.2 Transform stages (deterministic)
1. Load/clean AIS positions (`event_kind == ais_position`, valid MMSI/timestamp).
2. Attach call windows from dwell table (`arrival_time`, `departure_time`, `call_id`).
3. Assign mode (`transit`, `manoeuvring`, `berth`, `anchorage`) from strict speed/time rules.
4. Build interval durations between AIS points and repair outliers via guarded fallback.
5. Resolve vessel class + factor tables (fuel, SFC, NOx/PM, sulfur assumptions).
6. Compute fuel and emissions per row.
7. Build deterministic segment IDs and aggregate to segment/call/day-port outputs.
8. Add uncertainty intervals + confidence labels.
9. Persist evidence and params/version hash.

### 3.3 Persisted outputs
- `carbon_segments.parquet`:
  full segment-level computed inventory and quality fields.
- `carbon_emissions_segment.parquet`:
  segment pollutant outputs (+ lower/upper bounds).
- `carbon_emissions_daily_port.parquet`:
  date+port aggregates with confidence.
- `carbon_emissions_call.parquet`:
  call-level totals.
- `carbon_evidence.parquet`:
  evidence IDs, input row pointers, confidence/coverage metadata.
- `carbon_params_version.json`:
  factor version/checksum + run metadata + segment rules.

---

## 4) Segmentation rules (exact)
Rule source: `/Users/praharshchintu/Documents/New project/src/carbon/build.py`

Definitions (in order):
- `in_call`: timestamp inside active call window.
- `near_arrival`: `arrival-2h <= ts <= arrival+1h`.
- `near_departure`: `departure-1h <= ts <= departure+2h`.

Mode assignment:
1. `manoeuvring`:
   - `in_call` AND speed in `[0.5, 8.0] kn` AND (`near_arrival` OR `near_departure`)
2. `berth`:
   - `in_call` AND NOT manoeuvring AND speed `<= 0.5 kn`
   - if `berth_timestamp` exists and `ts >= berth_timestamp`, berth is prioritized (outside manoeuvring)
3. `anchorage`:
   - `in_call` AND NOT manoeuvring AND speed `(0.5, 2.0] kn`
4. `transit`:
   - default, and always forced outside call window

Segment boundaries:
- new segment if any of:
  - mode changed,
  - call_id changed,
  - previous interval gap `> 0.5 h`.

---

## 5) Formula set (exact)
All formulas from `/Users/praharshchintu/Documents/New project/src/carbon/build.py` and estimate path in `/Users/praharshchintu/Documents/New project/src/carbon/query.py`.

### 5.1 Load factor and power
For transit/manoeuvring:
- `speed_ratio = speed_kn / ref_speed_kn`
- `LF = clip(speed_ratio^3, 0.2, 1.0)`

Outside transit/manoeuvring:
- `LF = 0.0`

Then:
- `P_main_kw = MCR_kw * LF`
- `P_aux_kw = mode_aux_power_kw(vessel_class, mode)`

### 5.2 Interval fuel
- `fuel_t = ((P_main_kw * SFC_main_g_per_kWh + P_aux_kw * SFC_aux_g_per_kWh) * duration_h) / 1_000_000`
- clipped lower bound: `>= 0`

### 5.3 TTW pollutants
- `CO2_t = fuel_t * CF_CO2_t_per_t_fuel`
- `NOx_kg = fuel_t * NOx_kg_per_t_fuel(engine_family, mode)`
- `PM_kg = fuel_t * PM_kg_per_t_fuel(engine_family, mode)`
- `SOx_kg = fuel_t * sulfur_fraction(mode) * 1000 * sox_multiplier`
  - current `sox_multiplier = 2.0`

### 5.4 TTW/WTW greenhouse totals
- `TTW_CO2e_t = CO2_t`
- `WTT_CO2e_t = fuel_t * WTT_factor_t_per_t_fuel`
- `WTW_CO2e_t = TTW_CO2e_t + WTT_CO2e_t`

### 5.5 Boundary semantics
- `TTW`: uses `ttw_co2e_t` for CO2e reporting.
- `WTW`: uses `wtw_co2e_t` for CO2e reporting.
- `CO2`: direct `co2_t`.
- Non-GHG pollutants remain TTW physical mass outputs (`NOx_kg`, `SOx_kg`, `PM_kg`).

### 5.6 Duration handling safeguards
- `duration_h_raw = next_timestamp - timestamp`
- invalid (`<=0`), too large (`>6h`), or missing durations are replaced by MMSI median of valid intervals (`0<d<=2h`).
- remaining nulls fallback to `1/6h` (10 min).
- final clip: `[1/120h, 6h]`.

---

## 6) Uncertainty and confidence
### 6.1 Segment-level uncertainty
Relative sigma:
- base term: `sqrt(speed_sigma^2 + sfc_sigma^2 + factor_sigma^2)`
- penalties added:
  - interpolation penalty if speed interpolated,
  - gap penalty if AIS gap flag,
  - fallback penalty if fallback factors used.
- clipped to `[0.05, 0.95]`.

Current defaults (from factor registry):
- `speed_rel_sigma = 0.08`
- `sfc_rel_sigma = 0.10`
- `factor_rel_sigma = 0.08`
- `interpolation_penalty_sigma = 0.10`
- `gap_penalty_sigma = 0.08`
- `fallback_penalty_sigma = 0.12`

Per metric deterministic interval:
- `lower = max(0, point * (1 - 1.96*rel_sigma))`
- `upper = max(0, point * (1 + 1.96*rel_sigma))`

### 6.2 Aggregate uncertainty (Monte Carlo)
At aggregate level (daily port / call), 500 draws by default:
- Normal sampling around point total, std = `abs(total)*rel_sigma_weighted`
- Samples clipped at zero
- `2.5%` and `97.5%` quantiles used as lower/upper

### 6.3 Confidence label
From CI width + fallback ratio:
- `high`: `ci_width_rel <= 0.20` and `fallback_usage_ratio <= 0.05`
- `medium`: `ci_width_rel <= 0.40` or `fallback_usage_ratio <= 0.20`
- `low`: otherwise

Meaning:
- Confidence is **evidence/assumption strength**, not ground-truth certainty.

---

## 7) Result-state contract and gating
State constants in `/Users/praharshchintu/Documents/New project/src/carbon/query.py`:
- `COMPUTED`
- `COMPUTED_ZERO`
- `NOT_COMPUTABLE`
- `RETRIEVAL_ONLY`
- `FORECAST_ONLY`
- `UNSUPPORTED`

Semantics:
- `COMPUTED`: deterministic numeric carbon computed for scope.
- `COMPUTED_ZERO`: deterministic computation exists and total is truly zero.
- `NOT_COMPUTABLE`: no deterministic carbon rows/match for requested scope.
- `RETRIEVAL_ONLY`: traffic evidence retrieved but not sufficient for numeric carbon truth.
- `FORECAST_ONLY`: carbon forecast asked but runtime has no deterministic carbon forecast model.
- `UNSUPPORTED`: outside supported carbon scope.

UI gating behavior:
- For non-computed states, totals/intensity/relative level/deltas are `N/A` or unavailable.
- No fake 0.00 totals for unavailable states.
- Deterministic carbon evidence and retrieved traffic evidence are displayed separately.

---

## 8) Sanity diagnostics and inflation guards
Diagnostics fields (query layer):
- `unique_vessel_calls`
- `raw_rows_before_dedup`
- `rows_after_dedup`
- `duplicates_removed_rows`
- `total_duration_hours`
- `median_duration_hours`
- `total_tco2e`
- `mean_tco2e_per_call`
- `median_tco2e_per_call`
- `duplicated_call_ids_detected`
- `warnings[]`
- `sanity_status`

Configurable guardrails:
- `max_call_duration_h = 240`
- `max_call_tco2e = 500`
- `min_baseline_denominator_tco2e = 1.0`

Warnings trigger when:
- implausible call duration,
- per-call emissions above threshold,
- duplicate call IDs detected,
- baseline denominator too small for meaningful % comparison.

Sanity status values:
- `checked`
- `warning`
- `unstable baseline`
- `possible duplication`

---

## 9) Unit standards and labeling
Presentation source: `/Users/praharshchintu/Documents/New project/src/carbon/presentation.py` and app rendering.

Absolute GHG:
- default `tCO2e`
- auto-scale to `ktCO2e` / `MtCO2e`.

Intensity examples in UI:
- `kgCO2e/vessel-call`
- `tCO2e/day`
- `kgCO2e/hour`
- forecast card unit note: `tCO2e/forecast-window`

Operational units:
- speed: `kn`
- distance: `nm`
- time: UTC 24h

Congestion remains dimensionless and labeled as `index`.

---

## 10) What is implemented vs not implemented
### Implemented now
- Deterministic TTW pollutants and WTW CO2e.
- Segment/call/day-port outputs with uncertainty and confidence.
- Result-state gating to separate computable vs non-computable vs retrieval-only.
- Scenario estimate path (`estimate ... 2 hours at 6 knots`) without requiring port/date.
- Export support (CSV/JSON payload) where filesystem allows writes.

### Explicitly not implemented (current v1)
- Physics-grade fuel-consumption model calibrated with vessel-specific technical sheets.
- CH4/N2O explicit combustion modeling in totals (currently trace payload fields are present but set `0.0`).
- Deterministic carbon forecast model (forecast-only queries return explicit unavailable state).
- Berth-level operational truth (crane utilization, gate queues, TEU throughput).

---

## 11) Known data and interpretation limits (must be stated)
1. Carbon output is **inventory estimate** from AIS/port-call segmentation and factor assumptions; not direct stack measurement.
2. Relative level classification (`Low/Moderate/High/Very High`) is **dataset-relative percentile based** unless external threshold pack is added.
3. Missing call linkage can force fallback to deterministic daily proxy inventory (explicitly labeled proxy-based).
4. Recommendations should be treated as decision-support prompts, not automatic control directives.
5. Non-computable state must be interpreted as **unavailable**, never as low emissions.
