# Carbon Evaluation and QA (Current Coverage + Gaps)

## 1) Evaluation objective
This QA spec validates that Eagle Eye carbon outputs are:
1. numerically consistent,
2. clearly state-gated (computed vs unavailable),
3. reproducible from deterministic artifacts,
4. honest about uncertainty and baseline limits.

Primary sources:
- `/Users/praharshchintu/Documents/New project/tests/test_carbon_query_states.py`
- `/Users/praharshchintu/Documents/New project/tests/test_carbon_presentation.py`
- `/Users/praharshchintu/Documents/New project/tests/test_intent_reliability.py`
- `/Users/praharshchintu/Documents/New project/src/carbon/query.py`
- `/Users/praharshchintu/Documents/New project/src/carbon/presentation.py`
- `/Users/praharshchintu/Documents/New project/src/app/streamlit_app.py`

Current test execution snapshot:
- Command: `python -m unittest discover -s tests -p 'test_*.py'`
- Result: `Ran 20 tests ... OK`

---

## 2) End-to-end validation methodology

### 2.1 Single vessel-call trace checklist
For one `(mmsi, call_id)` pair:
1. Confirm call exists in `carbon_emissions_call.parquet`.
2. Pull matching segments from `carbon_emissions_segment.parquet` / `carbon_segments.parquet`.
3. Verify these fields:
   - arrival/departure timestamps,
   - duration_hours sum,
   - vessel_class/proxy class,
   - fuel_t,
   - TTW CO2 (`co2_t`),
   - WTT CO2e (`wtt_co2e_t`),
   - final total (`ttw_co2e_t` or `wtw_co2e_t` by boundary),
   - kg conversion (`t * 1000`).
4. Confirm single-count behavior:
   - `reconciliation_unique_call_count == 1` for query_vessel_call.

### 2.2 Aggregate reconciliation checks
For a port/date scope:
1. Build deterministic scope rows.
2. Verify:
   - displayed total = sum(metric column in deterministic scope),
   - displayed intensity = total_tCO2e*1000 / unique vessel-call count.
3. Ensure dedup occurred before aggregation (`segment_id` dedup path).
4. Verify diagnostics:
   - raw rows,
   - dedup rows,
   - duplicates removed,
   - warnings.

### 2.3 Intensity checks
- `intensity_kg_per_call = total_tco2e * 1000 / unique_call_count`
- `tco2e/day = total_tco2e / unique_days`
- `kgco2e/hour = total_tco2e * 1000 / duration_h_total`

### 2.4 Baseline percentage guardrail checks
- If baseline denominator < `min_baseline_denominator_tco2e` (default 1.0), percent delta must be `N/A`.
- Must not render misleading giant deltas for tiny denominators.

---

## 3) Test inventory and what each test proves

### 3.1 Carbon state and routing tests (`test_carbon_query_states.py`)
- `test_port_query_uses_daily_proxy_when_no_call_linked_segments`
  - proves deterministic daily-proxy fallback works when call-linked rows are absent.
- `test_vessel_call_computed_zero_state`
  - proves real computed-zero behavior is distinguished from no-data.
- `test_forecast_only_state_for_carbon_forecast_prompt`
  - proves forecast-only carbon query yields explicit `FORECAST_ONLY` no-data state.
- `test_estimate_query_routes_to_assumption_engine`
  - proves estimate/scenario path works without port/date.
- `test_call_id_parsing_and_resolution_accepts_call_id_underscore_form`
  - proves robust call_id normalization matching.
- `test_call_id_missing_stays_not_computable`
  - proves missing call-id remains strict no-data.

### 3.2 Presentation and numeric guard tests (`test_carbon_presentation.py`)
- `test_autoscale_tco2e_units`
  - proves tCO2e/ktCO2e/MtCO2e autoscaling.
- `test_threshold_classification`
  - proves Low/Moderate/High/Very High banding.
- `test_chart_annotation_generation`
  - proves chart findings are generated.
- `test_findings_generation`
  - proves deterministic findings text generation.
- `test_suggestions_generation`
  - proves recommendation generation and minimum count.
- `test_threshold_percentile_sanitization`
  - proves threshold percentile guard.
- `test_safe_percent_delta_guardrail`
  - proves denominator guard.
- `test_intensity_uses_unique_vessel_calls`
  - proves intensity denominator uses unique calls.

### 3.3 Intent/parser reliability tests (`test_intent_reliability.py`)
- verifies false tokens (`DAILY`, `TREND`) are not parsed as ports.
- verifies `call_id_...` parsing cleanup.
- verifies unsupported-scope variants classify to `G`.
- verifies aggressive fallback resolves invalid extracted port token to valid catalog candidate.
- verifies sample queries are parseable and call-level sample points to real data.

---

## 4) Query-category validation matrix (minimum 5–10 per category)
Run these through local app and public app, record status and evidence.

Status legend:
- `PASS`: computed/forecast/unsupported behavior matches expected semantics.
- `FAIL`: parser/routing/state/evidence mismatch.

### A) Traffic descriptive (6)
1. How many vessel arrivals were recorded at SEGOT in March 2022?
2. Which weekday is usually busiest at LVVNT?
3. Compare Friday and Monday arrivals at GDANSK in March 2022.
4. Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28.
5. What was the peak arrival day at SEGOT in March 2022?
6. Show cargo-ship arrivals at GDANSK during 2022-03.

Expected:
- deterministic KPI answer, evidence rows, chart, method steps.

### B) Congestion/forecast (6)
1. What will congestion be at LVVNT on Friday, February 20, 2026?
2. Predict congestion for SEGOT next Friday based on historical patterns.
3. Expected congestion at GDANSK on 2026-03-06?
4. Compare expected congestion next Friday between LVVNT and SEGOT.
5. Predict if Monday or Friday will be more congested at LVVNT next week.
6. Will SEGOT likely remain above baseline congestion next Friday?

Expected:
- forecast result with range + confidence + analog notes.

### C) Anomaly (6)
1. Show suspicious AIS jumps for MMSI 246521000 on 2022-03-10.
2. Summarize suspicious AIS jumps for MMSI 212575000 on 2021-01-01.
3. List AIS jump anomalies for MMSI 266232000 between 2021-01-01 and 2021-01-03.
4. Show movement anomalies for MMSI 246650000 in March 2022.
5. How many anomaly events for MMSI 255806245 in 2022-03?
6. Investigate unusual jumps near SEGVX for MMSI 377587000.

Expected:
- deterministic anomaly logic, no hallucinated unsupported metrics.

### D) Carbon deterministic (8)
1. What are TTW emissions at SEGOT in March 2022 for CO2e, NOx, SOx, and PM?
2. Show WTW CO2e emissions at LVVNT between 2022-02-01 and 2022-02-28.
3. Carbon emissions for SEGOT by month in 2022.
4. Report TTW CO2e and NOx at LVVNT for 2022-03 grouped by day.
5. Show WTW CO2e at SEGVX between 2022-03-01 and 2022-03-31.
6. Compare TTW versus WTW CO2e totals at SETRG for March 2022.
7. Show monthly WTW CO2e trend for SETRG in 2022.
8. What are call-level emissions for MMSI 209468000 and call_id 209468000_2021-01-06T10-17-56_SETRG?

Expected:
- state `COMPUTED` or `COMPUTED_ZERO`, metric cards with units, findings, deterministic evidence.

### E) Carbon retrieval-only / no-data (6)
1. Show WTW CO2e emissions at UNKNOWNPORT between 2022-02-01 and 2022-02-28.
2. Carbon emissions for MMSI 111111111 and call_id_111111111_2022-01-01T00-00-00_SEGOT.
3. Forecast carbon emissions at SEGOT next Friday.
4. Show TTW emissions at LVVNT for 2010-01-01 to 2010-01-31.
5. Show call-level emissions for MMSI 999999999 and missing call id.
6. Give carbon evidence IDs for a scope with no deterministic carbon rows.

Expected:
- `NOT_COMPUTABLE` / `RETRIEVAL_ONLY` / `FORECAST_ONLY` explicitly,
- no fake 0.00 totals,
- no misleading percentage deltas,
- deterministic evidence panel separated from retrieved traffic evidence.

### F) Unsupported scope (6)
1. What is crane utilization at berth 3 in SEGOT today?
2. What is gate queue length at Port of Gdansk right now?
3. How many TEU were handled per hour at berth 5 yesterday?
4. What is yard occupancy percentage at terminal block C right now?
5. Show quay crane productivity at LVVNT in March 2022.
6. What is truck turn-time at the gate for SEGOT today?

Expected:
- explicit unsupported refusal (no synthetic KPI/carbon number).

---

## 5) Pass/fail criteria per category

### Global pass criteria
1. No uncaught exceptions.
2. Correct intent branch selected.
3. Evidence/source label consistent with computation path.
4. Units shown for every emissions number.
5. Non-computable states never shown as valid low numeric outcomes.

### Carbon-specific pass criteria
1. `COMPUTED/COMPUTED_ZERO`:
   - numeric totals present,
   - uncertainty interval present (when enabled),
   - relative level available,
   - findings/recommendations tied to computed metrics.
2. `NOT_COMPUTABLE/RETRIEVAL_ONLY/FORECAST_ONLY/UNSUPPORTED`:
   - totals/intensity/deltas/relative level = unavailable,
   - clear state reason,
   - conservative data-quality guidance only.

---

## 6) Failure taxonomy and debug decision tree

### Failure taxonomy
- `PARSER`: wrong entities (port/date/call_id).
- `ROUTER`: wrong branch (carbon vs non-carbon).
- `STATE_GATING`: wrong result state semantics.
- `RECONCILIATION`: totals/intensity mismatch.
- `BASELINE`: denominator too small but deltas still shown.
- `DEPLOYMENT`: runtime assets/tunnel/container unavailable.

### Debug decision tree
1. Did query crash?
   - yes -> inspect stack trace and function signature mismatch.
2. No crash but wrong branch?
   - inspect intent diagnostics in technical mode.
3. Right branch but `NOT_COMPUTABLE` unexpectedly?
   - verify scope rows in `carbon_emissions_daily_port` and `carbon_emissions_call`.
4. `COMPUTED` but values look inflated?
   - check diagnostics: dedup rows, unique calls, duration outliers, per-call threshold warnings.
5. Deltas look absurd?
   - check baseline denominator guard (`min_baseline_denominator_tco2e`).
6. Public link fails but local works?
   - classify as deployment/tunnel issue, not model logic issue.

---

## 7) What is done vs what still needs stronger validation

### Done (validated)
- Unit conversion autoscaling.
- Relative-level classification helper.
- Chart finding generation.
- Percent-delta denominator guard.
- Intensity denominator correctness.
- Carbon result-state contract behavior in unit tests.
- Intent parsing hardening for false port tokens and call-id parsing.

### Needs stronger validation (next)
1. Large-scale numerical reconciliation against external benchmark inventory for selected ports.
2. Stress test on broader noisy natural-language variants.
3. Public URL black-box test matrix persistence (local vs public parity reports saved each run).
4. Additional carbon call-level fixtures covering ambiguous canonical call-id collisions.
5. Long-window performance benchmarks for daily/monthly carbon aggregation under full dataset load.
