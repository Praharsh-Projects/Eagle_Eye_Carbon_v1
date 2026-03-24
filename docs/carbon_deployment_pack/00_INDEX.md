# Eagle Eye Carbon + Deployment Knowledge Pack Index

This pack documents carbon logic, evaluation coverage, and deployment operations using current repository truth.

## Files
1. `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/01_CARBON_EMISSIONS_FULL_SPEC.md`
2. `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/02_CARBON_EVALUATION_AND_QA.md`
3. `/Users/praharshchintu/Documents/New project/docs/carbon_deployment_pack/03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md`

---

## Recommended reading order

### Quick briefing (10-15 min)
1. Read Section 1 + 10 + 11 in `01_CARBON_EMISSIONS_FULL_SPEC.md`
2. Read Section 7 in `02_CARBON_EVALUATION_AND_QA.md`
3. Read Section 1 + 8 + 9 in `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md`

### Technical deep dive (30-60 min)
1. `01_CARBON_EMISSIONS_FULL_SPEC.md` (all sections)
2. `02_CARBON_EVALUATION_AND_QA.md` (all sections)
3. `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` (runbooks + hardening roadmap)

### Operations/demo prep (15-20 min)
1. `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` Section 3 (runbooks)
2. `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` Section 5 (troubleshooting)
3. `02_CARBON_EVALUATION_AND_QA.md` Section 4 (query matrix)

---

## Section-to-question map

### “How is carbon computed?”
- `01_CARBON_EMISSIONS_FULL_SPEC.md` Sections 3, 4, 5

### “Which formulas and assumptions are used?”
- `01_CARBON_EMISSIONS_FULL_SPEC.md` Sections 5, 6

### “How do you handle confidence/uncertainty?”
- `01_CARBON_EMISSIONS_FULL_SPEC.md` Section 6
- `02_CARBON_EVALUATION_AND_QA.md` Sections 2, 3

### “How do you avoid fake low-emission outputs when data is missing?”
- `01_CARBON_EMISSIONS_FULL_SPEC.md` Section 7
- `02_CARBON_EVALUATION_AND_QA.md` Sections 3, 5

### “How was this validated?”
- `02_CARBON_EVALUATION_AND_QA.md` Sections 2, 3, 4, 5

### “How do we run and deploy reliably?”
- `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` Sections 3, 5, 6

### “What is already done vs what is pending?”
- `01_CARBON_EMISSIONS_FULL_SPEC.md` Section 10
- `02_CARBON_EVALUATION_AND_QA.md` Section 7
- `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` Sections 2, 7, 8

---

## If demo breaks: one-page recovery

### Symptom: Port 8501 unavailable
Go to:
- `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` Section 3.1 + 3.2

### Symptom: Docker daemon unreachable
Go to:
- `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` Section 3.3

### Symptom: Tunnel 503 / URL unreachable
Go to:
- `03_DEPLOYMENT_PLAN_CURRENT_AND_NEXT.md` Section 5

### Symptom: Carbon says not computable
Go to:
- `01_CARBON_EMISSIONS_FULL_SPEC.md` Section 7
- `02_CARBON_EVALUATION_AND_QA.md` Sections 4 and 6

### Symptom: reviewer asks why percentages are N/A
Go to:
- `01_CARBON_EMISSIONS_FULL_SPEC.md` Section 8
- `02_CARBON_EVALUATION_AND_QA.md` Section 2.4

---

## Documentation scope note
- This pack is documentation-only and does not modify runtime logic.
- Formulas, thresholds, states, and runbooks are mapped to current code/artifacts in this repository.
