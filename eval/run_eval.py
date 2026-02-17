"""Small RAG eval runner for demo readiness checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.rag.generator import AnswerGenerator
from src.rag.retriever import QueryFilters, RAGRetriever
from src.rag.router import RAGRouter


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_questions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_case(
    case: Dict[str, Any],
    retriever: RAGRetriever,
    router: RAGRouter,
    generator: AnswerGenerator,
) -> Dict[str, Any]:
    filters = QueryFilters(**case.get("filters", {}))
    question = case["question"]
    expect = case.get("expect", {})

    retrieval = router.retrieve(question, filters=filters)
    aggregation = None
    if retrieval.mode in {"traffic", "both"}:
        if retriever.is_jump_detection_question(question):
            aggregation = retriever.detect_sudden_jumps(filters=filters)
        else:
            aggregation = retriever.compute_traffic_count(filters=filters, question=question)

    answer = generator.generate(
        question=question,
        filters=filters,
        evidence_items=retrieval.evidence,
        aggregation_result=aggregation,
    )

    checks: Dict[str, bool] = {}
    min_sources = int(expect.get("min_sources", 0))
    checks["min_sources"] = len(answer.evidence_lines) >= min_sources

    expect_refusal = bool(expect.get("expect_refusal", False))
    checks["refusal"] = answer.refused if expect_refusal else True

    checks["filter_respected"] = True
    normalized_filters = filters.normalized()
    if expect.get("respect_filters", False):
        for item in retrieval.evidence:
            if item.source_kind != "traffic":
                continue
            meta = item.metadata
            if normalized_filters.mmsi and str(meta.get("mmsi", "")).strip() != normalized_filters.mmsi:
                checks["filter_respected"] = False
            if normalized_filters.imo and str(meta.get("imo", "")).strip() != normalized_filters.imo:
                checks["filter_respected"] = False
            if normalized_filters.locode and str(meta.get("locode_norm", "")).upper() != normalized_filters.locode:
                checks["filter_respected"] = False
            if normalized_filters.port_name and str(meta.get("port_name_norm", "")).lower() != normalized_filters.port_name:
                checks["filter_respected"] = False
            if normalized_filters.vessel_type and str(meta.get("vessel_type_norm", "")).lower() != normalized_filters.vessel_type:
                checks["filter_respected"] = False
            if normalized_filters.flag and str(meta.get("flag_norm", "")).upper() != normalized_filters.flag:
                checks["filter_respected"] = False
            if normalized_filters.destination and str(meta.get("destination_norm", "")).upper() != normalized_filters.destination:
                checks["filter_respected"] = False
            if normalized_filters.nav_status and str(meta.get("nav_status_norm", "")).lower() != normalized_filters.nav_status:
                checks["filter_respected"] = False
            if normalized_filters.date_from and str(meta.get("timestamp_date", "")) < normalized_filters.date_from:
                checks["filter_respected"] = False
            if normalized_filters.date_to and str(meta.get("timestamp_date", "")) > normalized_filters.date_to:
                checks["filter_respected"] = False
            lat = _as_float(meta.get("latitude"))
            lon = _as_float(meta.get("longitude"))
            if normalized_filters.lat_min is not None and (lat is None or lat < normalized_filters.lat_min):
                checks["filter_respected"] = False
            if normalized_filters.lat_max is not None and (lat is None or lat > normalized_filters.lat_max):
                checks["filter_respected"] = False
            if normalized_filters.lon_min is not None and (lon is None or lon < normalized_filters.lon_min):
                checks["filter_respected"] = False
            if normalized_filters.lon_max is not None and (lon is None or lon > normalized_filters.lon_max):
                checks["filter_respected"] = False

    passed = all(checks.values())
    return {
        "id": case.get("id"),
        "question": question,
        "mode": retrieval.mode,
        "passed": passed,
        "checks": checks,
        "num_evidence": len(answer.evidence_lines),
        "refused": answer.refused,
        "answer_preview": answer.answer[:180],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval checks")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--persist_dir", default="data/chroma")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    retriever = RAGRetriever(persist_dir=args.persist_dir, config_path=args.config)
    router = RAGRouter(retriever)
    generator = AnswerGenerator(config_path=args.config)

    results = [evaluate_case(q, retriever, router, generator) for q in questions]
    pass_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Eval: {pass_count}/{total} passed")
    for row in results:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status}] {row['id']} mode={row['mode']} evidence={row['num_evidence']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {"passed": pass_count, "total": total},
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Detailed results written to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
