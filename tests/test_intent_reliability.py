import unittest
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.qa.intent import classify_question
from src.app.streamlit_app import SAMPLE_QUERIES_BY_CATEGORY, _resolve_scope_with_aggressive_port_fallback


@dataclass
class _DummyKPI:
    port_catalog: pd.DataFrame


class IntentReliabilityTests(unittest.TestCase):
    def test_false_port_tokens_are_not_selected(self) -> None:
        q = "Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28."
        parsed = classify_question(q)
        ports = [str(p).upper() for p in parsed.entities.get("ports", [])]
        self.assertIn("LVVNT", ports)
        self.assertNotIn("DAILY", ports)

    def test_trend_query_does_not_parse_trend_as_port(self) -> None:
        q = "Show monthly WTW CO2e trend for SETRG in 2022."
        parsed = classify_question(q)
        ports = [str(p).upper() for p in parsed.entities.get("ports", [])]
        self.assertIn("SETRG", ports)
        self.assertNotIn("TREND", ports)

    def test_call_id_parsing_strips_leading_separator(self) -> None:
        q = "What are call-level emissions for MMSI 123456789 and call_id_123456789_2022-03-01T10-00-00_SEGOT?"
        parsed = classify_question(q)
        self.assertEqual(
            parsed.entities.get("call_id"),
            "123456789_2022-03-01T10-00-00_SEGOT",
        )

    def test_unsupported_variants_are_classified_as_unsupported(self) -> None:
        self.assertEqual(
            classify_question("What is truck turn-time at the gate for SEGOT today?").intent,
            "G",
        )
        self.assertEqual(
            classify_question("Give exact berth-level queue length for vessel arrivals at GDANSK.").intent,
            "G",
        )

    def test_aggressive_port_scope_fallback_prefers_valid_candidate(self) -> None:
        kpi = _DummyKPI(
            port_catalog=pd.DataFrame(
                [
                    {
                        "port_key": "LVVNT",
                        "locode_norm": "LVVNT",
                        "port_label": "Ventspils",
                        "port_name_norm": "ventspils",
                        "arrivals_total": 100,
                        "source_kind": "port_call",
                    },
                    {
                        "port_key": "SEGOT",
                        "locode_norm": "SEGOT",
                        "port_label": "Gothenburg",
                        "port_name_norm": "gothenburg",
                        "arrivals_total": 90,
                        "source_kind": "port_call",
                    },
                ]
            )
        )
        question = "Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28."
        entities = {
            "port": "DAILY",
            "ports": ["DAILY", "LVVNT"],
            "date_from": "2022-02-01",
            "date_to": "2022-02-28",
        }
        scope = _resolve_scope_with_aggressive_port_fallback(
            question=question,
            entities=entities,
            user_filters={"port": None, "date_from": None, "date_to": None},
            kpi=kpi,
        )
        self.assertEqual(scope.get("port"), "LVVNT")
        self.assertTrue(scope.get("correction_applied"))

    def test_sample_query_status_and_carbon_call_pair_integrity(self) -> None:
        statuses = {"A", "B", "C", "D", "E", "F", "G", "H"}
        for category, queries in SAMPLE_QUERIES_BY_CATEGORY.items():
            for q in queries:
                parsed = classify_question(q)
                self.assertIn(parsed.intent, statuses, msg=f"unexpected intent for sample: {category} :: {q}")

        call_level = [q for q in SAMPLE_QUERIES_BY_CATEGORY.get("Carbon & Emissions", []) if "call-level emissions" in q.lower()]
        self.assertTrue(call_level, msg="expected at least one call-level carbon sample query")
        sample = call_level[0]
        parsed = classify_question(sample)
        mmsi = str(parsed.entities.get("mmsi") or "").strip()
        call_id = str(parsed.entities.get("call_id") or "").strip()
        self.assertTrue(mmsi and call_id, msg="call-level sample query must include parseable mmsi and call_id")

        call_table = Path("data/processed/carbon_emissions_call.parquet")
        if call_table.exists():
            df = pd.read_parquet(call_table)
            if not df.empty:
                df["mmsi"] = df["mmsi"].fillna("").astype(str)
                df["call_id"] = df["call_id"].fillna("").astype(str)
                match = df[(df["mmsi"] == mmsi) & (df["call_id"] == call_id)]
                self.assertFalse(match.empty, msg="call-level sample query points to missing data in current dataset")


if __name__ == "__main__":
    unittest.main()
