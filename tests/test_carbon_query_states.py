import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.carbon.query import (
    CARBON_STATE_COMPUTED_ZERO,
    CARBON_STATE_FORECAST_ONLY,
    CARBON_STATE_NOT_COMPUTABLE,
    CarbonQueryEngine,
)


def _write_minimal_carbon_artifacts(
    out_dir: Path,
    segments: pd.DataFrame,
    calls: pd.DataFrame,
    daily_port: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(out_dir / "carbon_emissions_segment.parquet", index=False)
    calls.to_parquet(out_dir / "carbon_emissions_call.parquet", index=False)
    daily_port.to_parquet(out_dir / "carbon_emissions_daily_port.parquet", index=False)
    evidence.to_parquet(out_dir / "carbon_evidence.parquet", index=False)
    # Presence-only file for availability checks.
    segments.head(1).to_parquet(out_dir / "carbon_segments.parquet", index=False)
    (out_dir / "carbon_params_version.json").write_text(
        json.dumps({"version": "test-v1", "hash": "abc"}),
        encoding="utf-8",
    )


class CarbonQueryStateTests(unittest.TestCase):
    def test_port_query_not_computable_when_no_call_linked_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segments = pd.DataFrame(
                [
                    {
                        "segment_id": "seg-1",
                        "mmsi": "111111111",
                        "call_id": None,
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "timestamp_start": pd.Timestamp("2022-03-05T10:00:00Z"),
                        "timestamp_end": pd.Timestamp("2022-03-05T11:00:00Z"),
                        "duration_h": 1.0,
                        "row_count": 1,
                        "fallback_usage_ratio": 0.0,
                        "ci_width_rel": 0.2,
                        "confidence_reason": "ok",
                        "co2_t": 1.0,
                        "ttw_co2e_t": 1.0,
                        "wtt_co2e_t": 0.2,
                        "wtw_co2e_t": 1.2,
                        "nox_kg": 1.0,
                        "sox_kg": 1.0,
                        "pm_kg": 1.0,
                        "co2_t_lower": 0.8,
                        "ttw_co2e_t_lower": 0.8,
                        "wtt_co2e_t_lower": 0.1,
                        "wtw_co2e_t_lower": 0.9,
                        "nox_kg_lower": 0.8,
                        "sox_kg_lower": 0.8,
                        "pm_kg_lower": 0.8,
                        "co2_t_upper": 1.2,
                        "ttw_co2e_t_upper": 1.2,
                        "wtt_co2e_t_upper": 0.3,
                        "wtw_co2e_t_upper": 1.5,
                        "nox_kg_upper": 1.2,
                        "sox_kg_upper": 1.2,
                        "pm_kg_upper": 1.2,
                    }
                ]
            )
            calls = pd.DataFrame(columns=["call_id", "mmsi", "ttw_co2e_t", "wtw_co2e_t"])
            daily = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2022-03-05T00:00:00Z"),
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "ttw_co2e_t": 1.0,
                        "wtw_co2e_t": 1.2,
                    }
                ]
            )
            evidence = pd.DataFrame(
                [
                    {
                        "evidence_id": "ev-1",
                        "segment_id": "seg-1",
                        "mmsi": "111111111",
                        "call_id": None,
                        "port_key": "SEGOT",
                        "timestamp_start": pd.Timestamp("2022-03-05T10:00:00Z"),
                        "timestamp_end": pd.Timestamp("2022-03-05T11:00:00Z"),
                        "row_count": 1,
                    }
                ]
            )
            _write_minimal_carbon_artifacts(root, segments, calls, daily, evidence)
            engine = CarbonQueryEngine(processed_dir=root, auto_build=False)
            result = engine.query_port_emissions(
                port_id="SEGOT",
                date_from="2022-03-01",
                date_to="2022-03-31",
                boundary="TTW",
                pollutants=["CO2e"],
            )
            self.assertEqual(result.status, "no_data")
            self.assertEqual(result.result_state, CARBON_STATE_NOT_COMPUTABLE)
            self.assertIsNone(result.table)

    def test_vessel_call_computed_zero_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            call_id = "111111111_2022-03-05T10-00-00_SEGOT"
            mmsi = "111111111"
            segments = pd.DataFrame(
                [
                    {
                        "segment_id": "seg-2",
                        "mmsi": mmsi,
                        "call_id": call_id,
                        "mode": "berth",
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "timestamp_start": pd.Timestamp("2022-03-05T10:00:00Z"),
                        "timestamp_end": pd.Timestamp("2022-03-05T11:00:00Z"),
                        "duration_h": 1.0,
                        "row_count": 1,
                        "fallback_usage_ratio": 0.0,
                        "ci_width_rel": 0.1,
                        "confidence_reason": "ok",
                        "vessel_class": "cargo",
                        "co2_t": 0.0,
                        "ttw_co2e_t": 0.0,
                        "wtt_co2e_t": 0.0,
                        "wtw_co2e_t": 0.0,
                        "nox_kg": 0.0,
                        "sox_kg": 0.0,
                        "pm_kg": 0.0,
                        "co2_t_lower": 0.0,
                        "ttw_co2e_t_lower": 0.0,
                        "wtt_co2e_t_lower": 0.0,
                        "wtw_co2e_t_lower": 0.0,
                        "nox_kg_lower": 0.0,
                        "sox_kg_lower": 0.0,
                        "pm_kg_lower": 0.0,
                        "co2_t_upper": 0.0,
                        "ttw_co2e_t_upper": 0.0,
                        "wtt_co2e_t_upper": 0.0,
                        "wtw_co2e_t_upper": 0.0,
                        "nox_kg_upper": 0.0,
                        "sox_kg_upper": 0.0,
                        "pm_kg_upper": 0.0,
                    }
                ]
            )
            calls = pd.DataFrame(
                [
                    {
                        "call_id": call_id,
                        "mmsi": mmsi,
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "ci_width_rel": 0.1,
                        "fallback_usage_ratio": 0.0,
                        "ttw_co2e_t": 0.0,
                        "wtt_co2e_t": 0.0,
                        "wtw_co2e_t": 0.0,
                        "co2_t": 0.0,
                        "nox_kg": 0.0,
                        "sox_kg": 0.0,
                        "pm_kg": 0.0,
                        "ttw_co2e_t_lower": 0.0,
                        "wtt_co2e_t_lower": 0.0,
                        "wtw_co2e_t_lower": 0.0,
                        "co2_t_lower": 0.0,
                        "nox_kg_lower": 0.0,
                        "sox_kg_lower": 0.0,
                        "pm_kg_lower": 0.0,
                        "ttw_co2e_t_upper": 0.0,
                        "wtt_co2e_t_upper": 0.0,
                        "wtw_co2e_t_upper": 0.0,
                        "co2_t_upper": 0.0,
                        "nox_kg_upper": 0.0,
                        "sox_kg_upper": 0.0,
                        "pm_kg_upper": 0.0,
                    }
                ]
            )
            daily = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2022-03-05T00:00:00Z"),
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "ttw_co2e_t": 0.0,
                        "wtw_co2e_t": 0.0,
                    }
                ]
            )
            evidence = pd.DataFrame(
                [
                    {
                        "evidence_id": "ev-2",
                        "segment_id": "seg-2",
                        "mmsi": mmsi,
                        "call_id": call_id,
                        "port_key": "SEGOT",
                        "timestamp_start": pd.Timestamp("2022-03-05T10:00:00Z"),
                        "timestamp_end": pd.Timestamp("2022-03-05T11:00:00Z"),
                        "row_count": 1,
                    }
                ]
            )
            _write_minimal_carbon_artifacts(root, segments, calls, daily, evidence)
            engine = CarbonQueryEngine(processed_dir=root, auto_build=False)
            result = engine.query_vessel_call(mmsi=mmsi, call_id=call_id, boundary="WTW", pollutants=["CO2e"])
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.result_state, CARBON_STATE_COMPUTED_ZERO)
            self.assertIsNotNone(result.diagnostics.get("trace_single_call"))

    def test_forecast_only_state_for_carbon_forecast_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segments = pd.DataFrame(
                [
                    {
                        "segment_id": "seg-3",
                        "mmsi": "111111111",
                        "call_id": "call-x",
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "timestamp_start": pd.Timestamp("2022-03-05T10:00:00Z"),
                        "timestamp_end": pd.Timestamp("2022-03-05T11:00:00Z"),
                        "duration_h": 1.0,
                        "row_count": 1,
                        "fallback_usage_ratio": 0.0,
                        "ci_width_rel": 0.1,
                        "confidence_reason": "ok",
                        "co2_t": 1.0,
                        "ttw_co2e_t": 1.0,
                        "wtt_co2e_t": 0.2,
                        "wtw_co2e_t": 1.2,
                        "nox_kg": 1.0,
                        "sox_kg": 1.0,
                        "pm_kg": 1.0,
                        "co2_t_lower": 0.8,
                        "ttw_co2e_t_lower": 0.8,
                        "wtt_co2e_t_lower": 0.1,
                        "wtw_co2e_t_lower": 0.9,
                        "nox_kg_lower": 0.8,
                        "sox_kg_lower": 0.8,
                        "pm_kg_lower": 0.8,
                        "co2_t_upper": 1.2,
                        "ttw_co2e_t_upper": 1.2,
                        "wtt_co2e_t_upper": 0.3,
                        "wtw_co2e_t_upper": 1.5,
                        "nox_kg_upper": 1.2,
                        "sox_kg_upper": 1.2,
                        "pm_kg_upper": 1.2,
                    }
                ]
            )
            calls = pd.DataFrame(
                [
                    {
                        "call_id": "call-x",
                        "mmsi": "111111111",
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "ci_width_rel": 0.1,
                        "fallback_usage_ratio": 0.0,
                        "ttw_co2e_t": 1.0,
                        "wtt_co2e_t": 0.2,
                        "wtw_co2e_t": 1.2,
                        "co2_t": 1.0,
                        "nox_kg": 1.0,
                        "sox_kg": 1.0,
                        "pm_kg": 1.0,
                        "ttw_co2e_t_lower": 0.8,
                        "wtt_co2e_t_lower": 0.1,
                        "wtw_co2e_t_lower": 0.9,
                        "co2_t_lower": 0.8,
                        "nox_kg_lower": 0.8,
                        "sox_kg_lower": 0.8,
                        "pm_kg_lower": 0.8,
                        "ttw_co2e_t_upper": 1.2,
                        "wtt_co2e_t_upper": 0.3,
                        "wtw_co2e_t_upper": 1.5,
                        "co2_t_upper": 1.2,
                        "nox_kg_upper": 1.2,
                        "sox_kg_upper": 1.2,
                        "pm_kg_upper": 1.2,
                    }
                ]
            )
            daily = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2022-03-05T00:00:00Z"),
                        "port_key": "SEGOT",
                        "port_label": "Gothenburg",
                        "locode_norm": "SEGOT",
                        "ttw_co2e_t": 1.0,
                        "wtw_co2e_t": 1.2,
                    }
                ]
            )
            evidence = pd.DataFrame(
                [
                    {
                        "evidence_id": "ev-3",
                        "segment_id": "seg-3",
                        "mmsi": "111111111",
                        "call_id": "call-x",
                        "port_key": "SEGOT",
                        "timestamp_start": pd.Timestamp("2022-03-05T10:00:00Z"),
                        "timestamp_end": pd.Timestamp("2022-03-05T11:00:00Z"),
                        "row_count": 1,
                    }
                ]
            )
            _write_minimal_carbon_artifacts(root, segments, calls, daily, evidence)
            engine = CarbonQueryEngine(processed_dir=root, auto_build=False)
            result = engine.from_question_entities(
                question="Predict carbon emissions at SEGOT next Friday",
                entities={"boundary": "WTW", "pollutants": ["CO2e"], "port": "SEGOT"},
                user_filters={},
            )
            self.assertEqual(result.status, "no_data")
            self.assertEqual(result.result_state, CARBON_STATE_FORECAST_ONLY)


if __name__ == "__main__":
    unittest.main()
