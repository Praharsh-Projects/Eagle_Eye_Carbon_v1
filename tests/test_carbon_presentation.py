import unittest

import pandas as pd

from src.carbon.presentation import (
    build_emissions_findings,
    build_reduction_suggestions,
    classify_level,
    compute_emissions_metrics,
    derive_threshold_bands,
    extract_chart_findings,
    format_tco2e,
    safe_percent_delta,
    sanitize_threshold_percentiles,
    scale_tco2e,
)


class CarbonPresentationTests(unittest.TestCase):
    def test_autoscale_tco2e_units(self) -> None:
        self.assertEqual(scale_tco2e(12.0).unit, "tCO2e")
        self.assertEqual(scale_tco2e(12_000.0).unit, "ktCO2e")
        self.assertEqual(scale_tco2e(2_000_000.0).unit, "MtCO2e")
        self.assertIn("ktCO2e", format_tco2e(1_500.0))

    def test_threshold_classification(self) -> None:
        bands = derive_threshold_bands([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(classify_level(1.0, bands), "Low")
        self.assertEqual(classify_level(3.5, bands), "Moderate")
        self.assertEqual(classify_level(5.5, bands), "High")
        self.assertEqual(classify_level(9.0, bands), "Very High")

    def test_chart_annotation_generation(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=6, freq="D", tz="UTC"),
                "ttw_co2e_t": [10.0, 20.0, 40.0, 15.0, 18.0, 12.0],
            }
        ).set_index("date")
        findings = extract_chart_findings(df, max_findings=5)
        finding_text = " ".join(f.finding for f in findings)
        self.assertIn("Highest emissions", finding_text)
        self.assertIn("Largest drop", finding_text)

    def test_findings_generation(self) -> None:
        findings = build_emissions_findings(
            current_tco2e=50.0,
            level="Very High",
            change_vs_median_pct=42.0,
            source_label="Computed from AIS + port-call segmentation",
            ci_width_rel=0.2,
            chart_findings=[],
        )
        self.assertTrue(any("very high" in item["text"].lower() for item in findings))
        self.assertTrue(any("42.0%" in item["text"] for item in findings))

    def test_suggestions_generation(self) -> None:
        actions = build_reduction_suggestions(
            level="Very High",
            change_vs_median_pct=30.0,
            ci_width_rel=0.5,
            source_label="Computed with fallback defaults",
        )
        self.assertGreaterEqual(len(actions), 3)
        joined = " ".join(actions).lower()
        self.assertIn("staggered arrival", joined)
        self.assertIn("fallback", joined)

    def test_threshold_percentile_sanitization(self) -> None:
        self.assertEqual(
            sanitize_threshold_percentiles([0.2, 0.5, 0.8]),
            (0.2, 0.5, 0.8),
        )
        self.assertEqual(
            sanitize_threshold_percentiles([0.5, 0.2, 0.8]),
            (0.25, 0.50, 0.75),
        )

    def test_safe_percent_delta_guardrail(self) -> None:
        self.assertIsNone(safe_percent_delta(current_value=10.0, baseline_value=0.2, min_denominator=1.0))
        self.assertAlmostEqual(
            safe_percent_delta(current_value=12.0, baseline_value=10.0, min_denominator=1.0),
            20.0,
            places=6,
        )

    def test_intensity_uses_unique_vessel_calls(self) -> None:
        df = pd.DataFrame(
            {
                "call_id": ["A", "A", "B"],
                "ttw_co2e_t": [1.0, 2.0, 3.0],
                "date": ["2022-01-01", "2022-01-01", "2022-01-02"],
                "duration_h": [1.0, 1.0, 2.0],
            }
        )
        metrics = compute_emissions_metrics(df, boundary="TTW")
        self.assertAlmostEqual(float(metrics["total_tco2e"]), 6.0, places=6)
        # 6 tCO2e == 6000 kgCO2e over 2 unique calls => 3000 kgCO2e/call
        self.assertAlmostEqual(float(metrics["intensity_kg_per_call"]), 3000.0, places=6)


if __name__ == "__main__":
    unittest.main()
