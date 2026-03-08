"""Forecast congestion/arrival proxies from KPI daily time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.kpi.query import KPIQueryEngine


@dataclass
class ForecastResult:
    status: str
    answer: str
    history: Optional[pd.DataFrame]
    forecast: Optional[pd.DataFrame]
    coverage_notes: List[str]
    caveats: List[str]


class ForecastEngine:
    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)
        self.kpi = KPIQueryEngine(processed_dir=self.processed_dir)

    @staticmethod
    def _prepare_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        series = (
            df[[date_col, value_col]]
            .dropna(subset=[date_col, value_col])
            .groupby(date_col, dropna=False)[value_col]
            .sum()
            .sort_index()
            .astype(float)
        )
        return series

    @staticmethod
    def _one_step_prediction(history: List[float]) -> float:
        if not history:
            return 0.0
        if len(history) == 1:
            return float(max(0.0, history[-1]))
        seasonal = history[-7] if len(history) >= 7 else history[-1]
        window = history[-7:] if len(history) >= 7 else history
        moving_avg = float(np.mean(window))
        pred = 0.7 * float(seasonal) + 0.3 * moving_avg
        return float(max(0.0, pred))

    @classmethod
    def _forecast_with_intervals(cls, series: pd.Series, horizon_days: int) -> pd.DataFrame:
        if series.empty:
            return pd.DataFrame()

        values = series.tolist()
        residuals: List[float] = []
        for idx in range(1, len(values)):
            pred = cls._one_step_prediction(values[:idx])
            residuals.append(float(values[idx] - pred))

        if len(residuals) >= 3:
            residual_std = float(np.std(residuals, ddof=1))
        else:
            residual_std = float(np.std(values) if len(values) > 1 else max(values[0], 1.0) * 0.15)

        history = values.copy()
        last_date = pd.Timestamp(series.index.max()).floor("D")
        rows: List[Dict[str, float | pd.Timestamp]] = []
        for step in range(1, horizon_days + 1):
            pred = cls._one_step_prediction(history)
            sigma = residual_std * np.sqrt(step)
            lower = max(0.0, pred - 1.96 * sigma)
            upper = pred + 1.96 * sigma
            ts = last_date + pd.Timedelta(days=step)
            rows.append(
                {
                    "date": ts,
                    "predicted": float(pred),
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )
            history.append(pred)

        return pd.DataFrame(rows)

    @staticmethod
    def _confidence_label(sample_count: int, tier: int) -> str:
        if tier == 1 and sample_count >= 8:
            return "high"
        if sample_count >= 5:
            return "medium"
        return "low"

    @staticmethod
    def _seasonal_analog(
        series: pd.Series,
        target_date: pd.Timestamp,
    ) -> tuple[float, float, float, str, int, str]:
        hist = series.reset_index()
        hist.columns = ["date", "value"]
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=True).dt.floor("D")
        hist = hist.dropna(subset=["date", "value"])
        hist["month"] = hist["date"].dt.month
        hist["day_of_week"] = hist["date"].dt.day_name()

        target_month = int(target_date.month)
        target_dow = target_date.day_name()

        tiers = [
            ("same month + weekday", (hist["month"] == target_month) & (hist["day_of_week"] == target_dow)),
            ("same month", (hist["month"] == target_month)),
            ("same weekday", (hist["day_of_week"] == target_dow)),
            ("all history", pd.Series(True, index=hist.index)),
        ]

        selected = pd.Series(dtype=float)
        tier_label = "all history"
        tier_idx = 4
        for idx, (label, mask) in enumerate(tiers, start=1):
            sample = hist.loc[mask, "value"].astype(float).dropna()
            if len(sample) >= 3 or idx == len(tiers):
                selected = sample
                tier_label = label
                tier_idx = idx
                break

        if selected.empty:
            return 0.0, 0.0, 0.0, "no analog samples", 0, "low"

        pred = float(selected.mean())
        if len(selected) >= 2:
            lower = float(max(0.0, selected.quantile(0.10)))
            upper = float(selected.quantile(0.90))
        else:
            lower = float(max(0.0, pred * 0.80))
            upper = float(pred * 1.20)

        confidence = ForecastEngine._confidence_label(sample_count=len(selected), tier=tier_idx)
        return pred, lower, upper, tier_label, len(selected), confidence

    def forecast_arrivals(
        self,
        port: str,
        horizon_weeks: int = 4,
        vessel_type: Optional[str] = None,
    ) -> ForecastResult:
        if self.kpi.arrivals_daily.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["arrivals_daily.parquet is missing. Run KPI build first."],
            )

        work = self.kpi._filter_port(self.kpi.arrivals_daily, port)
        work = self.kpi._filter_vessel_type(work, vessel_type)
        if work.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No arrival history found for this port filter."],
            )

        series = self._prepare_series(work, "date", "arrivals_vessels")
        if len(series) < 14:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["Need at least 14 daily points for forecast stability."],
            )

        horizon_days = int(max(1, horizon_weeks) * 7)
        forecast_df = self._forecast_with_intervals(series, horizon_days=horizon_days)
        history_df = series.reset_index().rename(columns={"index": "date", "arrivals_vessels": "actual"})
        history_df.columns = ["date", "actual"]

        mean_pred = float(forecast_df["predicted"].mean())
        answer = f"Forecast mean arrivals are {mean_pred:.2f} vessels/day over the next {horizon_weeks} week(s)."
        notes = self.kpi.coverage_notes(work, "date")
        notes.append(f"Forecast horizon: {horizon_weeks} week(s)")

        return ForecastResult(
            status="ok",
            answer=answer,
            history=history_df,
            forecast=forecast_df,
            coverage_notes=notes,
            caveats=["Forecast uses weekly-seasonal baseline + 7-day moving average, with residual-based intervals."],
        )

    def forecast_congestion_for_date(
        self,
        port: str,
        target_date: str,
        horizon_weeks: int = 4,
    ) -> ForecastResult:
        target_ts = pd.to_datetime(target_date, errors="coerce", utc=True)
        if pd.isna(target_ts):
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["Target date is invalid. Use YYYY-MM-DD."],
            )
        target_ts = pd.Timestamp(target_ts).floor("D")

        if self.kpi.congestion.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["congestion_daily.parquet is missing. Run KPI build first."],
            )

        work = self.kpi._filter_port(self.kpi.congestion, port)
        if work.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No congestion history found for this port filter."],
            )

        series = self._prepare_series(work, "date", "congestion_index")
        if series.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No congestion values are available after filtering."],
            )

        history_df = series.reset_index().rename(columns={"index": "date", "congestion_index": "actual"})
        history_df.columns = ["date", "actual"]
        last_date = pd.Timestamp(series.index.max()).floor("D")

        if target_ts <= last_date and target_ts in series.index:
            actual = float(series.loc[target_ts])
            forecast_df = pd.DataFrame(
                [{"date": target_ts, "predicted": actual, "lower": actual, "upper": actual}]
            )
            return ForecastResult(
                status="ok",
                answer=(
                    f"Observed congestion index at {port or 'selected port'} on {target_ts.strftime('%Y-%m-%d')} "
                    f"was {actual:.2f}."
                ),
                history=history_df,
                forecast=forecast_df,
                coverage_notes=self.kpi.coverage_notes(work, "date"),
                caveats=[
                    "Target date is inside historical coverage; this is observed value, not a future forecast.",
                    "Congestion index is a proxy from arrivals and dwell-time availability, not berth-level operations.",
                ],
            )

        horizon_days = int(max(1, horizon_weeks) * 7)
        days_ahead = int((target_ts - last_date).days)

        if 0 < days_ahead <= horizon_days and len(series) >= 14:
            forecast_df = self._forecast_with_intervals(series, horizon_days=days_ahead)
            target_row = forecast_df.iloc[-1]
            pred = float(target_row["predicted"])
            lower = float(target_row["lower"])
            upper = float(target_row["upper"])
            confidence_note = (
                f"Confidence: medium (model-based near-term forecast, horizon {days_ahead} day(s))."
            )
            method_note = "Method: weekly-seasonal baseline + moving-average model."
        else:
            pred, lower, upper, tier_label, sample_count, confidence = self._seasonal_analog(
                series=series,
                target_date=target_ts,
            )
            forecast_df = pd.DataFrame(
                [{"date": target_ts, "predicted": pred, "lower": lower, "upper": upper}]
            )
            confidence_note = (
                f"Confidence: {confidence} (seasonal analog tier: {tier_label}, sample n={sample_count})."
            )
            method_note = f"Method: seasonal historical analog ({tier_label})."

        notes = self.kpi.coverage_notes(work, "date")
        notes.append(f"Target date: {target_ts.strftime('%Y-%m-%d')}")
        notes.append(method_note)

        caveats = [
            "Congestion index is a proxy from arrivals and dwell-time availability, not berth-level operations.",
            "Forecast is based on historical seasonal patterns in available data.",
            confidence_note,
        ]
        if "has_dwell" in work.columns and work["has_dwell"].fillna(False).mean() < 0.5:
            caveats.append("Dwell coverage is sparse, so forecast is more arrival-driven.")

        return ForecastResult(
            status="ok",
            answer=(
                f"Predicted congestion index at {port or 'selected port'} on {target_ts.strftime('%Y-%m-%d')} "
                f"is {pred:.2f} (range {lower:.2f} to {upper:.2f})."
            ),
            history=history_df,
            forecast=forecast_df,
            coverage_notes=notes,
            caveats=caveats,
        )

    def forecast_congestion(
        self,
        port: str,
        target_dow: str = "Friday",
        horizon_weeks: int = 4,
    ) -> ForecastResult:
        target = target_dow.strip().title()

        if self.kpi.congestion.empty:
            # Fall back to arrivals forecast if congestion table is unavailable.
            return self.forecast_arrivals(port=port, horizon_weeks=horizon_weeks)

        work = self.kpi._filter_port(self.kpi.congestion, port)
        if work.empty:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["No congestion history found for this port filter."],
            )

        series = self._prepare_series(work, "date", "congestion_index")
        if len(series) < 14:
            return ForecastResult(
                status="no_data",
                answer="I don't have evidence in the dataset to answer that.",
                history=None,
                forecast=None,
                coverage_notes=[],
                caveats=["Need at least 14 daily points for congestion forecast."],
            )

        horizon_days = int(max(1, horizon_weeks) * 7)
        forecast_df = self._forecast_with_intervals(series, horizon_days=horizon_days)
        forecast_df["day_of_week"] = pd.to_datetime(forecast_df["date"], utc=True).dt.day_name()
        target_rows = forecast_df[forecast_df["day_of_week"] == target]
        if target_rows.empty:
            target_rows = forecast_df.head(horizon_weeks)

        mean_pred = float(target_rows["predicted"].mean())
        low_pred = float(target_rows["lower"].mean())
        high_pred = float(target_rows["upper"].mean())

        answer = (
            f"Forecasted congestion index for {target} is {mean_pred:.2f} "
            f"(interval {low_pred:.2f} to {high_pred:.2f}) over the next {horizon_weeks} week(s)."
        )

        history_df = series.reset_index().rename(columns={"index": "date", "congestion_index": "actual"})
        history_df.columns = ["date", "actual"]

        notes = self.kpi.coverage_notes(work, "date")
        notes.append(f"Forecast target weekday: {target}")
        notes.append(f"Forecast horizon: {horizon_weeks} week(s)")

        caveats: List[str] = [
            "Congestion index is a proxy from arrivals and dwell-time availability, not berth-level operations.",
            "Forecast reflects historical weekly patterns only.",
        ]
        if work["has_dwell"].fillna(False).mean() < 0.5:
            caveats.append("Dwell coverage is sparse, so forecast is more arrival-driven.")

        return ForecastResult(
            status="ok",
            answer=answer,
            history=history_df,
            forecast=forecast_df,
            coverage_notes=notes,
            caveats=caveats,
        )
