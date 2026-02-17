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
