"""Deterministic analytics engine over KPI materialized tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _normalize_vessel_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().lower()


def _normalize_port_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip()


def _as_date_str(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _parse_window(window: Optional[str], max_date: Optional[pd.Timestamp]) -> tuple[Optional[str], Optional[str]]:
    if not window or max_date is None or pd.isna(max_date):
        return None, None
    m = re.match(r"last_(\d{1,2})_weeks", window)
    if not m:
        return None, None
    weeks = int(m.group(1))
    end = pd.Timestamp(max_date).floor("D")
    start = end - pd.Timedelta(days=7 * weeks)
    return _as_date_str(start), _as_date_str(end)


@dataclass
class AnalyticsResult:
    status: str
    answer: str
    table: Optional[pd.DataFrame]
    chart: Optional[pd.DataFrame]
    coverage_notes: List[str]
    caveats: List[str]


class KPIQueryEngine:
    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)
        self._arrivals_daily: Optional[pd.DataFrame] = None
        self._arrivals_hourly: Optional[pd.DataFrame] = None
        self._dwell: Optional[pd.DataFrame] = None
        self._occupancy: Optional[pd.DataFrame] = None
        self._congestion: Optional[pd.DataFrame] = None
        self._port_catalog: Optional[pd.DataFrame] = None
        self._caps: Optional[Dict[str, Any]] = None

    def _load_parquet(self, name: str) -> pd.DataFrame:
        path = self.processed_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    @property
    def arrivals_daily(self) -> pd.DataFrame:
        if self._arrivals_daily is None:
            df = self._load_parquet("arrivals_daily.parquet")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
            self._arrivals_daily = df
        return self._arrivals_daily

    @property
    def arrivals_hourly(self) -> pd.DataFrame:
        if self._arrivals_hourly is None:
            df = self._load_parquet("arrivals_hourly.parquet")
            if "datetime_hour" in df.columns:
                df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce", utc=True).dt.floor("h")
            self._arrivals_hourly = df
        return self._arrivals_hourly

    @property
    def dwell(self) -> pd.DataFrame:
        if self._dwell is None:
            df = self._load_parquet("dwell_time.parquet")
            for col in ("arrival_time", "departure_time", "arrival_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            self._dwell = df
        return self._dwell

    @property
    def occupancy(self) -> pd.DataFrame:
        if self._occupancy is None:
            df = self._load_parquet("occupancy_hourly.parquet")
            if "datetime_hour" in df.columns:
                df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce", utc=True).dt.floor("h")
            self._occupancy = df
        return self._occupancy

    @property
    def congestion(self) -> pd.DataFrame:
        if self._congestion is None:
            df = self._load_parquet("congestion_daily.parquet")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
            self._congestion = df
        return self._congestion

    @property
    def port_catalog(self) -> pd.DataFrame:
        if self._port_catalog is None:
            df = self._load_parquet("port_catalog.parquet")
            for col in ("first_seen", "last_seen"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.floor("D")
            self._port_catalog = df
        return self._port_catalog

    def capabilities(self) -> Dict[str, Any]:
        if self._caps is not None:
            return self._caps
        caps_path = self.processed_dir / "kpi_capabilities.json"
        if caps_path.exists():
            with caps_path.open("r", encoding="utf-8") as f:
                self._caps = json.load(f)
        else:
            self._caps = {
                "has_port_calls": bool((self.arrivals_daily.get("source_kind") == "port_call").any())
                if not self.arrivals_daily.empty
                else False,
                "has_ais_destination_proxy": bool((self.arrivals_daily.get("source_kind") == "ais_destination_proxy").any())
                if not self.arrivals_daily.empty
                else False,
                "has_dwell_time": not self.dwell.empty,
                "has_occupancy_hourly": not self.occupancy.empty,
            }
        return self._caps

    def coverage_notes(self, df: pd.DataFrame, date_col: str) -> List[str]:
        notes: List[str] = []
        if df.empty:
            notes.append("No rows available for the requested filters.")
            return notes
        start = pd.to_datetime(df[date_col], errors="coerce", utc=True).min()
        end = pd.to_datetime(df[date_col], errors="coerce", utc=True).max()
        if pd.notna(start) and pd.notna(end):
            notes.append(f"Coverage window: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        if "source_kind" in df.columns:
            sources = sorted(set(df["source_kind"].fillna("unknown").astype(str)))
            notes.append("Data sources used: " + ", ".join(sources))
        notes.append(f"Rows used: {len(df):,}")
        return notes

    def _filter_port(self, df: pd.DataFrame, port: Optional[str]) -> pd.DataFrame:
        token = _normalize_port_token(port)
        if not token or df.empty:
            return df
        code = token.upper().replace(" ", "")
        lower = token.lower()

        mask = pd.Series(False, index=df.index)
        if "port_key" in df.columns:
            mask |= df["port_key"].fillna("").astype(str).str.upper() == code
        if "locode_norm" in df.columns:
            mask |= df["locode_norm"].fillna("").astype(str).str.upper() == code
        if "port_name_norm" in df.columns:
            mask |= df["port_name_norm"].fillna("").astype(str).str.lower().str.contains(lower, regex=False)
        if "port_label" in df.columns:
            mask |= df["port_label"].fillna("").astype(str).str.lower().str.contains(lower, regex=False)

        filtered = df[mask]
        return filtered

    def _filter_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        date_from: Optional[str],
        date_to: Optional[str],
        window: Optional[str] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        date_series = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        out = df.copy()

        if (not date_from or not date_to) and window:
            win_from, win_to = _parse_window(window, date_series.max())
            date_from = date_from or win_from
            date_to = date_to or win_to

        if date_from:
            out = out[date_series >= pd.Timestamp(date_from, tz="UTC")]
            date_series = pd.to_datetime(out[date_col], errors="coerce", utc=True)
        if date_to:
            out = out[date_series <= pd.Timestamp(date_to, tz="UTC")]
        return out

    def _filter_vessel_type(self, df: pd.DataFrame, vessel_type: Optional[str]) -> pd.DataFrame:
        vt = _normalize_vessel_type(vessel_type)
        if not vt or df.empty or "vessel_type_norm" not in df.columns:
            return df
        return df[df["vessel_type_norm"].fillna("").astype(str).str.contains(vt, regex=False)]

    @staticmethod
    def _filter_dow(df: pd.DataFrame, date_col: str, dow: Optional[str]) -> pd.DataFrame:
        if not dow or df.empty:
            return df
        norm = dow.strip().title()
        work = df.copy()
        dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
        work = work[dates.dt.day_name() == norm]
        return work

    @staticmethod
    def unsupported(reason: str) -> AnalyticsResult:
        return AnalyticsResult(
            status="unsupported",
            answer="I don't have evidence in the dataset to answer that.",
            table=None,
            chart=None,
            coverage_notes=[],
            caveats=[reason],
        )

    @staticmethod
    def no_data(hint: str) -> AnalyticsResult:
        return AnalyticsResult(
            status="no_data",
            answer="I don't have evidence in the dataset to answer that.",
            table=None,
            chart=None,
            coverage_notes=[],
            caveats=[hint],
        )

    def get_arrivals(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing. Run `python -m src.kpi.build_kpis ...` first.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end, window=window)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_dow(work, "date", dow)

        if work.empty:
            return self.no_data("No arrival rows matched these filters. Broaden port/date/vessel-type constraints.")

        daily = (
            work.groupby("date", dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"), arrivals_events=("arrivals_events", "sum"))
            .reset_index()
            .sort_values("date")
        )
        total_arrivals = int(daily["arrivals_vessels"].sum())
        answer = (
            f"Matched {total_arrivals:,} vessel arrivals across {len(daily):,} day buckets"
            + (f" for {port}" if port else "")
            + "."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=daily,
            chart=daily.set_index("date")[["arrivals_vessels"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[
                "Arrivals are based on port-call events when available; AIS destination proxy is used otherwise.",
            ],
        )

    def top_ports_by_arrivals(
        self,
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
        top_n: int = 10,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")
        work = self._filter_dates(df, "date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_dow(work, "date", dow)
        if work.empty:
            return self.no_data("No arrivals available for this time range/filter.")

        top = (
            work.groupby(["port_key", "port_label"], dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"))
            .reset_index()
            .sort_values("arrivals_vessels", ascending=False)
            .head(max(1, top_n))
        )
        answer = f"Top {len(top)} ports by arrivals were computed for the selected filters."
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=top,
            chart=top.set_index("port_label")[["arrivals_vessels"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[],
        )

    def get_busiest_dow(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")
        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No rows available for day-of-week analysis.")

        dates = pd.to_datetime(work["date"], errors="coerce", utc=True)
        by_day = (
            work.assign(day_of_week=dates.dt.day_name())
            .groupby("day_of_week", dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"))
            .reset_index()
            .sort_values("arrivals_vessels", ascending=False)
        )
        busiest = by_day.iloc[0]
        answer = (
            f"Busiest weekday is {busiest['day_of_week']} with {int(busiest['arrivals_vessels']):,} arrivals"
            + (f" for {port}" if port else "")
            + "."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_day,
            chart=by_day.set_index("day_of_week")[["arrivals_vessels"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[],
        )

    def compare_weekdays(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        day_a: str,
        day_b: str,
        vessel_type: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No rows available for weekday comparison.")

        dates = pd.to_datetime(work["date"], errors="coerce", utc=True)
        by_day = (
            work.assign(day_of_week=dates.dt.day_name())
            .groupby("day_of_week", dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"))
            .reset_index()
        )
        day_a_title = day_a.title()
        day_b_title = day_b.title()
        pair = by_day[by_day["day_of_week"].isin([day_a_title, day_b_title])].copy()
        if pair.empty or len(pair) < 2:
            return self.no_data(f"Could not find both weekdays ({day_a_title}, {day_b_title}) in the filtered window.")

        a_val = float(pair[pair["day_of_week"] == day_a_title]["arrivals_vessels"].iloc[0])
        b_val = float(pair[pair["day_of_week"] == day_b_title]["arrivals_vessels"].iloc[0])
        if a_val > b_val:
            winner = day_a_title
            ratio = (a_val / max(b_val, 1.0))
        else:
            winner = day_b_title
            ratio = (b_val / max(a_val, 1.0))

        answer = (
            f"{winner} is busier in the filtered history. "
            f"{day_a_title}={int(a_val):,} vs {day_b_title}={int(b_val):,} arrivals "
            f"(~{ratio:.2f}x)."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=pair.sort_values("arrivals_vessels", ascending=False),
            chart=pair.set_index("day_of_week")[["arrivals_vessels"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=[],
        )

    def get_busiest_hour(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.arrivals_hourly
        if df.empty:
            return self.no_data("arrivals_hourly.parquet is missing.")
        work = self._filter_port(df, port)
        work = self._filter_dates(work, "datetime_hour", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        if work.empty:
            return self.no_data("No rows available for hourly pattern analysis.")

        hours = pd.to_datetime(work["datetime_hour"], errors="coerce", utc=True)
        by_hour = (
            work.assign(hour=hours.dt.hour)
            .groupby("hour", dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"))
            .reset_index()
            .sort_values("arrivals_vessels", ascending=False)
        )
        top = by_hour.iloc[0]
        answer = (
            f"Busiest hour is {int(top['hour']):02d}:00 UTC with {int(top['arrivals_vessels']):,} arrivals"
            + (f" for {port}" if port else "")
            + "."
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_hour,
            chart=by_hour.set_index("hour")[["arrivals_vessels"]],
            coverage_notes=self.coverage_notes(work, "datetime_hour"),
            caveats=[],
        )

    def get_avg_dwell_time(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
    ) -> AnalyticsResult:
        if self.dwell.empty:
            return self.unsupported("Dwell-time analysis requires port-call arrival and departure timestamps.")

        work = self._filter_port(self.dwell, port)
        work = self._filter_dates(work, "arrival_date", start, end)
        work = self._filter_vessel_type(work, vessel_type)
        work = self._filter_dow(work, "arrival_date", dow)
        if work.empty:
            return self.no_data("No dwell rows matched these filters.")

        median_dwell = float(work["dwell_minutes"].median())
        mean_dwell = float(work["dwell_minutes"].mean())
        answer = (
            f"Median dwell time is {median_dwell:.1f} minutes; mean dwell is {mean_dwell:.1f} minutes"
            + (f" for {port}" if port else "")
            + "."
        )

        by_type = (
            work.groupby("vessel_type_norm", dropna=False)
            .agg(
                calls=("mmsi", "size"),
                median_dwell_minutes=("dwell_minutes", "median"),
                mean_dwell_minutes=("dwell_minutes", "mean"),
            )
            .reset_index()
            .sort_values("calls", ascending=False)
        )
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_type,
            chart=None,
            coverage_notes=self.coverage_notes(work, "arrival_date"),
            caveats=[],
        )

    def get_congestion(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
        dow: Optional[str] = None,
        window: Optional[str] = None,
    ) -> AnalyticsResult:
        df = self.congestion
        if df.empty:
            return self.no_data("congestion_daily.parquet is missing.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end, window=window)
        work = self._filter_dow(work, "date", dow)
        if work.empty:
            return self.no_data("No congestion rows matched these filters.")

        by_day = (
            work.groupby("date", dropna=False)
            .agg(
                congestion_index=("congestion_index", "mean"),
                arrivals_vessels=("arrivals_vessels", "sum"),
                median_dwell_minutes=("median_dwell_minutes", "median"),
            )
            .reset_index()
            .sort_values("date")
        )
        mean_ci = float(by_day["congestion_index"].mean())
        max_row = by_day.loc[by_day["congestion_index"].idxmax()]

        answer = (
            f"Average congestion index is {mean_ci:.2f}; peak day is {max_row['date'].strftime('%Y-%m-%d')} "
            f"at {max_row['congestion_index']:.2f}."
        )

        caveats: List[str] = []
        if work["has_dwell"].fillna(False).mean() < 0.5:
            caveats.append("Dwell-time coverage is limited; congestion index leans more on arrivals volume.")

        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=by_day,
            chart=by_day.set_index("date")[["congestion_index"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=caveats,
        )

    def compare_ports(
        self,
        ports: Sequence[str],
        metric: str,
        start: Optional[str],
        end: Optional[str],
        vessel_type: Optional[str] = None,
        dow: Optional[str] = None,
    ) -> AnalyticsResult:
        if len(ports) < 2:
            return self.no_data("Comparison needs at least two ports in the question.")

        metric_norm = metric.lower()
        rows: List[Dict[str, Any]] = []

        for port in ports:
            if "dwell" in metric_norm:
                result = self.get_avg_dwell_time(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow)
                if result.status != "ok" or result.table is None:
                    continue
                value = float(result.table["median_dwell_minutes"].median())
                rows.append({"port": port, "metric": "median_dwell_minutes", "value": value})
            elif "congestion" in metric_norm:
                result = self.get_congestion(port=port, start=start, end=end, dow=dow)
                if result.status != "ok" or result.table is None:
                    continue
                value = float(result.table["congestion_index"].mean())
                rows.append({"port": port, "metric": "congestion_index", "value": value})
            else:
                result = self.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow)
                if result.status != "ok" or result.table is None:
                    continue
                value = float(result.table["arrivals_vessels"].sum())
                rows.append({"port": port, "metric": "arrivals_vessels", "value": value})

        if not rows:
            return self.no_data("No comparable metrics were available for the requested ports.")

        comp = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
        answer = f"{comp.iloc[0]['port']} ranks highest for {comp.iloc[0]['metric']} in this window."
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=comp,
            chart=comp.set_index("port")[["value"]],
            coverage_notes=[f"Ports compared: {', '.join(ports)}"],
            caveats=[],
        )

    def diagnose_congestion(
        self,
        port: Optional[str],
        target_date: Optional[str],
    ) -> AnalyticsResult:
        if not target_date:
            return self.no_data("Diagnostic questions need a specific date (YYYY-MM-DD).")

        arrivals = self._filter_port(self.arrivals_daily, port)
        arrivals = self._filter_dates(arrivals, "date", target_date, target_date)
        if arrivals.empty:
            return self.no_data("No arrivals found for this port/date diagnostic query.")

        arrivals_total = int(arrivals["arrivals_vessels"].sum())
        by_type = (
            arrivals.groupby("vessel_type_norm", dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"))
            .reset_index()
            .sort_values("arrivals_vessels", ascending=False)
        )

        dwell_note = ""
        dwell_stats = pd.DataFrame()
        if not self.dwell.empty:
            dwell = self._filter_port(self.dwell, port)
            dwell = self._filter_dates(dwell, "arrival_date", target_date, target_date)
            if not dwell.empty:
                dwell_note = (
                    f" Median dwell {float(dwell['dwell_minutes'].median()):.1f} minutes "
                    f"across {len(dwell):,} calls."
                )
                dwell_stats = (
                    dwell.groupby("vessel_type_norm", dropna=False)
                    .agg(median_dwell_minutes=("dwell_minutes", "median"), calls=("mmsi", "size"))
                    .reset_index()
                    .sort_values("calls", ascending=False)
                )

        answer = (
            f"On {target_date}, {arrivals_total:,} arrivals were recorded"
            + (f" for {port}" if port else "")
            + "."
            + dwell_note
        )

        merged = by_type.copy()
        if not dwell_stats.empty:
            merged = merged.merge(dwell_stats, on="vessel_type_norm", how="left")

        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=merged,
            chart=by_type.set_index("vessel_type_norm")[["arrivals_vessels"]],
            coverage_notes=[f"Diagnostic date: {target_date}", f"Rows used: {len(arrivals):,}"],
            caveats=["Diagnostic breakdown reflects available AIS/port-call fields only."],
        )

    def detect_arrival_spikes(
        self,
        port: Optional[str],
        start: Optional[str],
        end: Optional[str],
    ) -> AnalyticsResult:
        df = self.arrivals_daily
        if df.empty:
            return self.no_data("arrivals_daily.parquet is missing.")

        work = self._filter_port(df, port)
        work = self._filter_dates(work, "date", start, end)
        if work.empty:
            return self.no_data("No arrival rows matched these filters.")

        daily = (
            work.groupby("date", dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"))
            .reset_index()
            .sort_values("date")
            .reset_index(drop=True)
        )
        daily["roll_mean_7"] = daily["arrivals_vessels"].rolling(7, min_periods=3).mean().shift(1)
        daily["roll_std_7"] = daily["arrivals_vessels"].rolling(7, min_periods=3).std().shift(1)
        daily["threshold"] = daily["roll_mean_7"] + 2.0 * daily["roll_std_7"].fillna(0)
        spikes = daily[daily["arrivals_vessels"] > daily["threshold"]].copy()

        if spikes.empty:
            return AnalyticsResult(
                status="ok",
                answer="No statistically unusual arrival spikes were detected in the selected period.",
                table=daily.tail(20),
                chart=daily.set_index("date")[["arrivals_vessels", "threshold"]],
                coverage_notes=self.coverage_notes(work, "date"),
                caveats=["Spike rule: arrivals > rolling_mean_7 + 2*rolling_std_7."],
            )

        answer = f"Detected {len(spikes)} potential arrival spike days." 
        return AnalyticsResult(
            status="ok",
            answer=answer,
            table=spikes[["date", "arrivals_vessels", "threshold"]],
            chart=daily.set_index("date")[["arrivals_vessels", "threshold"]],
            coverage_notes=self.coverage_notes(work, "date"),
            caveats=["Spike rule: arrivals > rolling_mean_7 + 2*rolling_std_7."],
        )
