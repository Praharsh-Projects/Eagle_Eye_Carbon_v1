"""Backtest forecast quality for arrivals/congestion proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.forecast.forecast import ForecastEngine


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Stabilize percentage errors for near-zero congestion proxy values.
    denom = np.maximum(np.abs(y_true), 1.0)
    values = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(values) * 100.0)


def backtest_metric(
    engine: ForecastEngine,
    metric: str,
    min_history_days: int = 60,
    test_days: int = 28,
    max_ports: int = 20,
) -> Dict[str, object]:
    if metric == "congestion_index":
        base = engine.kpi.congestion.copy()
        value_col = "congestion_index"
        date_col = "date"
    else:
        base = engine.kpi.arrivals_daily.copy()
        value_col = "arrivals_vessels"
        date_col = "date"

    if base.empty:
        return {"metric": metric, "skipped": True, "reason": f"No data for {metric}"}

    port_rank = (
        base.groupby("port_key", dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(max_ports)
        .index.tolist()
    )

    rows: List[Dict[str, object]] = []
    for port in port_rank:
        if metric == "congestion_index":
            port_df = engine.kpi._filter_port(engine.kpi.congestion, port)
        else:
            port_df = engine.kpi._filter_port(engine.kpi.arrivals_daily, port)

        series = (
            port_df.groupby(date_col, dropna=False)[value_col]
            .sum()
            .sort_index()
            .astype(float)
        )

        if len(series) < (min_history_days + test_days):
            continue

        train = series.iloc[:-test_days]
        test = series.iloc[-test_days:]
        history = train.tolist()
        preds: List[float] = []

        for actual in test.tolist():
            pred = engine._one_step_prediction(history)
            preds.append(pred)
            history.append(float(actual))

        y_true = np.array(test.tolist(), dtype=float)
        y_pred = np.array(preds, dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = _mape(y_true, y_pred)

        rows.append(
            {
                "port_key": port,
                "mae": mae,
                "mape": mape,
                "test_points": int(len(test)),
            }
        )

    if not rows:
        return {
            "metric": metric,
            "skipped": True,
            "reason": "No ports with enough history for backtest.",
        }

    df = pd.DataFrame(rows).sort_values("mae")
    return {
        "metric": metric,
        "skipped": False,
        "ports_evaluated": int(len(df)),
        "mae_mean": float(df["mae"].mean()),
        "mape_mean": float(df["mape"].mean()),
        "per_port": df.to_dict(orient="records"),
    }


def run_backtest(
    processed_dir: str | Path = "data/processed",
    out_path: str | Path = "data/processed/forecast_backtest.json",
    min_history_days: int = 60,
    test_days: int = 28,
    max_ports: int = 20,
) -> Dict[str, object]:
    engine = ForecastEngine(processed_dir=processed_dir)
    arrivals_metrics = backtest_metric(
        engine,
        metric="arrivals_vessels",
        min_history_days=min_history_days,
        test_days=test_days,
        max_ports=max_ports,
    )
    congestion_metrics = backtest_metric(
        engine,
        metric="congestion_index",
        min_history_days=min_history_days,
        test_days=test_days,
        max_ports=max_ports,
    )

    payload = {
        "settings": {
            "processed_dir": str(processed_dir),
            "min_history_days": min_history_days,
            "test_days": test_days,
            "max_ports": max_ports,
        },
        "arrivals": arrivals_metrics,
        "congestion": congestion_metrics,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run forecast backtest for arrivals/congestion proxies.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out", default="data/processed/forecast_backtest.json")
    parser.add_argument("--min_history_days", type=int, default=60)
    parser.add_argument("--test_days", type=int, default=28)
    parser.add_argument("--max_ports", type=int, default=20)
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    payload = run_backtest(
        processed_dir=args.processed_dir,
        out_path=args.out,
        min_history_days=args.min_history_days,
        test_days=args.test_days,
        max_ports=args.max_ports,
    )
    arrivals = payload["arrivals"]
    congestion = payload["congestion"]

    if arrivals.get("skipped"):
        print(f"Arrivals backtest skipped: {arrivals.get('reason')}")
    else:
        print(
            "Arrivals backtest:",
            f"ports={arrivals['ports_evaluated']}",
            f"MAE={arrivals['mae_mean']:.3f}",
            f"MAPE={arrivals['mape_mean']:.2f}%",
        )

    if congestion.get("skipped"):
        print(f"Congestion backtest skipped: {congestion.get('reason')}")
    else:
        print(
            "Congestion backtest:",
            f"ports={congestion['ports_evaluated']}",
            f"MAE={congestion['mae_mean']:.3f}",
            f"MAPE={congestion['mape_mean']:.2f}%",
        )

    print(f"Backtest output: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
