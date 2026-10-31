"""Cumulative metrics for immutable genuine pre-open forward snapshots."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from .contracts import OHLC, digest
from .daily import (
    OHLCEvaluationV1,
    OHLCRealizationV1,
    load_snapshot,
)


def _read_record(path: Path, record_type):
    return record_type.model_validate_json(path.read_text(encoding="utf-8"))


def _direction(value_bps: float, neutral_bps: float) -> str:
    if value_bps > neutral_bps:
        return "up"
    if value_bps < -neutral_bps:
        return "down"
    return "flat"


def _market_metrics(rows: list[dict], neutral_bps: float) -> dict:
    if not rows:
        return {
            "evaluated_days": 0,
            "ohlc": {},
            "direction": {
                "neutral_bps": neutral_bps,
                "accuracy": None,
                "confusion_matrix": {},
            },
            "intervals": {},
            "model_usage": {},
            "fallback_days": 0,
        }
    ohlc = {}
    for field in OHLC:
        point_errors = [row["absolute_error_points"][field] for row in rows]
        bps_errors = [row["absolute_error_bps"][field] for row in rows]
        signed = [row["signed_error_points"][field] for row in rows]
        ohlc[field] = {
            "mae_points": sum(point_errors) / len(rows),
            "mae_bps": sum(bps_errors) / len(rows),
            "rmse_points": math.sqrt(sum(value * value for value in signed) / len(rows)),
        }

    labels = ("down", "flat", "up")
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    correct = 0
    for row in rows:
        actual = _direction(row["actual_close_return_bps"], neutral_bps)
        predicted = _direction(row["predicted_close_return_bps"], neutral_bps)
        confusion[actual][predicted] += 1
        correct += actual == predicted

    intervals = {}
    for field in OHLC:
        covered = []
        widths = []
        scores = []
        for row in rows:
            lower, upper = row["intervals"][field]
            actual = row["actual"][field]
            covered.append(lower <= actual <= upper)
            widths.append(upper - lower)
            score = upper - lower
            if actual < lower:
                score += 10 * (lower - actual)
            elif actual > upper:
                score += 10 * (actual - upper)
            scores.append(score)
        intervals[field] = {
            "coverage": sum(covered) / len(rows),
            "mean_width_points": sum(widths) / len(rows),
            "mean_interval_score": sum(scores) / len(rows),
        }

    return {
        "evaluated_days": len(rows),
        "ohlc": ohlc,
        "gap_mae_bps": sum(abs(row["gap_error_bps"]) for row in rows) / len(rows),
        "close_return_mae_bps": (
            sum(abs(row["close_return_error_bps"]) for row in rows) / len(rows)
        ),
        "range_mae_points": sum(abs(row["range_error_points"]) for row in rows)
        / len(rows),
        "direction": {
            "neutral_bps": neutral_bps,
            "accuracy": correct / len(rows),
            "confusion_matrix": confusion,
        },
        "intervals": intervals,
        "model_usage": dict(sorted(Counter(row["model_name"] for row in rows).items())),
        "fallback_days": sum(row["fallback_used"] for row in rows),
    }


def build_forward_metrics(
    output_root: Path,
    *,
    generated_at: datetime,
    neutral_bps: float,
) -> dict:
    """Recompute metrics from every frozen snapshot and append-only evaluation."""
    if generated_at.utcoffset() is None:
        raise ValueError("forward metrics generated_at must be timezone-aware")
    snapshots = []
    seen_dates = set()
    market_rows = {"TAIEX": [], "TX": []}
    missing = []
    for path in sorted((output_root / "forward-snapshots").glob("*.json")):
        snapshot = load_snapshot(path)
        if path.stem != snapshot.trading_date.isoformat():
            raise ValueError("forward snapshot filename must equal trading date")
        if snapshot.trading_date in seen_dates:
            raise ValueError("duplicate forward snapshot trading date")
        seen_dates.add(snapshot.trading_date)
        if snapshot.timeliness != "ON_TIME" or snapshot.data_quality.status != "PASS":
            raise ValueError("forward metrics accept only on-time PASS snapshots")
        snapshots.append(
            {
                "trading_date": str(snapshot.trading_date),
                "forecast_id": snapshot.forecast_id,
                "snapshot_hash": snapshot.snapshot_hash,
            }
        )
        for forecast in snapshot.forecasts:
            contract = forecast.contract or "cash"
            realization_path = (
                output_root
                / "realizations-v1"
                / snapshot.forecast_id
                / f"{forecast.instrument}-{contract}.json"
            )
            evaluation_path = (
                output_root
                / "evaluations-v1"
                / snapshot.forecast_id
                / f"{forecast.instrument}-{contract}.json"
            )
            if not realization_path.exists() or not evaluation_path.exists():
                missing.append(
                    {
                        "trading_date": str(snapshot.trading_date),
                        "instrument": forecast.instrument,
                        "reason": "post-close realization/evaluation not yet available",
                    }
                )
                continue
            realization = _read_record(realization_path, OHLCRealizationV1)
            evaluation = _read_record(evaluation_path, OHLCEvaluationV1)
            if (
                realization.forecast_id != snapshot.forecast_id
                or realization.snapshot_hash != snapshot.snapshot_hash
                or evaluation.forecast_id != snapshot.forecast_id
                or evaluation.snapshot_hash != snapshot.snapshot_hash
                or evaluation.realization_id != realization.realization_id
            ):
                raise ValueError("forward evaluation is not bound to its frozen snapshot")
            market_rows[forecast.instrument].append(
                {
                    "trading_date": str(snapshot.trading_date),
                    "actual": realization.actual.model_dump(mode="python"),
                    "absolute_error_points": evaluation.absolute_error_points.model_dump(
                        mode="python"
                    ),
                    "absolute_error_bps": evaluation.absolute_error_bps.model_dump(
                        mode="python"
                    ),
                    "signed_error_points": evaluation.signed_error_points.model_dump(
                        mode="python"
                    ),
                    "gap_error_bps": evaluation.gap_error_bps,
                    "close_return_error_bps": evaluation.close_return_error_bps,
                    "range_error_points": evaluation.range_error_points,
                    "predicted_close_return_bps": (
                        forecast.point.close / forecast.previous_close - 1
                    )
                    * 10_000,
                    "actual_close_return_bps": (
                        realization.actual.close / forecast.previous_close - 1
                    )
                    * 10_000,
                    "intervals": {
                        field: [
                            getattr(forecast.intervals, field).lower,
                            getattr(forecast.intervals, field).upper,
                        ]
                        for field in OHLC
                    },
                    "model_name": forecast.model.model_name,
                    "fallback_used": forecast.model.fallback_used,
                }
            )
    report = {
        "schema_version": "genuine-forward-metrics-v1",
        "status": "FORWARD_EVIDENCE_ACCUMULATING",
        "generated_at": generated_at.isoformat(),
        "production_switch": False,
        "genuine_forward_only": True,
        "snapshot_days": len(snapshots),
        "fully_evaluated_days": min(len(market_rows["TAIEX"]), len(market_rows["TX"])),
        "snapshots": snapshots,
        "pending_evaluations": missing,
        "instruments": {
            market: _market_metrics(rows, neutral_bps)
            for market, rows in market_rows.items()
        },
        "interpretation": (
            "Forward sample metrics describe forecast error only. Small samples do not "
            "establish an 85% accuracy claim, model acceptance, or profitability."
        ),
    }
    report["report_hash"] = digest(report)
    return report


def write_forward_metrics(report: dict, output_root: Path, forecast_id: str) -> Path:
    path = output_root / "forward-metrics" / f"{forecast_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path
