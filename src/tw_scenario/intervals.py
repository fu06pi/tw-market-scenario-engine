"""Past-only 80% marginal absolute-error bands, with explicit monitoring scores.

Temporal nonexchangeability means nominal coverage is a target, not a guarantee.
These separate price bands are not an OHLC joint distribution.
"""

import math

import numpy as np

FIELDS = ("open", "high", "low", "close")
MIN_CALIBRATION = 30


def calibrate(rows, forecast_date, alpha=0.2):
    dates = [r["date"] for r in rows]
    if not 0 < alpha < 1 or len(rows) < MIN_CALIBRATION:
        raise ValueError("invalid alpha or fewer than 30 calibration observations")
    if dates != sorted(set(dates)) or dates[-1] >= forecast_date:
        raise ValueError("calibration must be unique, ordered and strictly past")
    if len({(r["market"], r["method"]) for r in rows}) != 1:
        raise ValueError("calibration cannot pool markets or models")
    result = {
        "version": "marginal-absolute-bp-v1", "alpha": alpha, "n": len(rows),
        "market": rows[0]["market"], "method": rows[0]["method"],
        "first": dates[0], "last": dates[-1], "radius_bps": {},
        "joint_coverage_claim": False,
        "historical_availability_verified": False,
    }
    rank = min(len(rows), math.ceil((len(rows) + 1) * (1 - alpha)))
    for field in FIELDS:
        values = [r[f"{field}_ae_bps"] for r in rows]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("calibration errors must be finite and nonnegative")
        result["radius_bps"][field] = float(sorted(values)[rank - 1])
    return result


def bands(calibration, prices, reference, market, method, forecast_date):
    if calibration["market"] != market or calibration["method"] != method:
        raise ValueError("interval model/market mismatch")
    if calibration["last"] >= forecast_date:
        raise ValueError("non-past interval calibration")
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("invalid reference price")
    result = {}
    for field in FIELDS:
        radius = calibration["radius_bps"][field] * reference / 10000
        center = getattr(prices, field)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("invalid calibrated radius")
        result[field] = {"lower": max(0.0, center - radius), "upper": center + radius}
    return result


def interval_score(lower, upper, actual, alpha=0.2):
    if not all(math.isfinite(v) for v in (lower, upper, actual, alpha)):
        raise ValueError("non-finite interval score input")
    if lower > upper or not 0 < alpha < 1:
        raise ValueError("invalid interval bounds or alpha")
    return upper - lower + 2 / alpha * max(lower - actual, actual - upper, 0)


def summarize_intervals(rows):
    if not rows:
        return {"n": 0, "coverage": None, "width_bps": None, "interval_score_bps": None}
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError("duplicate interval evaluation dates")
    return {
        "n": len(rows),
        **{key: float(np.mean([r[key] for r in rows]))
           for key in ("coverage", "width_bps", "interval_score_bps")},
    }
