from __future__ import annotations

from math import fsum

from .models import Direction, MarketRealization, PredictionSnapshot
from .policy import ScoringPolicy


def realized_direction(return_bps: float, flat_threshold_bps: float = 5.0) -> Direction:
    if return_bps > flat_threshold_bps:
        return Direction.UP
    if return_bps < -flat_threshold_bps:
        return Direction.DOWN
    return Direction.FLAT


def direction_hit(predicted: Direction, realized: Direction) -> float:
    return float(predicted == realized)


def absolute_error(predicted: float | None, realized: float) -> float | None:
    if predicted is None:
        return None
    return abs(predicted - realized)


def multiclass_brier(probabilities: dict[str, float], outcome: str) -> float:
    classes = ("bull", "base", "bear")
    if set(probabilities) != set(classes):
        raise ValueError("probabilities must contain bull/base/bear")
    return fsum((probabilities[c] - float(c == outcome)) ** 2 for c in classes) / len(classes)


def infer_simple_outcome(
    realization: MarketRealization, policy: ScoringPolicy | None = None
) -> str:
    policy = policy or ScoringPolicy()
    r = realization.close_return_bps
    if r > policy.bull_close_return_bps:
        return "bull"
    if r < policy.bear_close_return_bps:
        return "bear"
    return "base"


def evaluate_snapshot(
    snapshot: PredictionSnapshot,
    realization: MarketRealization,
    policy: ScoringPolicy | None = None,
) -> dict:
    """Diagnostic scores only; formal inclusion is decided by the audit gate."""
    policy = policy or ScoringPolicy()
    if snapshot.trading_date != realization.trading_date:
        raise ValueError("prediction and realization dates do not match")

    open_realized = realized_direction(realization.open_gap_bps, policy.flat_threshold_bps)
    close_realized = realized_direction(realization.close_return_bps, policy.flat_threshold_bps)
    probs = snapshot.scenarios.model_dump()
    scenario_outcome = infer_simple_outcome(realization, policy)

    return {
        "trading_date": snapshot.trading_date,
        "open_direction_realized": open_realized.value,
        "open_direction_hit": direction_hit(snapshot.targets.open_direction, open_realized),
        "close_direction_realized": close_realized.value,
        "close_direction_hit": direction_hit(snapshot.targets.close_direction, close_realized),
        "open_gap_bps_realized": realization.open_gap_bps,
        "open_gap_mae_bps": absolute_error(
            snapshot.targets.expected_open_gap_bps, realization.open_gap_bps
        ),
        "close_return_bps_realized": realization.close_return_bps,
        "close_return_mae_bps": absolute_error(
            snapshot.targets.expected_close_return_bps, realization.close_return_bps
        ),
        "mfe_bps": realization.mfe_bps,
        "mae_bps": realization.mae_bps,
        "scenario_outcome": scenario_outcome,
        "brier_score": multiclass_brier(probs, scenario_outcome),
    }
