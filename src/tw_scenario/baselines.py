from __future__ import annotations

from dataclasses import dataclass

from .models import Direction


@dataclass(frozen=True)
class OvernightFeatures:
    taifex_night_return_pct: float
    sox_return_pct: float
    tsmc_adr_return_pct: float


def overnight_gap_baseline(x: OvernightFeatures) -> Direction:
    """Naive baseline for OPEN direction only.

    Intentionally does not claim to predict the close. This is a benchmark the
    richer model must beat out-of-sample.
    """
    score = (
        0.50 * x.taifex_night_return_pct
        + 0.25 * x.sox_return_pct
        + 0.25 * x.tsmc_adr_return_pct
    )
    if score > 0.15:
        return Direction.UP
    if score < -0.15:
        return Direction.DOWN
    return Direction.FLAT
