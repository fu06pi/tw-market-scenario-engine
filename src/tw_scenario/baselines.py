from __future__ import annotations

from dataclasses import dataclass
from math import expm1, isfinite
from statistics import mean

from .metrics import realized_direction
from .models import Direction
from .policy import ScoringPolicy
from .split_forecast import OHLC, Components, PreopenInputs, TrainingExample, encode_components

VERSION = "simple-baselines-v1"
PRICE_BASELINES = ("zero_return", "expanding_mean", "previous_return", "rolling_mean_20")
DIRECTION_BASELINES = (
    "always_up",
    "always_down",
    "always_flat",
    "historical_majority",
    "previous_direction",
)
OVERNIGHT_FEATURES = ("taifex_night_return_pct", "sox_return_pct", "tsmc_adr_return_pct")


@dataclass(frozen=True)
class OvernightFeatures:
    taifex_night_return_pct: float
    sox_return_pct: float
    tsmc_adr_return_pct: float

    def __post_init__(self):
        if not all(isfinite(getattr(self, name)) for name in OVERNIGHT_FEATURES):
            raise ValueError("overnight features must be finite")


def overnight_gap_baseline(x: OvernightFeatures) -> Direction:
    """Naive baseline for OPEN direction only.

    Intentionally does not claim to predict the close. This is a benchmark the
    richer model must beat out-of-sample.
    """
    score = (
        0.50 * x.taifex_night_return_pct + 0.25 * x.sox_return_pct + 0.25 * x.tsmc_adr_return_pct
    )
    if score > 0.15:
        return Direction.UP
    if score < -0.15:
        return Direction.DOWN
    return Direction.FLAT


@dataclass(frozen=True)
class BaselinePrediction:
    name: str
    directions: dict[str, Direction]
    components: Components | None = None
    prices: OHLC | None = None
    version: str = VERSION
    information_set: str = "past_day_ohlc"


def component_directions(components: Components, policy: ScoringPolicy) -> dict[str, Direction]:
    return {
        name: realized_direction(expm1(value) * 10000, policy.flat_threshold_bps)
        for name, value in (
            ("gap", components.g),
            ("body", components.b),
            ("close", components.g + components.b),
        )
    }


def price_baseline(
    name: str,
    past: list[Components],
    previous_close: float,
    policy: ScoringPolicy | None = None,
) -> BaselinePrediction:
    """Pure research core. Pass past labels only; pre-open users use the gated wrapper.

    All price baselines use the same expanding mean log-wick estimator. The
    fixed 20-session window affects gap/body means only; no window search.
    """
    if name not in PRICE_BASELINES:
        raise ValueError(f"unknown price baseline: {name}")
    if not past or not isfinite(previous_close) or previous_close <= 0:
        raise ValueError("price baselines require history and a finite positive reference")
    if name == "rolling_mean_20" and len(past) < 20:
        raise ValueError("rolling_mean_20 requires 20 past samples")
    selected = past[-20:] if name == "rolling_mean_20" else past
    if name == "zero_return":
        g = b = 0.0
    elif name == "previous_return":
        g, b = past[-1].g, past[-1].b
    else:
        g, b = mean(c.g for c in selected), mean(c.b for c in selected)
    components = Components(g=g, b=b, u=mean(c.u for c in past), d=mean(c.d for c in past))
    return BaselinePrediction(
        name,
        component_directions(components, policy or ScoringPolicy()),
        components,
        components.reconstruct(previous_close),
    )


def direction_baseline(
    name: str,
    past: list[Components],
    policy: ScoringPolicy | None = None,
) -> BaselinePrediction:
    """Classification-only baselines never invent numerical prices or probabilities."""
    if name not in DIRECTION_BASELINES:
        raise ValueError(f"unknown direction baseline: {name}")
    policy = policy or ScoringPolicy()
    constants = {
        "always_up": Direction.UP,
        "always_down": Direction.DOWN,
        "always_flat": Direction.FLAT,
    }
    if name in constants:
        directions = {target: constants[name] for target in ("gap", "body", "close")}
    elif not past:
        raise ValueError("historical direction baselines require past samples")
    elif name == "previous_direction":
        directions = component_directions(past[-1], policy)
    else:
        historical = [component_directions(c, policy) for c in past]
        # Predeclared tie order: flat, up, down. No current outcome is consulted.
        tie_order = (Direction.FLAT, Direction.UP, Direction.DOWN)
        directions = {
            target: max(tie_order, key=lambda d: sum(r[target] == d for r in historical))
            for target in ("gap", "body", "close")
        }
    return BaselinePrediction(name, directions)


def predict_preopen_baseline(
    name: str,
    inputs: PreopenInputs,
    history: list[TrainingExample],
    policy: ScoringPolicy | None = None,
) -> BaselinePrediction:
    """Use the same date, cutoff, market and contract gates as numerical forecasts.

    This returns a research candidate, not a persisted/formally eligible forecast.
    """
    policy = policy or ScoringPolicy()
    ordered = sorted(history, key=lambda row: row.inputs.trading_date)
    if len({r.inputs.trading_date for r in ordered}) != len(ordered):
        raise ValueError("duplicate baseline training dates")
    for row in ordered:
        if row.inputs.trading_date >= inputs.trading_date:
            raise ValueError("baseline history must strictly precede forecast date")
        if row.actual.available_at > inputs.data_cutoff_at:
            raise ValueError("baseline outcome was unavailable at cutoff")
        if (row.inputs.market, row.inputs.contract) != (inputs.market, inputs.contract):
            raise ValueError("baseline history market/contract must match the request")
    if name == "overnight_gap":
        values = {f.name: f.value for f in inputs.features}
        missing = set(OVERNIGHT_FEATURES) - set(values)
        if missing:
            raise ValueError(f"missing overnight features: {sorted(missing)}")
        direction = overnight_gap_baseline(
            OvernightFeatures(**{key: values[key] for key in OVERNIGHT_FEATURES})
        )
        return BaselinePrediction(name, {"gap": direction}, information_set="night_and_overseas")
    past = [encode_components(r.inputs.previous_close, r.actual) for r in ordered]
    if name in PRICE_BASELINES:
        return price_baseline(name, past, inputs.previous_close, policy)
    return direction_baseline(name, past, policy)
