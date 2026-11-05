"""Pre-open research forecaster: independently fit gap g and session body b.

All heads consume only information available by 08:30. Today's actual open is
never an input to the body head. Wick sizes are historical means, not fitted
TimesFM forecasts or calibrated intervals. This module does not switch live jobs.
"""

from __future__ import annotations

from datetime import date, datetime, time
from math import exp, expm1, isclose, log
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import realized_direction
from .policy import ScoringPolicy
from .snapshot import HASH_VERSION, document_hash

TAIPEI = ZoneInfo("Asia/Taipei")


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class TimedFeature(ContractModel):
    name: str = Field(min_length=1)
    value: float
    available_at: AwareDatetime
    observed_at: AwareDatetime
    kind: Literal["historical", "completed_overnight", "completed_overseas", "calendar"]
    source: str = Field(min_length=1)
    availability_basis: str = Field(min_length=1)


class PreopenInputs(ContractModel):
    forecast_id: str = Field(min_length=1)
    trading_date: date
    market: Literal["TAIEX", "TX"]
    contract: str | None = Field(default=None, pattern=r"^[12]\d{3}(0[1-9]|1[0-2])$")
    previous_close: float = Field(gt=0)
    previous_close_date: date
    previous_close_available_at: AwareDatetime
    previous_close_source: str = Field(min_length=1)
    previous_close_availability_basis: str = Field(min_length=1)
    previous_close_contract: str | None = None
    data_cutoff_at: AwareDatetime
    generated_at: AwareDatetime
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    features: tuple[TimedFeature, ...]

    @model_validator(mode="after")
    def enforce_asof(self):
        cutoff = self.data_cutoff_at.astimezone(TAIPEI)
        generated = self.generated_at.astimezone(TAIPEI)
        if cutoff.date() != self.trading_date or cutoff.time() > time(8, 30):
            raise ValueError("pre-open cutoff must be on the trading date by 08:30 Taipei")
        if (
            generated.date() != self.trading_date
            or generated < cutoff
            or generated.time() >= time(8, 40)
        ):
            raise ValueError("generation must follow cutoff and precede 08:40 Taipei")
        if self.previous_close_date >= self.trading_date:
            raise ValueError("previous close must be from a preceding trading day")
        if self.previous_close_available_at > self.data_cutoff_at:
            raise ValueError("previous close was unavailable at forecast cutoff")
        if self.market == "TX":
            if not self.contract or self.previous_close_contract != self.contract:
                raise ValueError(
                    "TX requires an explicit contract and same-contract previous close"
                )
        elif self.contract is not None or self.previous_close_contract is not None:
            raise ValueError("TAIEX must not have a futures contract")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        for feature in self.features:
            if feature.available_at > self.data_cutoff_at:
                raise ValueError(f"feature {feature.name} was unavailable at cutoff")
            observed = feature.observed_at.astimezone(TAIPEI)
            if feature.kind != "calendar" and feature.available_at < feature.observed_at:
                raise ValueError("observed market data cannot be available before observation")
            if feature.kind != "calendar" and observed > cutoff:
                raise ValueError(f"feature {feature.name} is from an unfinished session")
            if feature.kind == "historical" and observed.date() >= self.trading_date:
                raise ValueError("same-day daytime observations are not historical features")
        return self


class OHLC(ContractModel):
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("OHLC requires H >= max(O,C) and L <= min(O,C)")
        return self


class ObservedBar(OHLC):
    trading_date: date
    market: Literal["TAIEX", "TX"]
    contract: str | None = Field(default=None, pattern=r"^[12]\d{3}(0[1-9]|1[0-2])$")
    session: Literal["day"] = "day"
    available_at: AwareDatetime
    source: str = Field(min_length=1)
    availability_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def observation_identity(self):
        if self.market == "TX" and not self.contract:
            raise ValueError("TX observations require a contract")
        if self.market == "TAIEX" and self.contract is not None:
            raise ValueError("TAIEX observations must not have a contract")
        end = datetime.combine(
            self.trading_date, time(13, 45) if self.market == "TX" else time(13, 30), tzinfo=TAIPEI
        )
        if self.available_at < end:
            raise ValueError("a complete daytime OHLC cannot be available before session end")
        return self


class TrainingExample(ContractModel):
    inputs: PreopenInputs
    actual: ObservedBar

    @model_validator(mode="after")
    def matching_identity(self):
        for field in ("trading_date", "market", "contract"):
            if getattr(self.inputs, field) != getattr(self.actual, field):
                raise ValueError(f"training inputs and actual {field} do not match")
        return self


class SplitConfig(ContractModel):
    version: str = "split-ridge-v1"
    gap_features: tuple[str, ...] = ("night_return_bps",)
    body_features: tuple[str, ...] = ("previous_body_bps", "previous_range_bps")
    gap_alpha: float = Field(default=1, gt=0)
    body_alpha: float = Field(default=1, gt=0)
    min_training_samples: int = Field(default=60, ge=3)

    @model_validator(mode="after")
    def nonempty_feature_sets(self):
        for names in (self.gap_features, self.body_features):
            if not names or len(set(names)) != len(names):
                raise ValueError("each head needs nonempty, unique feature names")
        return self


class Components(ContractModel):
    g: float
    b: float
    u: float = Field(ge=0)
    d: float = Field(ge=0)

    def reconstruct(self, previous_close: float) -> OHLC:
        opening = previous_close * exp(self.g)
        closing = opening * exp(self.b)
        return OHLC(
            open=opening,
            close=closing,
            high=max(opening, closing) * exp(self.u),
            low=min(opening, closing) * exp(-self.d),
        )


def encode_components(previous_close: float, bar: OHLC) -> Components:
    if previous_close <= 0:
        raise ValueError("previous_close must be positive")
    return Components(
        g=log(bar.open / previous_close),
        b=log(bar.close / bar.open),
        u=log(bar.high / max(bar.open, bar.close)),
        d=log(min(bar.open, bar.close) / bar.low),
    )


class SplitForecast(ContractModel):
    schema_version: Literal["ohlc-split-v2"] = "ohlc-split-v2"
    hash_version: Literal["sha256-json-v2"] = HASH_VERSION
    performance_scope: Literal["research_candidate"] = "research_candidate"
    model_version: str = "split-ridge-v1"
    inputs: PreopenInputs
    config: SplitConfig
    scoring_policy: ScoringPolicy = ScoringPolicy()
    components: Components
    prices: OHLC
    training_dates: tuple[date, ...]
    training_digest: str
    head_parameters: dict[str, dict]
    wick_method: Literal["past_mean_nonnegative_log_wicks"] = "past_mean_nonnegative_log_wicks"
    snapshot_hash: str | None = None

    @model_validator(mode="after")
    def consistent_snapshot(self):
        if (
            len(self.training_dates) < self.config.min_training_samples
            or tuple(sorted(set(self.training_dates))) != self.training_dates
            or any(d >= self.inputs.trading_date for d in self.training_dates)
        ):
            raise ValueError("training dates must be unique, ordered and strictly in the past")
        rebuilt = self.components.reconstruct(self.inputs.previous_close)
        if any(
            not isclose(getattr(rebuilt, k), getattr(self.prices, k), rel_tol=1e-10)
            for k in OHLC.model_fields
        ):
            raise ValueError("prices disagree with g/b/u/d components")
        return self


def _vector(inputs: PreopenInputs, names: tuple[str, ...]) -> list[float]:
    values = {f.name: f.value for f in inputs.features}
    missing = set(names) - set(values)
    if missing:
        raise ValueError(f"missing features: {sorted(missing)}")
    return [values[name] for name in names]


def fit_ridge_head(x, y, query, alpha: float) -> tuple[float, dict]:
    """Shared numerical head. Callers enforce their own temporal/data contract.

    The live caller validates PreopenInputs; retrospective research explicitly
    records availability assumptions instead of forging historical timestamps.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, y)
    prediction = float(model.predict([query])[0])
    scaler, regressor = model.steps[0][1], model.steps[1][1]
    return prediction, {
        "alpha": alpha,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": regressor.coef_.tolist(),
        "intercept": float(regressor.intercept_),
    }


def forecast_split(
    examples: list[TrainingExample],
    inputs: PreopenInputs,
    config: SplitConfig | None = None,
    policy: ScoringPolicy | None = None,
) -> SplitForecast:
    config = config or SplitConfig()
    if len(examples) < config.min_training_samples:
        raise ValueError(f"need at least {config.min_training_samples} training samples")
    ordered = sorted(examples, key=lambda e: e.inputs.trading_date)
    dates = tuple(e.inputs.trading_date for e in ordered)
    if len(set(dates)) != len(dates):
        raise ValueError("duplicate training trading dates")
    for example in ordered:
        if example.inputs.trading_date >= inputs.trading_date:
            raise ValueError("training dates must precede forecast trading date")
        if example.actual.available_at > inputs.data_cutoff_at:
            raise ValueError("training outcome was not yet available at forecast cutoff")
        if example.inputs.market != inputs.market:
            raise ValueError("fit each market independently")
        if example.inputs.feature_version != inputs.feature_version:
            raise ValueError("training and prediction feature versions must match")
    components = [encode_components(e.inputs.previous_close, e.actual) for e in ordered]
    heads = {}
    predictions = {}
    for name, names, alpha in (
        ("g", config.gap_features, config.gap_alpha),
        ("b", config.body_features, config.body_alpha),
    ):
        # Separate standardizers and estimators; neither head sees today's O/H/L/C.
        x = [_vector(e.inputs, names) for e in ordered]
        predictions[name], parameters = fit_ridge_head(
            x, [getattr(c, name) for c in components], _vector(inputs, names), alpha
        )
        heads[name] = {"features": list(names), **parameters}
    result = Components(
        **predictions,
        u=float(np.mean([c.u for c in components])),
        d=float(np.mean([c.d for c in components])),
    )
    training_payload = {"examples": [e.model_dump(mode="json") for e in ordered]}
    return SplitForecast(
        inputs=inputs,
        config=config,
        scoring_policy=policy or ScoringPolicy(),
        model_version=config.version,
        components=result,
        prices=result.reconstruct(inputs.previous_close),
        training_dates=dates,
        training_digest=document_hash(training_payload),
        head_parameters=heads,
    )


def evaluate_split(
    forecast: SplitForecast, actual: ObservedBar, policy: ScoringPolicy | None = None
) -> dict:
    policy = policy or forecast.scoring_policy
    if policy != forecast.scoring_policy:
        raise ValueError("evaluation policy must match the policy frozen with the forecast")
    for field in ("trading_date", "market", "contract"):
        if getattr(forecast.inputs, field) != getattr(actual, field):
            raise ValueError(f"forecast and actual {field} do not match")
    observed = encode_components(forecast.inputs.previous_close, actual)
    result = {}
    for name, predicted, realized in (
        ("open_gap", forecast.components.g, observed.g),
        ("session_body", forecast.components.b, observed.b),
        ("close_return", forecast.components.g + forecast.components.b, observed.g + observed.b),
    ):
        pred_bps, real_bps = expm1(predicted) * 10_000, expm1(realized) * 10_000
        result[f"{name}_predicted_bps"] = pred_bps
        result[f"{name}_realized_bps"] = real_bps
        result[f"{name}_mae_bps"] = abs(pred_bps - real_bps)
        result[f"{name}_direction_hit"] = int(
            realized_direction(pred_bps, policy.flat_threshold_bps)
            == realized_direction(real_bps, policy.flat_threshold_bps)
        )
    for name in OHLC.model_fields:
        error = abs(getattr(forecast.prices, name) - getattr(actual, name))
        result[f"{name}_absolute_error_points"] = error
        result[f"{name}_absolute_error_bps"] = error / forecast.inputs.previous_close * 10_000
    return result
