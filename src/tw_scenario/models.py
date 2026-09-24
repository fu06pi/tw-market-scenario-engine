from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class Regime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    GAP_FADE = "gap_fade"
    RANGE = "range"


class ScenarioProbabilities(BaseModel):
    bull: float = Field(ge=0, le=1)
    base: float = Field(ge=0, le=1)
    bear: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self):
        total = self.bull + self.base + self.bear
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"scenario probabilities must sum to 1, got {total}")
        return self


class PredictionTargets(BaseModel):
    open_direction: Direction
    close_direction: Direction
    intraday_regime: Regime
    breadth_direction: Direction
    expected_open_gap_bps: float | None = None
    expected_close_return_bps: float | None = None


class NarrativeForecast(BaseModel):
    market_view: str
    open_hypothesis: str
    intraday_hypothesis: str
    close_hypothesis: str
    breadth_hypothesis: str
    bull_script: str
    base_script: str
    bear_script: str


class V0Reference(BaseModel):
    trading_date: str
    item: str
    note: str | None = None
    source_row: str | None = None
    extracted_at: datetime | None = None
    content_hash: str | None = None


class PredictionSnapshot(BaseModel):
    trading_date: str
    generated_at: datetime
    data_cutoff_at: datetime
    model_version: str
    feature_version: str
    prompt_version: str | None = None
    data_manifest_hash: str
    scenarios: ScenarioProbabilities
    targets: PredictionTargets
    narrative: NarrativeForecast
    key_drivers: list[str]
    invalidation_conditions: list[str]
    v0_references: list[V0Reference] = []
    snapshot_hash: str | None = None


class MarketRealization(BaseModel):
    trading_date: str
    previous_close: float
    open: float
    high: float
    low: float
    close: float
    turnover: float | None = None
    breadth_up: int | None = None
    breadth_down: int | None = None
    otc_return_pct: float | None = None
    heavyweight_contribution_points: dict[str, float] = {}

    @property
    def open_gap_bps(self) -> float:
        return (self.open / self.previous_close - 1.0) * 10_000

    @property
    def close_return_bps(self) -> float:
        return (self.close / self.previous_close - 1.0) * 10_000

    @property
    def mfe_bps(self) -> float:
        return (self.high / self.previous_close - 1.0) * 10_000

    @property
    def mae_bps(self) -> float:
        return (self.low / self.previous_close - 1.0) * 10_000
