"""One scoring contract for the CLI, audits and model comparisons."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScoringPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False, extra="forbid")
    version: str = "1.0"
    flat_threshold_bps: float = Field(default=5, ge=0)
    bull_close_return_bps: float = 50
    bear_close_return_bps: float = -50

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.bear_close_return_bps >= self.bull_close_return_bps:
            raise ValueError("bear threshold must be below bull threshold")
        return self


def load_scoring_policy(path: str | Path) -> ScoringPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ScoringPolicy(
        version=str(data["version"]),
        flat_threshold_bps=data["flat_threshold_bps"],
        **data["scenario_thresholds"],
    )
