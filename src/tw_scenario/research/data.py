"""Immutable research datasets and explicitly retrospective sample generation."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

FIELDS = ("open", "high", "low", "close")


class Bar(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    market: str
    contract: str | None = None
    trading_date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.market not in ("TAIEX", "TX"):
            raise ValueError("unrecognized market")
        if (self.market == "TX") != bool(self.contract):
            raise ValueError("explicit TX contract required; no contract for TAIEX")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("incoherent OHLC")
        return self


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def load_dataset(path: Path) -> tuple[dict, list[Bar]]:
    raw = json.loads(path.read_text())
    checksum = raw["checksum"]
    if canonical_hash({k: v for k, v in raw.items() if k != "checksum"}) != checksum:
        raise ValueError("dataset checksum mismatch")
    bars = [Bar.model_validate(row) for row in raw["bars"]]
    if not bars or [b.trading_date for b in bars] != sorted({b.trading_date for b in bars}):
        raise ValueError("empty, unsorted or duplicate market bars")
    if len({(b.market, b.contract) for b in bars}) != 1:
        raise ValueError("mixed contracts/markets")
    return raw, bars


def transform(bars: list[Bar], representation: str) -> np.ndarray:
    rows = []
    for previous, current in pairwise(bars):
        if (previous.market, previous.contract) != (current.market, current.contract):
            raise ValueError("cross-contract returns forbidden")
        p = previous.close
        if representation == "relative":
            rows.append([getattr(current, f) / p - 1 for f in FIELDS])
        elif representation == "gbud":
            o, h, l, c = current.open, current.high, current.low, current.close
            rows.append(
                [np.log(o / p), np.log(c / o), np.log(h / max(o, c)), np.log(min(o, c) / l)]
            )
        else:
            raise ValueError("unsupported representation")
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite transformed targets")
    return result


def reconstruct(values: np.ndarray, previous: np.ndarray, representation: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError("invalid four-channel prediction")
    if representation == "relative":
        prices = previous[:, None] * (1 + values)
        o, h, l, c = prices.T
        h = np.maximum.reduce([h, o, c])
        l = np.minimum.reduce([l, o, c])
    elif representation == "gbud":
        g, b, u, d = values.T
        o = previous * np.exp(g)
        c = o * np.exp(b)
        h = np.maximum(o, c) * np.exp(np.maximum(u, 0))
        l = np.minimum(o, c) * np.exp(-np.maximum(d, 0))
    else:
        raise ValueError("unsupported representation")
    result = np.stack([o, h, l, c], axis=1)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("invalid reconstructed prices")
    return result


def indices(bars: list[Bar], context: int, start: str, end: str) -> list[int]:
    # i indexes transformed target i, which corresponds to bar i+1.
    # Context [i-context:i] ends at bar i; target bar i+1 never enters inputs.
    return [
        i for i in range(context, len(bars) - 1) if start <= str(bars[i + 1].trading_date) <= end
    ]


def sample_arrays(bars: list[Bar], representation: str, context: int, selected: list[int]):
    values = transform(bars, representation)
    x = np.stack([values[i - context : i].T for i in selected])
    y = values[selected]
    previous = np.array([bars[i].close for i in selected])
    actual = np.array([[getattr(bars[i + 1], f) for f in FIELDS] for i in selected])
    return x, y, previous, actual


def metrics(predicted: np.ndarray, actual: np.ndarray, previous: np.ndarray) -> dict:
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("mismatched OHLC evaluation")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite evaluation")
    error = predicted - actual
    result = {}
    for col, field in enumerate(FIELDS):
        result[field] = {
            "mae_points": float(np.abs(error[:, col]).mean()),
            "rmse_points": float(np.sqrt(np.mean(error[:, col] ** 2))),
            "mae_bps": float(np.mean(np.abs(error[:, col]) / previous * 10000)),
        }
    result["n"] = len(actual)
    return result
