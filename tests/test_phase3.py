from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from tw_scenario.quant.contracts import Prices
from tw_scenario.quant.phase3 import (
    ResidualRecord,
    RidgeState,
    _candidate_rows,
    _corrected_prices,
    _correction_fields,
    _fit_ridge,
    _validate_config,
)

TZ = ZoneInfo("Asia/Taipei")


def _config() -> dict:
    return json.loads(
        Path("config/quant/phase3_price_residual_ridge.json").read_text(encoding="utf-8")
    )


def _record(index: int, *, instrument: str = "TAIEX") -> ResidualRecord:
    day = date(2020, 1, 1) + timedelta(days=index)
    reference = 10_000.0
    anchor = Prices(open=10_010, high=10_080, low=9_940, close=10_020)
    target = np.array([index, index * 0.5, -index * 0.25, index * 0.75], dtype=float)
    actual_values = anchor.array() + reference * target / 10_000
    actual = Prices(
        open=float(actual_values[0]),
        high=float(max(actual_values[1], actual_values[0], actual_values[3])),
        low=float(min(actual_values[2], actual_values[0], actual_values[3])),
        close=float(actual_values[3]),
    )
    return ResidualRecord(
        day=day,
        features=np.array([index, index % 5, 1.0], dtype=float),
        target_bps=(actual.array() - anchor.array()) / reference * 10_000,
        anchor=anchor,
        actual=actual,
        previous_close=reference,
        contract="" if instrument == "TAIEX" else "202001",
        label_available_at=datetime.combine(day, datetime.min.time(), TZ).replace(hour=18),
        anchor_model="night_regression",
    )


def test_phase3_config_is_frozen_and_rejects_production():
    config = _config()
    _validate_config(config)
    config["production_switch"] = True
    with pytest.raises(ValueError, match="cannot enable production"):
        _validate_config(config)


def test_component_correction_fields_describe_oc_extensions():
    assert _correction_fields("actual_minus_anchor_oc_extensions_bps") == (
        "open",
        "close",
        "upper_extension",
        "lower_extension",
    )


def test_price_residual_ridge_reconstructs_positive_coherent_ohlc():
    records = [_record(index) for index in range(1, 40)]
    state = _fit_ridge(records, alpha=10.0, window=30)
    point, correction = _corrected_prices(records[-1], state, 500.0)

    assert state.training_rows == 30
    assert correction.shape == (4,)
    assert np.all(np.abs(correction) <= 500.0)
    assert point.high >= max(point.open, point.close)
    assert point.low <= min(point.open, point.close)
    assert min(point.open, point.high, point.low, point.close) > 0


def test_outer_predictions_fit_only_strictly_prior_labels(monkeypatch):
    records = [_record(index) for index in range(1, 12)]
    validation = [records[-2].day, records[-1].day]
    observed_training_ends = []

    def fake_fit(rows, *, alpha, window):
        observed_training_ends.append(rows[-1].day)
        return RidgeState(
            alpha=alpha,
            window=window,
            mean=np.zeros(3),
            scale=np.ones(3),
            coef=np.zeros((4, 3)),
            intercept=np.zeros(4),
            training_rows=len(rows),
            training_start=rows[0].day,
            training_end=rows[-1].day,
        )

    monkeypatch.setattr("tw_scenario.quant.phase3._fit_ridge", fake_fit)
    rows = _candidate_rows(
        records,
        validation,
        alpha=1.0,
        window=20,
        minimum_fit_rows=2,
        max_abs_correction_bps=500.0,
        target="actual_minus_anchor_ohlc_bps",
        instrument="TAIEX",
    )

    assert [row["status"] for row in rows] == ["OK", "OK"]
    assert all(training_end < forecast_day for training_end, forecast_day in zip(
        observed_training_ends, validation, strict=True
    ))
