from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tw_scenario.quant.timesfm3 import TimesFM3Adapter, gradient_smoke_contract
from tw_scenario.quant.timesfm_backtest import _acceptance, _joint_history
from tw_scenario.quant.timesfm_gradient_smoke import _training_inputs


class FakeForecaster:
    def __init__(self):
        self.calls = []

    def predict(self, context, horizon, **kwargs):
        context = np.asarray(context)
        self.calls.append((context.copy(), horizon, kwargs))
        if context.ndim == 1:
            point = np.array([context[-1] + 1])
            quantiles = np.arange(9, dtype=float)[None, :] + point[:, None]
        else:
            point = context[:, -1, None] + 1
            quantiles = np.repeat(point[..., None], 9, axis=-1)
        return SimpleNamespace(forecast=point, quantiles=quantiles)


def history():
    return np.arange(24, dtype=float).reshape(6, 4)


def test_univariate_uses_four_independent_zero_shot_calls():
    fake = FakeForecaster()
    result = TimesFM3Adapter(fake).zero_shot_univariate(history())
    assert result.mode == "zero_shot_univariate"
    assert result.point.shape == (4,)
    assert result.quantiles.shape == (4, 9)
    assert len(fake.calls) == 4
    assert all(call[2]["make_positive"] is False for call in fake.calls)


def test_multivariate_passes_known_forecast_period_covariates():
    fake = FakeForecaster()
    past = np.arange(18, dtype=float).reshape(6, 3)
    future = np.array([101.0, 102.0, 103.0])
    result = TimesFM3Adapter(fake).zero_shot_multivariate(history(), past, future)
    assert result.point.shape == (4,)
    assert result.quantiles.shape == (4, 9)
    context, horizon, kwargs = fake.calls[0]
    assert context.shape == (4, 6)
    assert horizon == 1
    assert kwargs["past_future_covariates"].shape == (3, 7)
    np.testing.assert_array_equal(kwargs["past_future_covariates"][:, -1], future)


def test_joint_multivariate_uses_eight_aligned_targets_and_known_covariates():
    fake = FakeForecaster()
    joint_history = np.arange(48, dtype=float).reshape(6, 8)
    past = np.arange(18, dtype=float).reshape(6, 3)
    future = np.array([101.0, 102.0, 103.0])
    result = TimesFM3Adapter(fake).zero_shot_joint_multivariate(
        joint_history, past, future
    )
    assert result.mode == "zero_shot_joint_multivariate"
    assert result.point.shape == (8,)
    assert result.quantiles.shape == (8, 9)
    context, horizon, kwargs = fake.calls[0]
    assert context.shape == (8, 6)
    assert horizon == 1
    assert kwargs["past_future_covariates"].shape == (3, 7)
    np.testing.assert_array_equal(kwargs["past_future_covariates"][:, -1], future)


def test_multivariate_rejects_partial_or_nonfinite_covariates():
    adapter = TimesFM3Adapter(FakeForecaster())
    with pytest.raises(ValueError, match="supplied together"):
        adapter.zero_shot_multivariate(history(), np.ones((6, 2)), None)
    bad = np.ones((6, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        adapter.zero_shot_multivariate(history(), bad, np.ones(2))


def test_joint_history_rejects_cross_market_date_misalignment():
    class MisalignedData:
        def history(self, instrument, *_args):
            dates = ["2026-01-01", "2026-01-02"]
            if instrument == "TX":
                dates[-1] = "2026-01-03"
            return np.ones((2, 4)), np.ones((2, 3)), dates

    with pytest.raises(ValueError, match="dates are not aligned"):
        _joint_history(
            MisalignedData(),
            "2026-01-04",
            "gbud",
            {"window": 252, "min_history": 2},
        )


def test_gradient_smoke_is_not_mislabelled_as_official_finetuning():
    contract = gradient_smoke_contract()
    assert contract["status"] == "NOT_RUN"
    assert contract["official_timesfm3_finetuning"] is False
    assert "nonzero_output_head_gradient" in contract["required_evidence"]


def test_gradient_smoke_inputs_use_context_then_masked_horizon_patches():
    model = SimpleNamespace(input_patch_len=32, output_patch_len=64)
    inputs, cpm_mask, context_patches = _training_inputs(
        model, np.arange(256, dtype=np.float32).reshape(64, 4)
    )
    assert inputs["values"].shape == (1, 4, 4, 32)
    assert inputs["masks"].shape == (1, 4, 4, 32)
    assert context_patches == 2
    assert not inputs["masks"][:, :, :2].any()
    assert inputs["masks"][:, :, 2:].all()
    assert not cpm_mask[:, :2].any()
    assert cpm_mask[:, 2:].all()
    with pytest.raises(ValueError, match="align to input patches"):
        _training_inputs(model, np.ones((63, 4), dtype=np.float32))


def _acceptance_rows(error: float):
    rows = []
    for index in range(50):
        actual = {name: 100.0 for name in ("open", "high", "low", "close")}
        predicted = {name: 100.0 + error for name in actual}
        interval = {
            name: ([99.0, 101.0] if index < 40 else [102.0, 104.0])
            for name in actual
        }
        rows.append({
            "date": f"2026-01-{index + 1:02d}",
            "status": "OK",
            "previous_close": 100.0,
            "actual": actual,
            "predicted": predicted,
            "interval": interval,
        })
    return rows


def test_timesfm_acceptance_uses_paired_rows_and_all_fixed_gates():
    config = {
        "neutral_bps": 5.0,
        "min_validation_coverage": 0.95,
        "bootstrap_block": 20,
        "bootstrap_draws": 20,
        "seed": 4655,
    }
    result = _acceptance(_acceptance_rows(9.0), _acceptance_rows(10.0), config)
    assert result["status"] == "PASS"
    assert result["paired_days"] == 50
    assert result["close_relative_improvement"] == pytest.approx(0.1)
    assert result["interval_coverage"] == {
        "open": 0.8,
        "high": 0.8,
        "low": 0.8,
        "close": 0.8,
    }

    failed = _acceptance(_acceptance_rows(10.0), _acceptance_rows(10.0), config)
    assert failed["status"] == "FAIL"
