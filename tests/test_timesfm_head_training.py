from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pytest

from tw_scenario.quant.timesfm_head_training import (
    HeadSample,
    HeadTrainingSpec,
    _batch_inputs,
    _spec_from_config,
    development_acceptance,
    expanding_folds,
    output_head_loss,
    select_output_head_candidate,
    train_output_head,
)


def _sample(day: date, target: float = 0.1) -> HeadSample:
    return HeadSample(
        day=day,
        history=np.full((32, 4), target, dtype=np.float32),
        past_features=np.column_stack([
            np.linspace(0, 1, 32, dtype=np.float32),
            np.linspace(1, 2, 32, dtype=np.float32),
        ]),
        forecast_features=np.array([1.1, 2.1], dtype=np.float32),
        target=np.full(4, target, dtype=np.float32),
        previous_close=100.0,
        actual={field: 100.0 for field in ("open", "high", "low", "close")},
        contract="",
    )


def test_expanding_folds_are_chronological_and_disjoint():
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(40)]
    folds = expanding_folds(days, folds=3, validation_sessions=5, minimum_train=20)
    assert [len(validation) for _, validation in folds] == [5, 5, 5]
    assert [len(train) for train, _ in folds] == [25, 30, 35]
    assert all(train[-1] < validation[0] for train, validation in folds)
    with pytest.raises(ValueError, match="insufficient sessions"):
        expanding_folds(days[:30], folds=3, validation_sessions=5, minimum_train=20)


def test_training_grid_selection_is_explicit_and_bounded():
    config = {
        "training_grid": [
            {"context": 256, "learning_rate": 1e-4, "weight_decay": 0},
            {"context": 512, "learning_rate": 3e-4, "weight_decay": 1e-4},
        ],
        "max_epochs": 5,
        "patience": 2,
        "batch_size": 1,
        "point_weight": 1,
        "close_weight": 2,
        "quantile_weight": 0.5,
        "crossing_weight": 0.1,
    }
    assert _spec_from_config(config, 1).context == 512
    with pytest.raises(ValueError, match="candidate_index"):
        _spec_from_config(config, 2)


def test_training_inputs_keep_future_features_known_but_targets_masked():
    model = type("Model", (), {"input_patch_len": 32, "output_patch_len": 64})()
    sample = _sample(date(2026, 1, 2))
    inputs, cpm_mask, patches = _batch_inputs(
        model, [sample], np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    )
    assert patches == 1
    assert inputs["values"].shape == (1, 6, 3, 32)
    assert inputs["masks"][:, :4, 1:].all()
    assert not inputs["masks"][:, 4:, 1, 0].any()
    assert inputs["masks"][:, 4:, 1, 1:].all()
    assert inputs["patch_is_target"][:, :4].all()
    assert not inputs["patch_is_target"][:, 4:].any()
    assert not cpm_mask[:, :1].any()
    assert cpm_mask[:, 1:].all()


def test_output_head_loss_penalizes_bad_point_and_crossing():
    torch = pytest.importorskip("torch")
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    target = torch.zeros((2, 4))
    good = torch.linspace(-0.2, 0.2, 9).repeat(2, 4, 1)
    bad = torch.flip(good, dims=(2,)) + 2.0
    spec = HeadTrainingSpec(context=32, learning_rate=1e-3, weight_decay=0)
    assert output_head_loss(good, target, quantiles, spec) < output_head_loss(
        bad, target, quantiles, spec
    )


def test_training_freezes_backbone_and_updates_only_output_head():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        input_patch_len = 32
        output_patch_len = 64
        quantiles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Parameter(torch.tensor(1.0))
            self.output_head = torch.nn.Linear(1, 9)

        def forward(self, inputs, patch_cpm_mask):
            del patch_cpm_mask
            base = inputs["values"][:, :, :, :1] * self.backbone
            return {"logits": self.output_head(base).unsqueeze(-2)}

    model = TinyModel()
    original_backbone = model.backbone.detach().clone()
    original_head = model.output_head.weight.detach().clone()
    samples = [_sample(date(2026, 1, 1) + timedelta(days=index), index / 100) for index in range(8)]
    spec = HeadTrainingSpec(
        context=32,
        learning_rate=1e-2,
        weight_decay=0,
        max_epochs=3,
        patience=2,
        batch_size=2,
    )
    result = train_output_head(
        model,
        samples[:6],
        samples[6:],
        spec,
        seed=4655,
        deadline=time.monotonic() + 10,
    )
    assert result["epochs_completed"] >= 1
    torch.testing.assert_close(model.backbone, original_backbone)
    assert not torch.equal(model.output_head.weight, original_head)
    assert result["trainable_parameter_names"] == [
        "output_head.weight",
        "output_head.bias",
    ]


def _rows(error: float, count: int = 100):
    rows = []
    for index in range(count):
        actual = {field: 100.0 for field in ("open", "high", "low", "close")}
        predicted = {field: 100.0 + error for field in actual}
        inside = index < 80
        interval = {
            field: ([99.0, 101.0] if inside else [102.0, 104.0])
            for field in actual
        }
        rows.append({
            "date": str(date(2025, 1, 1) + timedelta(days=index)),
            "status": "OK",
            "previous_close": 100.0,
            "actual": actual,
            "predicted": predicted,
            "interval": interval,
        })
    return rows


def test_development_gate_requires_buffer_folds_uncertainty_and_intervals():
    config = {
        "neutral_bps": 5.0,
        "promotion_close_improvement": 0.075,
        "promotion_min_coverage": 0.99,
        "bootstrap_block": 20,
        "bootstrap_draws": 100,
        "seed": 4655,
    }
    passed = development_acceptance(
        _rows(8.0), _rows(10.0), config, fold_improvements=[0.2, 0.1, -0.01]
    )
    assert passed["status"] == "PASS"
    assert passed["close_relative_improvement"] == pytest.approx(0.2)
    failed = development_acceptance(
        _rows(9.5), _rows(10.0), config, fold_improvements=[0.05, -0.01, -0.02]
    )
    assert failed["status"] == "FAIL"
    assert not failed["requirements"]["close_buffer"]
    assert not failed["requirements"]["fold_majority"]


def test_candidate_selection_stops_after_three_no_gain_rounds():
    def result(index, improvement, status="FAIL"):
        return {
            "status": "DEVELOPMENT_GATE_FAIL",
            "identity": {
                "dataset_id": "dataset",
                "instrument": "TAIEX",
                "config_hash": "config",
                "candidate_index": index,
            },
            "development_gate": {
                "status": status,
                "close_relative_improvement": improvement,
            },
        }

    selection = select_output_head_candidate(
        [
            result(0, 0.04),
            result(1, 0.08, "PASS"),
            result(2, 0.07),
            result(3, 0.06),
            result(4, 0.05),
            result(5, 0.20, "PASS"),
        ],
        3,
    )
    assert selection["status"] == "LOCKED_AWAITING_NEW_FORWARD_HOLDOUT"
    assert selection["selected"]["candidate_index"] == 1
    assert selection["stopped_after_no_gain"]
    assert [item["candidate_index"] for item in selection["evaluated"]] == [0, 1, 2, 3, 4]
