from __future__ import annotations

import json
import time
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from tw_scenario.quant.contracts import Prices
from tw_scenario.quant.timesfm_head_training import (
    HeadSample,
    HeadTrainingSpec,
    _batch_inputs,
    _decode_quantile_block,
    _persist_sample_audit,
    _spec_from_config,
    audited_head_samples,
    build_joint_head_samples,
    development_acceptance,
    expanding_folds,
    joint_development_acceptance,
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
    inputs, cpm_mask, patches, target_count = _batch_inputs(
        model, [sample], np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    )
    assert patches == 1
    assert target_count == 4
    assert inputs["values"].shape == (1, 6, 3, 32)
    assert inputs["masks"][:, :4, 1:].all()
    assert not inputs["masks"][:, 4:, 1, 0].any()
    assert inputs["masks"][:, 4:, 1, 1:].all()
    assert inputs["patch_is_target"][:, :4].all()
    assert not inputs["patch_is_target"][:, 4:].any()
    assert not cpm_mask[:, :1].any()
    assert cpm_mask[:, 1:].all()


def test_joint_samples_align_markets_and_keep_separate_references():
    class Data:
        misalign = False

        def research_history(self, instrument, day, representation, context, window):
            del day, representation, window
            dates = [date(2025, 12, 30), date(2025, 12, 31)]
            if instrument == "TX" and self.misalign:
                dates[-1] = date(2026, 1, 1)
            offset = 0.0 if instrument == "TAIEX" else 1.0
            return (
                np.full((context, 4), offset, dtype=np.float32),
                np.full((context, 2), offset, dtype=np.float32),
                dates,
            )

        def reference(self, instrument, day):
            del day
            return SimpleNamespace(close=100.0 if instrument == "TAIEX" else 200.0)

        def actual(self, instrument, day):
            del day
            close = 100.0 if instrument == "TAIEX" else 200.0
            return Prices(open=close, high=close, low=close, close=close)

        def research_features(self, instrument, day):
            del day
            return np.array([0.0, 1.0], dtype=np.float32) + (
                0.0 if instrument == "TAIEX" else 2.0
            )

        def contract(self, instrument, day):
            del day
            return "" if instrument == "TAIEX" else "202601"

    data = Data()
    sample = build_joint_head_samples(data, [date(2026, 1, 2)], 2)[0]
    assert sample.history.shape == (2, 8)
    assert sample.past_features.shape == (2, 4)
    assert sample.forecast_features.shape == (4,)
    assert sample.target.shape == (8,)
    assert sample.previous_close == (100.0, 200.0)
    assert sample.contract == {"TAIEX": "", "TX": "202601"}

    data.misalign = True
    with pytest.raises(ValueError, match="not aligned"):
        build_joint_head_samples(data, [date(2026, 1, 2)], 2)


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


def test_joint_training_inputs_and_close_loss_cover_both_markets():
    torch = pytest.importorskip("torch")
    model = type("Model", (), {"input_patch_len": 32, "output_patch_len": 64})()
    base = _sample(date(2026, 1, 2))
    sample = HeadSample(
        day=base.day,
        history=np.column_stack([base.history, base.history + 1]),
        past_features=np.column_stack([base.past_features, base.past_features + 1]),
        forecast_features=np.concatenate([base.forecast_features, base.forecast_features + 1]),
        target=np.zeros(8, dtype=np.float32),
        previous_close=(100.0, 200.0),
        actual={"TAIEX": base.actual, "TX": base.actual},
        contract={"TAIEX": "", "TX": "202601"},
    )
    inputs, _, _, target_count = _batch_inputs(
        model, [sample], np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    )
    assert target_count == 8
    assert inputs["values"].shape == (1, 12, 3, 32)
    assert inputs["masks"][:, :8, 1:].all()
    assert not inputs["masks"][:, 8:, 1, 0].any()

    quantiles = [0.1, 0.5, 0.9]
    spec = HeadTrainingSpec(context=32, learning_rate=1e-3, weight_decay=0)
    good = torch.zeros((1, 8, 3))
    bad_second_close = good.clone()
    bad_second_close[:, 4:6, :] = 2.0
    assert output_head_loss(good, torch.zeros((1, 8)), quantiles, spec) < output_head_loss(
        bad_second_close, torch.zeros((1, 8)), quantiles, spec
    )


def test_joint_decoding_uses_separate_reference_prices():
    transformed = np.zeros((4, 3), dtype=np.float32)
    taiex, _, _ = _decode_quantile_block(transformed, 100.0)
    tx, _, _ = _decode_quantile_block(transformed, 200.0)
    assert taiex.close == pytest.approx(100.0)
    assert tx.close == pytest.approx(200.0)
    assert taiex.high >= max(taiex.open, taiex.close)
    assert tx.low <= min(tx.open, tx.close)


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


def test_joint_gate_fails_if_either_market_fails():
    passed = {"status": "PASS", "close_relative_improvement": 0.10}
    failed = {"status": "FAIL", "close_relative_improvement": 0.02}
    gate = joint_development_acceptance({"TAIEX": passed, "TX": failed})
    assert gate["status"] == "FAIL"
    assert gate["close_relative_improvement"] == pytest.approx(0.02)
    assert not gate["requirements"]["both_markets_pass"]


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


def test_candidate_selection_accepts_joint_worst_market_score():
    result = {
        "status": "DEVELOPMENT_GATE_FAIL",
        "identity": {
            "dataset_id": "dataset",
            "instrument": "JOINT",
            "config_hash": "config",
            "candidate_index": 0,
        },
        "development_gate": {
            "status": "FAIL",
            "close_relative_improvement": 0.03,
            "instrument_close_relative_improvement": {"TAIEX": 0.08, "TX": 0.03},
        },
    }
    selection = select_output_head_candidate([result], 3)
    assert selection["instrument"] == "JOINT"
    assert selection["selected"]["close_relative_improvement"] == pytest.approx(0.03)
    assert selection["status"] == "DEVELOPMENT_SEARCH_STOPPED_NO_PASS"


def test_training_materialization_audits_history_and_propagates_unknown_error(monkeypatch):
    import tw_scenario.quant.timesfm_head_training as module

    days = [date(2026, 1, 1), date(2026, 1, 2)]

    def builder(data, instrument, requested, context):
        if requested[0] == days[0]:
            raise ValueError('insufficient contiguous expanded history: 12 < 32')
        return [_sample(requested[0])]

    monkeypatch.setattr(module, 'build_head_samples', builder)
    samples, audit = module.materialize_training_samples(None, 'TAIEX', days, 32)
    assert [sample.day for sample in samples] == days[1:]
    assert audit['requested'] == 2
    assert audit['ineligible'] == 1
    assert audit['dates'][0]['status'] == 'INELIGIBLE_HISTORY'

    def broken(*args):
        raise ValueError('unexpected shape bug')

    monkeypatch.setattr(module, 'build_head_samples', broken)
    with pytest.raises(ValueError, match='unexpected shape bug'):
        module.materialize_training_samples(None, 'TAIEX', days, 32)


def test_candidate_selection_does_not_promote_close_only_winner():
    def candidate(index, improvement, gate):
        return {
            "identity": {
                "dataset_id": "dataset",
                "instrument": "TAIEX",
                "config_hash": "config",
                "candidate_index": index,
            },
            "development_gate": {
                "status": gate,
                "close_relative_improvement": improvement,
            },
        }

    selection = select_output_head_candidate(
        [candidate(0, 0.20, "FAIL"), candidate(1, 0.08, "PASS")], 3
    )
    assert selection["selected"]["candidate_index"] == 1
    assert selection["best_close_only_research_candidate"]["candidate_index"] == 0
    assert selection["status"] == "LOCKED_AWAITING_NEW_FORWARD_HOLDOUT"
    assert selection["production_candidate"] is False


def test_paired_validation_preserves_failures_and_uses_identical_dates():
    from tw_scenario.quant.timesfm_head_training import paired_validation_metrics

    candidate = _rows(8, 2)
    baseline = _rows(10, 2)
    candidate[1] = {"date": candidate[1]["date"], "status": "FAILED", "error": "missing"}
    baseline[1]["predicted"]["close"] = 1000
    comparison = paired_validation_metrics(candidate, baseline, 5)
    assert comparison["close_relative_improvement"] == pytest.approx(0.2)
    assert comparison["paired_coverage"] == 0.5
    assert comparison["candidate_all_dates_metrics"]["failed_days"] == [
        {"date": candidate[1]["date"], "reason": "missing"}
    ]
    with pytest.raises(ValueError, match="duplicate"):
        paired_validation_metrics(candidate + candidate[:1], baseline, 5)
    baseline[0]["actual"]["close"] = 101
    with pytest.raises(ValueError, match="truth/reference mismatch"):
        paired_validation_metrics(candidate, baseline, 5)


def test_all_failed_validation_is_a_persistable_rejection():
    import json

    config = {"neutral_bps": 5}
    failed = [{"date": "2025-01-01", "status": "FAILED", "error": "inference"}]
    gate = development_acceptance(failed, _rows(10, 1), config, fold_improvements=[None])
    assert gate["status"] == "FAIL_NO_PAIRED_DAYS"
    assert gate["close_relative_improvement"] is None
    joint = joint_development_acceptance({"TAIEX": gate, "TX": gate})
    assert joint["status"] == "FAIL"
    selection = select_output_head_candidate([{
        "identity": {"dataset_id": "d", "instrument": "JOINT", "config_hash": "c",
                     "candidate_index": 0},
        "development_gate": joint,
    }], 3)
    assert selection["status"] == "DEVELOPMENT_SEARCH_STOPPED_NO_PASS"
    json.dumps(selection, allow_nan=False)


def test_zero_error_baseline_fails_comparison_without_division_by_zero():
    gate = development_acceptance(_rows(1, 2), _rows(0, 2), {"neutral_bps": 5},
                                  fold_improvements=[None])
    assert gate["status"] == "FAIL_ZERO_BASELINE_ERROR"


@pytest.mark.parametrize("instrument", ["TAIEX", "TX", "JOINT"])
def test_bad_training_days_remain_in_audit_without_compressing_context(monkeypatch, instrument):
    import tw_scenario.quant.timesfm_head_training as training

    days = [date(2020, 1, day) for day in (2, 3, 6, 7)]
    calls = []

    def build(data, requested, context):
        calls.append((requested, context))
        if requested[0] in days[:2]:
            raise ValueError("insufficient contiguous expanded history: 12 < 256")
        return [_sample(requested[0])]

    monkeypatch.setattr(training, "build_joint_head_samples", build)
    monkeypatch.setattr(
        training, "build_head_samples",
        lambda data, market, requested, context: build(data, requested, context),
    )
    samples, audit = audited_head_samples(None, instrument, days, 256)
    assert [sample.day for sample in samples] == days[2:]
    assert (audit["requested"], audit["usable"], audit["failed"]) == (4, 2, 2)
    assert [row["date"] for row in audit["rows"]] == list(map(str, days))
    assert [row["status"] for row in audit["rows"]] == ["INELIGIBLE_HISTORY", "INELIGIBLE_HISTORY", "OK", "OK"]
    assert audit["rows"][0]["reason"] == "insufficient contiguous expanded history: 12 < 256"
    assert calls == [([day], 256) for day in days]


def test_unexpected_sample_code_error_is_not_silently_excluded(monkeypatch):
    import tw_scenario.quant.timesfm_head_training as training

    def broken(*args):
        raise TypeError("unexpected sample schema")

    monkeypatch.setattr(training, "build_joint_head_samples", broken)
    with pytest.raises(TypeError, match="unexpected sample schema"):
        audited_head_samples(None, "JOINT", [date(2020, 1, 2)], 256)


def test_sample_audit_is_immutable_and_empty_split_failure_is_persisted(tmp_path):
    identity = {"instrument": "JOINT", "fold": 1}
    good = {"requested": 60, "usable": 60, "failed": 0,
            "rows": [{"date": str(date(2020, 1, 2) + timedelta(days=i)),
                      "status": "OK"} for i in range(60)]}
    path = tmp_path / "samples.json"
    first = _persist_sample_audit(path, identity, good, good)
    raw = path.read_bytes()
    assert _persist_sample_audit(path, identity, good, good) == first
    assert path.read_bytes() == raw
    with pytest.raises(ValueError, match="changed during resume"):
        _persist_sample_audit(path, {**identity, "fold": 2}, good, good)
    assert path.read_bytes() == raw

    bad = {"requested": 1, "usable": 0, "failed": 1,
           "rows": [{"date": "2020-01-02", "status": "FAILED", "error": "gap"}]}
    empty_path = tmp_path / "empty.json"
    with pytest.raises(ValueError, match="fewer than 60 training samples"):
        _persist_sample_audit(empty_path, identity, bad, good)
    assert json.loads(empty_path.read_text())["training"]["rows"] == bad["rows"]
    assert json.loads(empty_path.read_text())["validation_dates_filtered"] is False
