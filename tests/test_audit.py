import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tw_scenario.audit import assess_eligibility, audit_repository, git_persistence, write_audit
from tw_scenario.cli import app
from tw_scenario.metrics import evaluate_snapshot
from tw_scenario.models import MarketRealization, PredictionSnapshot
from tw_scenario.policy import ScoringPolicy, load_scoring_policy
from tw_scenario.snapshot import (
    document_hash,
    freeze_document,
    freeze_snapshot,
    verify_document,
)
from tw_scenario.split_forecast import forecast_split


@pytest.fixture
def legacy_payload(tmp_path):
    snapshot = PredictionSnapshot.model_validate_json(
        Path("examples/sample_prediction.json").read_text()
    )
    target = tmp_path / "frozen.json"
    freeze_snapshot(snapshot, target)
    return json.loads(target.read_text())


def evidence(payload):
    day = payload["trading_date"]
    return {"recorded_at": f"{day}T08:56:00+08:00", "content_changed": False}


def gate(payload, manifest=None, evaluation=None, persistence=None, has_actual=True):
    return assess_eligibility(
        payload,
        verify_document(payload).to_dict(),
        has_actual=has_actual,
        manifest=manifest or {},
        evaluation=evaluation or {},
        persistence=persistence,
    )


def test_raw_integer_legacy_hash_and_tamper_detection(legacy_payload):
    payload = deepcopy(legacy_payload)
    payload["targets"]["expected_open_gap_bps"] = 120
    payload["snapshot_hash"] = document_hash(payload)
    verified = verify_document(payload)
    assert verified.valid and verified.raw_match
    assert verified.method == "legacy_raw"
    payload["targets"]["expected_open_gap_bps"] = 121
    assert not verify_document(payload).valid


@pytest.mark.parametrize("nested", [False, True])
def test_normalization_must_not_hide_unsigned_fields(legacy_payload, nested):
    payload = deepcopy(legacy_payload)
    destination = payload["targets"] if nested else payload
    destination["unsigned_extra"] = "not covered by the old model hash"
    assert not verify_document(payload).valid


def test_legacy_normalized_default_compatibility(legacy_payload):
    payload = deepcopy(legacy_payload)
    payload["v0_references"] = []
    snapshot = PredictionSnapshot.model_validate(payload)
    from tw_scenario.snapshot import compute_snapshot_hash

    payload["snapshot_hash"] = compute_snapshot_hash(snapshot)
    del payload["v0_references"]
    assert verify_document(payload).valid
    assert verify_document(payload).method == "legacy_normalized"


def test_v2_version_and_hash_roundtrip_and_exclusive_create(tmp_path, split_data):
    payload = forecast_split(*split_data).model_dump(mode="json")
    path = tmp_path / "v2.json"
    freeze_document(payload, path)
    persisted = json.loads(path.read_text())
    assert verify_document(persisted).valid
    with pytest.raises(FileExistsError):
        freeze_document(payload, path)
    persisted["hash_version"] = "unknown"
    assert not verify_document(persisted).valid
    persisted.pop("hash_version")
    assert not verify_document(persisted).valid


def test_legacy_freeze_is_exclusive(tmp_path, legacy_payload):
    target = tmp_path / "existing.json"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        freeze_snapshot(PredictionSnapshot.model_validate(legacy_payload), target)
    assert target.read_text() == "original"


def test_false_flag_wins_over_true_flag_and_conflict_is_reported(legacy_payload):
    result = gate(
        legacy_payload,
        {"aggregate_metric_eligibility": True},
        {"aggregate_metric_eligibility": False},
        evidence(legacy_payload),
    )
    assert not result["eligible"]
    assert "evaluation_explicit_exclusion" in result["reasons"]
    assert "conflicting_historical_inclusion_flags" in result["warnings"]


@pytest.mark.parametrize("flag", ["reconstructed", "late_generated"])
def test_reconstruction_cannot_be_promoted_by_a_true_flag(legacy_payload, flag):
    result = gate(
        legacy_payload,
        {"aggregate_metric_eligibility": True, flag: True},
        persistence=evidence(legacy_payload),
    )
    assert not result["eligible"]


def test_hash_and_generation_time_do_not_prove_persistence(legacy_payload):
    result = gate(legacy_payload)
    assert not result["eligible"]
    assert "missing_persistence_evidence" in result["reasons"]


@pytest.mark.parametrize("timestamp", ["09:00:00", "09:01:00"])
def test_late_legacy_persistence_excluded(legacy_payload, timestamp):
    proof = evidence(legacy_payload)
    proof["recorded_at"] = f"{legacy_payload['trading_date']}T{timestamp}+08:00"
    result = gate(legacy_payload, persistence=proof)
    assert "persistence_contract_failed" in result["reasons"]


def test_generation_cannot_follow_claimed_persistence(legacy_payload):
    proof = evidence(legacy_payload)
    proof["recorded_at"] = f"{legacy_payload['trading_date']}T08:50:00+08:00"
    result = gate(legacy_payload, persistence=proof)
    assert "generation_after_recorded_persistence" in result["reasons"]


def test_missing_result_and_changed_record_are_visible(legacy_payload):
    proof = evidence(legacy_payload) | {"content_changed": True}
    result = gate(legacy_payload, persistence=proof, has_actual=False)
    assert {"missing_actual", "historical_prediction_changed"} <= set(result["reasons"])


def test_v2_uses_its_own_early_deadline_and_cohort(tmp_path, split_data):
    path = tmp_path / "v2.json"
    freeze_document(forecast_split(*split_data).model_dump(mode="json"), path)
    payload = json.loads(path.read_text())
    proof = {"recorded_at": "2026-10-01T08:39:59+08:00"}
    good = gate(payload, persistence=proof)
    assert good["eligible"] and good["cohort"] == "split_v2_preopen"
    assert good["new_ohlc_acceptance_proven"] is False
    bad = gate(payload, persistence={"recorded_at": "2026-10-01T08:40:00+08:00"})
    assert "persistence_contract_failed" in bad["reasons"]


def test_config_controls_direction_and_scenario_scoring(legacy_payload):
    payload = deepcopy(legacy_payload)
    payload["targets"].update(open_direction="flat", close_direction="flat")
    prediction = PredictionSnapshot.model_validate(payload)
    actual = MarketRealization(
        trading_date=payload["trading_date"],
        previous_close=100,
        open=100.06,
        high=101,
        low=99,
        close=100.06,
    )
    assert evaluate_snapshot(prediction, actual)["close_direction_hit"] == 0
    assert (
        evaluate_snapshot(prediction, actual, ScoringPolicy(flat_threshold_bps=10))[
            "close_direction_hit"
        ]
        == 1
    )
    assert load_scoring_policy("config/targets.yaml") == ScoringPolicy()
    with pytest.raises(ValueError):
        ScoringPolicy(flat_threshold_bps=float("nan"))


def init_git(root, recorded_at):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env=os.environ | {"GIT_AUTHOR_DATE": recorded_at, "GIT_COMMITTER_DATE": recorded_at},
        )

    git("init", "-b", "main")
    git("config", "user.email", "synthetic@example.invalid")
    git("config", "user.name", "Synthetic Test")
    git("add", ".")
    git("commit", "-m", "synthetic fixture")
    return git


def test_git_evidence_detects_working_tree_edit_and_reverted_history(tmp_path):
    path = tmp_path / "prediction.json"
    path.write_text("original")
    git = init_git(tmp_path, "2026-10-01T08:35:00+08:00")
    assert git_persistence(tmp_path, path)["content_changed"] is False
    path.write_text("changed")
    assert git_persistence(tmp_path, path)["content_changed"] is True
    git("add", ".")
    git("commit", "-m", "change")
    path.write_text("original")
    git("add", ".")
    git("commit", "-m", "revert")
    assert git_persistence(tmp_path, path)["content_changed"] is True


def test_audit_keeps_missing_and_invalid_records_and_never_changes_history(
    tmp_path, legacy_payload
):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    path = predictions / f"{legacy_payload['trading_date']}.json"
    path.write_text(json.dumps(legacy_payload))
    bad = predictions / "invalid.json"
    bad.write_text("{broken")
    (predictions / "array.json").write_text("[]")
    original = path.read_bytes()
    init_git(tmp_path, evidence(legacy_payload)["recorded_at"])
    report = audit_repository(tmp_path)
    assert report["prediction_count"] == 3
    assert report["eligible_count"] == 0
    assert any("missing_actual" in r["eligibility"]["reasons"] for r in report["rows"])
    assert any("invalid_record" in r["eligibility"]["reasons"] for r in report["rows"])
    assert report["aggregates"][0]["metrics"]["close_direction_hit"] == {"mean": None, "n": 0}
    output = tmp_path / "audits" / "new.json"
    write_audit(report, tmp_path, output)
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_audit(report, tmp_path, output)
    with pytest.raises(ValueError):
        write_audit(report, tmp_path, predictions / "audit.json")


def test_cli_verify_failure_is_nonzero_and_scores_are_diagnostic(tmp_path, legacy_payload):
    runner = CliRunner()
    path = tmp_path / "prediction.json"
    path.write_text(json.dumps(legacy_payload | {"snapshot_hash": "invalid"}))
    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == 1 and "FAILED" in result.stdout
    actual = tmp_path / "actual.json"
    actual.write_text(
        json.dumps(
            {
                "trading_date": legacy_payload["trading_date"],
                "previous_close": 100,
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
            }
        )
    )
    result = runner.invoke(app, ["evaluate", str(path), str(actual)])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["performance_scope"] == "diagnostic_only"
    assert json.loads(result.stdout)["integrity"]["valid"] is False


def test_invalid_actual_remains_in_its_model_denominator(tmp_path, legacy_payload):
    day = legacy_payload["trading_date"]
    for name in ("predictions", "realizations"):
        (tmp_path / name).mkdir()
    (tmp_path / "predictions" / f"{day}.json").write_text(json.dumps(legacy_payload))
    (tmp_path / "realizations" / f"{day}.json").write_text(
        json.dumps(
            {
                "trading_date": day,
                "previous_close": 100,
                "open": 100,
                "high": 99,
                "low": 101,
                "close": 100,
            }
        )
    )
    report = audit_repository(tmp_path)
    assert report["prediction_count"] == 1
    assert report["aggregates"][0]["observed_forecast_count"] == 1
    assert report["aggregates"][0]["excluded_count"] == 1
    assert report["aggregates"][0]["eligible_count"] == 0


def test_cli_forecast_verify_evaluate_roundtrip(tmp_path, split_data):
    from conftest import bar_for

    runner = CliRunner()
    examples, request, config = split_data
    for name, payload in {
        "training": [e.model_dump(mode="json") for e in examples],
        "request": request.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "actual": bar_for(request).model_dump(mode="json"),
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    output = tmp_path / "forecast.json"
    args = [
        "forecast-split",
        str(tmp_path / "training.json"),
        str(tmp_path / "request.json"),
        str(output),
        "--config",
        str(tmp_path / "config.json"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (result.stdout, result.exception)
    assert runner.invoke(app, ["verify", str(output)]).stdout.strip() == "OK"
    evaluated = runner.invoke(app, ["evaluate", str(output), str(tmp_path / "actual.json")])
    assert evaluated.exit_code == 0, (evaluated.stdout, evaluated.exception)
    assert "session_body_mae_bps" in json.loads(evaluated.stdout)
    original = output.read_bytes()
    assert runner.invoke(app, args).exit_code != 0
    assert output.read_bytes() == original


def test_unknown_schema_and_noncanonical_date_cannot_access_sidecar_paths(tmp_path, legacy_payload):
    directory = tmp_path / "predictions"
    directory.mkdir()
    for name, payload in (
        ("unknown", legacy_payload | {"schema_version": "unknown"}),
        ("unsafe", legacy_payload | {"trading_date": "../secret"}),
    ):
        (directory / f"{name}.json").write_text(json.dumps(payload))
    report = audit_repository(tmp_path)
    assert report["eligible_count"] == 0
    assert all(row["eligibility"]["reasons"] == ["invalid_record"] for row in report["rows"])
