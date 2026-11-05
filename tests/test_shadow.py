"""Synthetic clocks, prices and API metadata; not forward-market evidence."""

import hashlib
import io
import json
import zipfile
from copy import deepcopy
from datetime import date, datetime, time

import pytest

from scripts import monitor_shadow
from scripts.monitor_shadow import monitor
from scripts.verify_shadow_artifact import assess
from tests.test_refinements import samples
from tw_scenario import shadow
from tw_scenario.intervals import calibrate
from tw_scenario.policy import ScoringPolicy
from tw_scenario.refinements import fit
from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import TAIPEI


def stamp(day, hour=8, minute=20):
    return datetime.combine(day, time(hour, minute), TAIPEI)


def example():
    day = date(2026, 11, 5)
    row = {**samples()[-1], "date": day.isoformat(), "feature_latest_date": "2026-11-04",
           "contract": None}
    row.pop("target")
    cal_rows = [{"date": f"2025-12-{i + 1:02d}", "market": "TAIEX", "method": "night_carry",
                 **{f"{k}_ae_bps": 100 for k in ("open", "high", "low", "close")}}
                for i in range(30)]
    bundle = shadow.seal({"schema_version": "shadow-checkpoint-v1",
                         "scoring_policy": ScoringPolicy().model_dump(),
                         "frozen_at": stamp(date(2026, 11, 4)).isoformat(), "markets": {
        "TAIEX": {"models": {"night_carry": fit(samples(), "night_carry", "TAIEX")},
                  "default_baseline": "night_carry", "intervals": {
                      "night_carry": calibrate(cal_rows, day.isoformat())}}}})
    receipt = {"fetched_at": stamp(day).isoformat()}
    forecast = shadow.generate(bundle, {"TAIEX": row}, [receipt], stamp(day, 8, 25))
    return day, row, bundle, receipt, forecast


@pytest.mark.parametrize("at", [datetime(2026, 11, 5, 8, 30, tzinfo=TAIPEI),
                                  datetime(2026, 11, 7, 8, 15, tzinfo=TAIPEI)])
def test_late_or_weekend_runs_persist_failure_without_fetch(monkeypatch, tmp_path, at):
    monkeypatch.setattr(shadow, "now", lambda: at)
    assert shadow.run(tmp_path / "absent.json", tmp_path / "run") == 2
    assert (tmp_path / "run/failure.json").exists()
    assert not (tmp_path / "run/forecast.json").exists()


def test_late_receipt_late_generation_tampered_and_same_day_checkpoint_rejected():
    day, row, bundle, receipt, _ = example()
    with pytest.raises(ValueError, match="08:30"):
        shadow.generate(bundle, {"TAIEX": row}, [{"fetched_at": stamp(day, 8, 31).isoformat()}],
                        stamp(day, 8, 35))
    with pytest.raises(ValueError, match="08:40"):
        shadow.generate(bundle, {"TAIEX": row}, [receipt], stamp(day, 8, 40))
    bundle["frozen_at"] = stamp(day).isoformat()
    with pytest.raises(ValueError, match="hash"):
        shadow.generate(bundle, {"TAIEX": row}, [receipt], stamp(day, 8, 25))
    bundle = shadow.seal(bundle)
    with pytest.raises(ValueError, match="predate"):
        shadow.generate(bundle, {"TAIEX": row}, [receipt], stamp(day, 8, 25))


def test_synthetic_official_parsers_validate_date_session_and_last_price():
    raw = ('dataStart 日期：2026/11/05 臺股期貨 TX 2026/11/04 15:00~次日05:00 盤後'
           '<table><th>最後成交價</th><tr>' + ''.join(f'<td>{v}</td>' for v in
           ['TX', '202611', '100', '103', '98', '101', '1', '1%', '200', '999']) +
           '</tr></table>').encode()
    rows, start = shadow.futures_rows(raw, date(2026, 11, 5), "night")
    assert rows["202611"].close == 101  # settlement 999 must never be used
    assert start == date(2026, 11, 4)
    with pytest.raises(ValueError, match="date"):
        shadow.futures_rows(raw, date(2026, 11, 6), "night")
    with pytest.raises(ValueError, match="session"):
        shadow.futures_rows(raw, date(2026, 11, 5), "day")
    raw = json.dumps({"stat": "OK", "fields": ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"],
                      "data": [["115/11/04", "100", "103", "98", "101"]]}).encode()
    assert shadow.cash_rows(raw)[date(2026, 11, 4)].close == 101


def remote_fixture():
    _, _, _, _, forecast = example()
    receipt = {"snapshot_hash": forecast["snapshot_hash"],
               "local_persisted_at": "2026-11-05T08:26:00+08:00"}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("forecast.json", json.dumps(forecast))
        zipped.writestr("local-receipt.json", json.dumps(receipt))
    archive = stream.getvalue()
    digest = "sha256:" + hashlib.sha256(archive).hexdigest()
    metadata = {"id": 123, "expired": False, "digest": digest,
                "url": "https://api.github.com/repos/owner/repo/actions/artifacts/123",
                "created_at": "2026-11-05T00:27:00Z", "updated_at": "2026-11-05T00:28:00Z"}
    return forecast, receipt, archive, metadata, digest


def test_remote_evidence_requires_uploaded_bytes_and_finalized_before_deadline():
    forecast, receipt, archive, metadata, digest = remote_fixture()
    assert assess(metadata, archive, forecast, receipt, "123", digest, "owner/repo")[
        "remote_persistence_verified"]
    assert assess(metadata, archive, forecast, receipt, "123", digest.removeprefix("sha256:"),
                  "owner/repo")["remote_persistence_verified"]
    with pytest.raises(ValueError, match="digest"):
        assess(metadata, archive + b"tampered", forecast, receipt, "123", digest, "owner/repo")
    with pytest.raises(ValueError, match="repository"):
        assess(metadata, archive, forecast, receipt, "123", digest, "other/repo")
    metadata["updated_at"] = "2026-11-05T00:40:00Z"
    with pytest.raises(ValueError, match="deadline"):
        assess(metadata, archive, forecast, receipt, "123", digest, "owner/repo")


def test_monitor_missing_actuals_and_failures_do_not_manufacture_accuracy(tmp_path):
    forecast, receipt, *_ = remote_fixture()
    good = tmp_path / "good"
    write_immutable_json(forecast, good / "forecast.json")
    write_immutable_json(receipt, good / "local-receipt.json")
    bad = tmp_path / "bad"
    write_immutable_json({"reason": "source missed cutoff"}, bad / "failure.json")
    report = monitor([good, bad], [])
    assert report["attempted_runs"] == 2 and len(report["failed_runs"]) == 1
    assert len(report["pending_labels"]) == 1
    assert report["groups"][0]["point_metrics"]["scored"] == 0
    assert report["groups"][0]["marginal_80"]["close"]["coverage"] is None
    assert report["formal_accuracy_eligible"] is False
    changed = deepcopy(forecast)
    changed["markets"]["TAIEX"]["models"]["night_carry"]["prices"]["close"] = 1
    changed = shadow.seal(changed)
    # The receipt no longer binds to the document; that whole run must fail.
    other = tmp_path / "other"
    write_immutable_json(changed, other / "forecast.json")
    write_immutable_json(receipt, other / "local-receipt.json")
    assert len(monitor([other], [])["failed_runs"]) == 1


def test_monitor_scores_actual_and_rejects_whole_run_on_late_model_corruption(tmp_path, monkeypatch):
    forecast, receipt, *_ = remote_fixture()
    monkeypatch.setattr(monitor_shadow, "now", lambda: stamp(date(2026, 11, 6)))
    actual = {"trading_date": forecast["trading_date"], "market": "TAIEX", "session": "day",
              "available_at": "2026-11-05T14:00:00+08:00", "source": "synthetic",
              "availability_basis": "synthetic unit test", **forecast["markets"]["TAIEX"][
                  "models"]["night_carry"]["prices"]}
    good = tmp_path / "good"
    write_immutable_json(forecast, good / "forecast.json")
    write_immutable_json(receipt, good / "local-receipt.json")
    result = monitor([good], [actual])
    assert result["groups"][0]["point_metrics"]["close_mae_bps"] == pytest.approx(0)
    assert result["groups"][0]["marginal_80"]["close"]["coverage"] == 1
    corrupt = deepcopy(forecast)
    corrupt["markets"]["TAIEX"]["models"]["broken"] = {"prices": {"open": -1}}
    corrupt = shadow.seal(corrupt)
    bad = tmp_path / "bad"
    write_immutable_json(corrupt, bad / "forecast.json")
    write_immutable_json({**receipt, "snapshot_hash": corrupt["snapshot_hash"]}, bad / "local-receipt.json")
    result = monitor([bad], [actual])
    assert len(result["failed_runs"]) == 1 and result["groups"] == []
