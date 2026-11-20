"""Synthetic local round trip; does not claim any live Google Sheet write."""

import hashlib
import io
import json
import zipfile
from copy import deepcopy
from datetime import date

import pytest

from scripts import finish_shadow_day, monitor_shadow
from tests.test_shadow import example, stamp
from tw_scenario.morning_adapter import (
    HEADERS,
    IMMUTABLE,
    close_export,
    morning_export,
    verify_readback,
    verify_review_readback,
)
from tw_scenario.shadow import seal
from tw_scenario.snapshot import write_immutable_json


def paired():
    *_, forecast = example()
    tx = deepcopy(forecast["markets"]["TAIEX"])
    tx["inputs"].update({"market": "TX_selected_monthly", "contract": "202611"})
    forecast["markets"]["TX_selected_monthly"] = tx
    return seal(forecast)


def actuals(forecast):
    return [{"trading_date": forecast["trading_date"], "market":
             "TX" if m == "TX_selected_monthly" else "TAIEX", "contract": d["inputs"]["contract"],
             "session": "day", "available_at": "2026-11-05T14:00:00+08:00",
             "source": "synthetic", "availability_basis": "synthetic unit test",
             **d["models"][d["default_baseline"]]["prices"]}
            for m, d in forecast["markets"].items()]


def test_render_review_and_readback_preserve_every_morning_cell():
    forecast = paired()
    morning = morning_export(forecast, HEADERS)
    rows = [r["values"][:] for r in morning["records"]]
    assert verify_readback(morning, rows)
    assert all(not row[i] for row in rows for i in (*range(6, 14), 15))
    assert "未估計方向機率" in rows[0][4]
    assert "202611" in rows[1][1]
    patch = close_export(forecast, morning, rows, actuals(forecast), stamp(date(2026, 11, 5), 14, 30))
    for row, change in zip(rows, patch["records"]):
        for column, value in change["patch_by_zero_based_column"].items():
            row[int(column)] = value
    assert verify_review_readback(patch, rows)
    assert verify_readback(morning, rows, reviewed=True)
    assert all(rows[0][i] == morning["records"][0]["values"][i] for i in IMMUTABLE)
    assert not rows[0][6].startswith("實際開盤")
    assert "先後" in rows[0][7] and "未執行因果歸因" in rows[0][11]
    rows[0][2] += " change a number"
    with pytest.raises(ValueError, match="morning"):
        verify_review_readback(patch, rows)


def test_changed_headers_mismatched_actual_or_tampered_narrative_is_rejected():
    forecast = paired()
    with pytest.raises(ValueError, match="headers"):
        morning_export(forecast, ["changed", *HEADERS[1:]])
    morning = morning_export(forecast, HEADERS)
    rows = [r["values"][:] for r in morning["records"]]
    wrong = actuals(forecast)
    wrong[1]["contract"] = "202612"
    with pytest.raises(ValueError, match="missing"):
        close_export(forecast, morning, rows, wrong, stamp(date(2026, 11, 5), 14, 30))
    changed = deepcopy(morning)
    changed["records"][0]["values"][2] = "invented numerical forecast"
    changed = seal(changed)
    with pytest.raises(ValueError, match="differs"):
        close_export(forecast, changed, [r["values"] for r in changed["records"]],
                     actuals(forecast), stamp(date(2026, 11, 5), 14, 30))


def test_local_full_cycle_scores_two_markets_without_fabricating_preopen(monkeypatch, tmp_path):
    forecast = paired()
    morning = morning_export(forecast, HEADERS)
    run = tmp_path / "runs" / "one"
    write_immutable_json(forecast, run / "forecast.json")
    write_immutable_json({"snapshot_hash": forecast["snapshot_hash"],
                          "local_persisted_at": "2026-11-05T08:26:00+08:00"}, run / "local-receipt.json")
    write_immutable_json(morning, run / "morning-export.json")
    at = stamp(date(2026, 11, 5), 14, 30)
    monkeypatch.setattr(finish_shadow_day, "now", lambda: at)
    monkeypatch.setattr(monitor_shadow, "now", lambda: at)
    monkeypatch.setattr(finish_shadow_day, "collect_day", lambda *args: {
        "actuals": actuals(forecast), "failures": []})
    result = finish_shadow_day.finish_local([run], tmp_path / "output", at)
    assert result["actual_count"] == 2 and result["failures"] == []
    assert result["monitor_scored_groups"] == 2
    assert result["reviews"][0]["status"] == "local_review_verified"
    later = stamp(date(2026, 11, 6), 14, 30)
    missing = finish_shadow_day.finish_local([run], tmp_path / "next-day", later)
    assert missing["actual_count"] == 0 and missing["failures"]


def test_history_uses_remote_bytes_keeps_versions_and_never_claims_formal_accuracy(monkeypatch, tmp_path):
    forecast = paired()
    receipt = {"snapshot_hash": forecast["snapshot_hash"],
               "local_persisted_at": "2026-11-05T08:26:00+08:00"}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("job.json", json.dumps(seal({"status": "diagnostic_complete"})))
        zipped.writestr("actuals.json", json.dumps(actuals(forecast)))
        zipped.writestr("runs/run-100/forecast.json", json.dumps(forecast))
        zipped.writestr("runs/run-100/local-receipt.json", json.dumps(receipt))
    archive = stream.getvalue()
    meta = {"id": 300, "name": "shadow-close-200-1", "workflow_run": {"id": 200},
            "expired": False, "digest": "sha256:" + hashlib.sha256(archive).hexdigest()}

    def api(url, binary=False):
        if binary:
            return archive
        if "/workflows/" in url:
            return {"total_count": 1, "workflow_runs": [{"id": 200, "created_at": "2026-11-05T07:30:00Z"}]}
        return {"total_count": 1, "artifacts": [meta]}
    at = stamp(date(2026, 11, 6), 14, 30)
    monkeypatch.setattr(finish_shadow_day, "api", api)
    monkeypatch.setattr(monitor_shadow, "now", lambda: at)
    report = finish_shadow_day.aggregate_recent("owner/repo", [], [], tmp_path / "history", at)
    assert report["attempted_runs"] == 1 and len(report["groups"]) == 2
    assert all(g["point_metrics"]["scored"] == 1 for g in report["groups"])
    assert report["history_issues"] == []
    assert not report["formal_accuracy_eligible"] and not report["all_time_accuracy_claim"]
    meta["digest"] = "sha256:bad"
    invalid = finish_shadow_day.aggregate_recent("owner/repo", [], [], tmp_path / "bad-history", at)
    assert invalid["groups"] == [] and invalid["history_issues"]
