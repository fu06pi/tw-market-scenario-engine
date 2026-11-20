"""Synthetic checks of fixed research transforms and third-party receipt gates."""

from copy import deepcopy
from datetime import date

import numpy as np
import pytest

from scripts.diagnose_thresholds_and_errors import decompose_error, threshold_metrics
from scripts.run_timesfm_comparison import METHODS, context_for, reconstruct
from tests.test_morning_adapter import actuals, paired
from tests.test_shadow import stamp
from tw_scenario.forward_trial import (
    aggregate_trial_comments,
    assess_comment,
    comment_body,
    package_preopen,
    parse_comment,
)
from tw_scenario.shadow import seal
from tw_scenario.split_forecast import OHLC


def historical():
    return [{"date": f"2025-01-{i + 1:02d}", "previous_close": 100,
             "night_return_bps": i, "actual": OHLC(open=100, high=104, low=98, close=102)} for i in range(5)]


def test_context_ignores_today_labels_and_past_only_ignores_today_night():
    rows = historical()
    first = context_for(rows, 4, METHODS[0], 4)
    residual = context_for(rows, 4, METHODS[1], 4)
    changed = deepcopy(rows)
    changed[4]["actual"] = OHLC(open=1, high=1000, low=1, close=999)
    changed[4]["night_return_bps"] = 500
    after = context_for(changed, 4, METHODS[0], 4)
    assert np.array_equal(first[0], after[0]) and first[1] == after[1]
    assert np.array_equal(residual[0], context_for(changed, 4, METHODS[1], 4)[0])
    assert residual[1] != context_for(changed, 4, METHODS[1], 4)[1]
    rows[1]["date"] = rows[4]["date"]
    with pytest.raises(ValueError): context_for(rows, 4, METHODS[0], 4)


def test_positive_projection_preserves_open_close_and_counts_invalid_wicks():
    prices, changed = reconstruct([0, -.1, .2, .05], 100, 0)
    assert prices.open == 100 and prices.close == pytest.approx(100 * np.exp(.05))
    assert prices.high == prices.close and prices.low == prices.open
    assert changed == {"high": True, "low": True}
    with pytest.raises(ValueError): reconstruct([0, 0, float("nan"), 0], 100, 0)


def test_error_location_is_exact_and_direction_is_not_point_tolerance():
    actual = OHLC(open=110, high=115, low=100, close=105)
    predicted = OHLC(open=110, high=121, low=100, close=120)
    d = decompose_error(100, actual, predicted)
    assert d["gap_error_log_bp"] == pytest.approx(0)
    assert d["body_error_log_bp"] == pytest.approx(d["close_error_log_bp"])
    rows = [{"date": "2026-01-02", "actual": actual.model_dump(), "predicted": predicted.model_dump()}]
    assert threshold_metrics(rows, {"2026-01-02": 100}, 5)["close"]["direction_accuracy"] == 1
    assert abs(actual.close - predicted.close) == 15  # direction match does not imply small price error


def receipt():
    forecast = paired()
    local = {"snapshot_hash": forecast["snapshot_hash"], "local_persisted_at": "2026-11-05T08:26:00+08:00"}
    payload = package_preopen(forecast, local, [], "a" * 40, "b" * 64, stamp(date(2026, 11, 5), 8, 27))
    metadata = {"id": 1, "issue_url": "https://api.github.com/repos/owner/repo/issues/4",
                "user": {"login": "owner"}, "created_at": "2026-11-05T00:28:00Z",
                "updated_at": "2026-11-05T00:28:00Z", "body": comment_body(payload)}
    return payload, metadata


def test_comment_receipt_requires_exact_body_owner_ledger_and_deadline():
    payload, metadata = receipt()
    assert parse_comment(metadata["body"]) == payload
    assert payload["missing_news_markets"] == ["TAIEX", "TX_selected_monthly"]
    assert assess_comment(metadata, "owner/repo", 4, "b" * 64, "a" * 40)["timing_verified"]
    for field, value in [("updated_at", "2026-11-05T00:40:00Z"),
                         ("issue_url", "https://api.github.com/repos/other/repo/issues/4"),
                         ("body", metadata["body"].replace("preopen", "changed", 1))]:
        changed = {**metadata, field: value}
        with pytest.raises(ValueError): assess_comment(changed, "owner/repo", 4, "b" * 64, "a" * 40)
    metadata["user"] = {"login": "someone-else"}
    with pytest.raises(ValueError): assess_comment(metadata, "owner/repo", 4, "b" * 64, "a" * 40)


def test_late_packaging_and_clock_rollback_are_rejected():
    forecast = paired()
    local = {"snapshot_hash": forecast["snapshot_hash"], "local_persisted_at": "2026-11-05T08:26:00+08:00"}
    for at in [stamp(date(2026, 11, 5), 8, 40), stamp(date(2026, 11, 5), 8, 24)]:
        with pytest.raises(ValueError): package_preopen(forecast, local, [], "a" * 40, "b" * 64, at)


def test_aggregate_keeps_missing_news_and_rejects_duplicate_preopen_dates():
    payload, metadata = receipt()
    close_payload = seal({"kind": "close", "source_commit": "a" * 40,
                          "protocol_sha256": "b" * 64, "preopen_comment_id": metadata["id"],
                          "official_actuals": actuals(payload["forecast"]), "actual_failures": []})
    close_meta = {**metadata, "id": 2, "body": comment_body(close_payload),
                  "created_at": "2026-11-05T07:30:00Z", "updated_at": "2026-11-05T07:30:00Z"}
    report = aggregate_trial_comments([metadata, close_meta], "owner/repo", 4, "b" * 64, "a" * 40)
    assert report["verified_preopen_days"] == 1 and report["expected_pairs"] == 2
    assert report["paired_n"] == 0 and len(report["missing"]) == 2
    duplicated = aggregate_trial_comments([metadata, {**metadata, "id": 3}, close_meta],
                                          "owner/repo", 4, "b" * 64, "a" * 40)
    assert duplicated["verified_preopen_days"] == 0 and duplicated["rejected_preopen_days"] == 1
    assert duplicated["receipt_issues"]
