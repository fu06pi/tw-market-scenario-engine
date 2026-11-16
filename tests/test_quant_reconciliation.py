from __future__ import annotations

from datetime import date, timedelta

import pytest

from tw_scenario.quant.contracts import Bar, at
from tw_scenario.quant.reconciliation import reconcile_bars


def bar(*, source: str, close: float = 101, settlement: float | None = 100.5):
    day = date(2026, 1, 6)
    return Bar(
        instrument="TX",
        contract="202601",
        trading_date=day,
        session="day",
        open=100,
        high=103,
        low=98,
        close=close,
        settlement=settlement,
        session_start_at=at(day, "08:45"),
        session_end_at=at(day, "13:45"),
        available_at=at(day, "18:00"),
        fetched_at=at(day + timedelta(days=1)),
        availability_basis="ASSUMPTION: test fixture",
        source=source,
        source_hash=("1" if source == "FinMind" else "2") * 64,
    )


def test_reconciliation_keeps_close_and_settlement_separate():
    result = reconcile_bars([bar(source="FinMind")], [bar(source="official")])
    assert result["status"] == "PASS_RETROSPECTIVE_VALUE_AGREEMENT"
    assert result["fields"]["close"]["paired"] == 1
    assert result["fields"]["settlement"]["paired"] == 1
    assert result["point_in_time_verified"] is False


def test_reconciliation_reports_every_missing_key_and_value_mismatch():
    result = reconcile_bars(
        [bar(source="FinMind", close=102, settlement=None)],
        [bar(source="official", close=101, settlement=100.5)],
    )
    assert result["status"] == "FAIL"
    assert result["fields"]["close"]["mismatches"] == 1
    assert result["fields"]["settlement"]["mismatches"] == 1
    assert result["mismatch_rows"][0]["fields"]["settlement"]["reason"] == (
        "availability_mismatch"
    )


def test_reconciliation_rejects_duplicates_and_empty_input():
    duplicate = bar(source="FinMind")
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_bars([duplicate, duplicate], [])
    assert reconcile_bars([], [])["status"] == "FAIL_EMPTY"
