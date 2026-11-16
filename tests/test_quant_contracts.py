from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from tw_scenario.quant.contracts import (
    TZ,
    Bar,
    Calendar,
    Prices,
    asof,
    at,
    decode,
    encode,
    reference,
)


def bar(day: date, *, instrument="TX", contract="202601", session="day", available=None):
    start = at(day if session == "day" else day - timedelta(days=1), "08:45" if session == "day" else "15:00")
    end = at(day, "13:45" if session == "day" else "05:00")
    return Bar(
        instrument=instrument,
        contract=contract if instrument == "TX" else "",
        trading_date=day,
        session=session,
        open=100,
        high=103,
        low=98,
        close=101,
        session_start_at=start,
        session_end_at=end,
        available_at=available or at(day, "18:00" if session == "day" else "07:30"),
        fetched_at=at(day + timedelta(days=1), "12:00"),
        availability_basis="test fixture",
        source="synthetic-test",
        source_hash="0" * 64,
    )


def test_gbud_round_trip_and_coherence():
    original = Prices(open=101, high=106, low=97, close=103)
    transformed = encode(original, 100, "gbud")
    restored = decode(transformed, 100, "gbud")
    np.testing.assert_allclose(restored.array(), original.array(), rtol=1e-12)
    assert restored.high >= max(restored.open, restored.close)
    assert restored.low <= min(restored.open, restored.close)


def test_relative_decode_projects_to_coherent_ohlc():
    restored = decode(np.log(np.array([101, 99, 102, 104]) / 100), 100, "relative")
    assert restored.high == 104
    assert restored.low == 101


def test_cutoff_excludes_same_day_day_session_and_late_night_data():
    day = date(2026, 1, 6)
    prior = bar(day - timedelta(days=1))
    night = bar(day, session="night")
    late_night = bar(day, contract="202602", session="night", available=at(day, "08:31"))
    same_day = bar(day)
    available = asof([prior, night, late_night, same_day], day)
    assert night in available
    assert prior in available
    assert late_night not in available
    assert same_day not in available


def test_rolls_five_exchange_sessions_before_expiry():
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(15)]
    calendar = Calendar(
        days=days,
        expiries={"202601": days[10], "202602": days[14]},
        source="synthetic-test",
        available_at=at(days[0]),
    )
    assert calendar.contract_for(days[4]) == "202601"
    assert calendar.contract_for(days[5]) == "202602"


def test_roll_calendar_tail_requires_verified_successor_expiry():
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(11)]
    calendar = Calendar(
        days=days,
        expiries={"202601": days[10]},
        source="synthetic-test",
        available_at=at(days[0]),
    )
    with pytest.raises(ValueError, match="tail lacks a verified successor"):
        calendar.contract_for(days[5])


def test_reference_requires_previous_exchange_day_same_contract():
    days = [date(2026, 1, 5), date(2026, 1, 6)]
    calendar = Calendar(
        days=days,
        expiries={"202601": days[-1]},
        source="synthetic-test",
        available_at=at(days[0]),
    )
    previous_old_contract = bar(days[0], contract="202512")
    with pytest.raises(ValueError, match="same-contract"):
        reference([previous_old_contract], calendar, days[1], "TX", "202601")


def test_night_session_physical_dates_can_span_weekend():
    monday = date(2026, 1, 12)
    friday = date(2026, 1, 9)
    night = Bar(
        instrument="TX",
        contract="202601",
        trading_date=monday,
        session="night",
        open=100,
        high=103,
        low=98,
        close=101,
        session_start_at=at(friday, "15:00"),
        session_end_at=at(monday, "05:00"),
        available_at=at(monday, "07:30"),
        fetched_at=at(monday, "12:00"),
        availability_basis="test fixture",
        source="synthetic-test",
        source_hash="0" * 64,
    )
    assert night.session_start_at.astimezone(TZ).date() == friday
    assert night.trading_date == monday
