import gzip
import io
import json
from datetime import date, datetime
from typing import ClassVar

import pytest

from tw_scenario.split_forecast import OHLC
from tw_scenario.taifex import FuturesRow, fetch_month, night_feature, parse_tx_csv

HEADER = "交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,成交量,結算價,交易時段\n"


def row(day, contract, session, close=100):
    return FuturesRow(
        date.fromisoformat(day),
        contract,
        session,
        OHLC(open=close, high=close, low=close, close=close),
        100,
        "synthetic.csv",
    )


def cutoff(day, hour="08:30:00"):
    return datetime.fromisoformat(f"{day}T{hour}+08:00")


def test_csv_keeps_sessions_and_last_price_not_settlement():
    raw = (
        HEADER
        + "2025/12/31,TX,202603,100,101,99,100,20,999,一般\n"
        + "2025/12/31,TX,202603,100,102,99,101,10,-,盤後\n"
    ).encode("cp950")
    rows = parse_tx_csv(raw, "synthetic.csv")
    assert [r.session for r in rows] == ["day", "night"]
    assert rows[0].prices.close == 100
    assert rows[1].prices.close == 101


def test_bad_schema_duplicates_unknown_session_and_incoherent_prices_rejected():
    with pytest.raises(ValueError, match="schema"):
        parse_tx_csv(b"<html>not data</html>", "synthetic.csv")
    valid = "2025/12/31,TX,202603,100,101,99,100,20,100,一般\n"
    with pytest.raises(ValueError, match="duplicate"):
        parse_tx_csv((HEADER + valid + valid).encode(), "synthetic.csv")
    with pytest.raises(ValueError, match="session"):
        parse_tx_csv((HEADER + valid.replace("一般", "unknown")).encode(), "synthetic.csv")
    with pytest.raises(ValueError):
        parse_tx_csv(
            (HEADER + valid.replace("100,101,99,100", "100,99,101,100")).encode(), "synthetic.csv"
        )


def test_weekly_and_spreads_are_excluded_and_missing_last_trade_is_not_filled():
    data = (
        HEADER
        + "2025/12/31,TX,202603,100,101,99,-,0,999,一般\n"
        + "2025/12/31,TX,202603W1,100,101,99,100,1,100,一般\n"
        + "2025/12/31,TX,202603/202606,100,101,99,100,1,100,一般\n"
    )
    rows = parse_tx_csv(data.encode(), "synthetic.csv")
    assert len(rows) == 1 and rows[0].prices is None


def test_rollover_uses_selected_contracts_own_previous_close():
    rows = [
        row("2026-01-21", "202601", "day", 10000),
        row("2026-01-21", "202602", "day", 20000),
        row("2026-01-22", "202602", "night", 20100),
    ]
    result = night_feature(date(2026, 1, 22), rows, cutoff("2026-01-22"))
    assert result["contract"] == result["reference_contract"] == "202602"
    assert result["night_return_bps"] == pytest.approx(50)
    assert result["available_at"] is None and result["historical_availability_verified"] is False


def test_missing_front_reference_does_not_silently_select_a_different_contract():
    rows = [
        row("2026-01-20", "202602", "day"),
        row("2026-01-21", "202601", "night"),
        row("2026-01-21", "202602", "night"),
    ]
    with pytest.raises(ValueError, match="same-contract"):
        night_feature(date(2026, 1, 21), rows, cutoff("2026-01-21"))


def test_friday_night_is_mondays_exchange_date_but_ends_saturday():
    rows = [row("2026-01-16", "202601", "day"), row("2026-01-19", "202601", "night")]
    result = night_feature(date(2026, 1, 19), rows, cutoff("2026-01-19"))
    assert result["session_end"] == "2026-01-17T05:00:00+08:00"
    assert result["age_hours"] == 51.5


@pytest.mark.parametrize(
    "at",
    [
        "2026-01-20T04:59:00+08:00",
        "2026-01-20T08:31:00+08:00",
        "2026-01-20T08:30:00",
        "2026-01-21T08:30:00+08:00",
    ],
)
def test_unfinished_late_naive_or_wrong_day_cutoff_rejected(at):
    rows = [row("2026-01-19", "202601", "day"), row("2026-01-20", "202601", "night")]
    with pytest.raises(ValueError):
        night_feature(date(2026, 1, 20), rows, datetime.fromisoformat(at))


def test_same_day_daytime_prices_cannot_change_the_night_feature():
    rows = [row("2026-01-19", "202601", "day"), row("2026-01-20", "202601", "night", 101)]
    before = night_feature(date(2026, 1, 20), rows, cutoff("2026-01-20"))
    after = night_feature(
        date(2026, 1, 20), rows + [row("2026-01-20", "202601", "day", 200)], cutoff("2026-01-20")
    )
    assert before == after


def test_month_download_decompresses_and_cache_is_immutable(tmp_path, monkeypatch):
    raw = (HEADER + "2025/12/31,TX,202603,100,101,99,100,20,999,一般\n").encode("cp950")

    class Response(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {"Content-Encoding": "gzip"}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(gzip.compress(raw)))
    metadata = fetch_month("202512", tmp_path)
    assert metadata["available_at"] is None and not metadata["historical_availability_verified"]
    assert (tmp_path / "TX-202512.csv").read_bytes() == raw

    def forbid_network(*a, **k):
        raise AssertionError("valid cache must not trigger another download")

    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    assert fetch_month("202512", tmp_path) == metadata
    (tmp_path / "TX-202512.csv").write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="changed"):
        fetch_month("202512", tmp_path)


def test_cached_range_mismatch_is_rejected(tmp_path, monkeypatch):
    raw = (HEADER + "2025/12/31,TX,202603,100,101,99,100,20,999,一般\n").encode("cp950")

    class Response(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(raw))
    fetch_month("202512", tmp_path)
    path = tmp_path / "TX-202512.metadata.json"
    metadata = json.loads(path.read_text())
    metadata["form"]["queryStartDate"] = "2025/11/01"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="date range"):
        fetch_month("202512", tmp_path)
"""Synthetic CSV fixtures only; not market data or performance evidence."""

