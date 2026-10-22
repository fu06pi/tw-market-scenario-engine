"""Synthetic fixtures validate public-data archival safeguards, not model accuracy."""

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from tw_scenario import twse_history as h


def payload(rows=None):
    return {
        "stat": "OK",
        "fields": h.FIELDS,
        "data": rows or [[" 88/01/05", "6,000.00", "6,100.00", "5,900.00", "6,050.00"]],
    }


def test_roc_decimal_precision_and_historical_saturdays():
    p = payload([[" 88/01/30", "6,000.001", "6,100", "5,900", "6,050"]])
    rows, _ = h.parse_month(p, date(1999, 1, 1), date(1999, 1, 31))
    assert (
        rows[0]["date"] == "1999-01-30"
    )  # Saturday trading existed; do not impose modern weekdays.
    assert rows[0]["open"] == "6000.001"


@pytest.mark.parametrize(
    "row",
    [
        ["88/01/05", "NaN", "6100", "5900", "6050"],
        ["88/01/05", "6000", "5999", "5900", "6050"],
        ["88/01/05", "6000", "6100", "6101", "6050"],
        ["88/01/05", "0", "6100", "5900", "6050"],
        ["88/01/05", "--", "6100", "5900", "6050"],
    ],
)
def test_reject_invalid_source_ohlc(row):
    with pytest.raises(ValueError):
        h.parse_month(payload([row]), date(1999, 1, 1), date(1999, 1, 31))


def test_reject_schema_month_duplicate_and_out_of_order():
    p = payload()
    p["fields"] = ["unknown"]
    with pytest.raises(ValueError, match="schema"):
        h.parse_month(p, date(1999, 1, 1), date(1999, 1, 31))
    p = payload()
    p["data"][0][0] = "88/02/01"
    with pytest.raises(ValueError, match="another month"):
        h.parse_month(p, date(1999, 1, 1), date(1999, 1, 31))
    p = payload()
    p["data"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        h.parse_month(p, date(1999, 1, 1), date(1999, 1, 31))
    p = payload(
        [["88/01/06", "6000", "6100", "5900", "6050"], ["88/01/05", "6000", "6100", "5900", "6050"]]
    )
    with pytest.raises(ValueError, match="unsorted"):
        h.parse_month(p, date(1999, 1, 1), date(1999, 1, 31))


def test_exclude_incomplete_day_and_keep_historical_dates():
    p = payload(
        [
            ["115/10/21", "30000", "31000", "29000", "30500"],
            ["115/10/22", "30000", "31000", "29000", "30500"],
        ]
    )
    rows, excluded = h.parse_month(p, date(2026, 10, 1), date(2026, 10, 21))
    assert len(rows) == 1 and excluded == ["2026-10-22"]


def test_mutable_month_refresh_and_historical_cache_reuse():
    meta = {"as_of_date": "2026-10-21"}
    assert not h.needs_refresh(meta, date(1999, 1, 1), date(2026, 10, 22))
    assert h.needs_refresh(meta, date(2026, 10, 1), date(2026, 11, 1))


def test_cache_checksum_and_resume_avoids_network(tmp_path, monkeypatch):
    folder = tmp_path / "monthly"
    folder.mkdir()
    p = folder / "1999-01.json"
    body = json.dumps(payload()).encode()
    p.write_bytes(body)
    meta = {"as_of_date": "2026-10-21", "response_sha256": h.sha256(body)}
    p.with_suffix(".metadata.json").write_text(json.dumps(meta))
    monkeypatch.setattr(h, "download", lambda *a, **k: pytest.fail("valid cache must not download"))
    assert h.load_or_fetch(tmp_path, date(1999, 1, 1), date(2026, 10, 21), 2)[2] is False
    p.write_bytes(body + b" ")
    with pytest.raises(ValueError, match="checksum"):
        h.load_or_fetch(tmp_path, date(1999, 1, 1), date(2026, 10, 21), 2)


def test_rate_limit_honors_long_retry_after_without_retry(monkeypatch):
    from urllib.error import HTTPError

    calls = []

    def limited(*args, **kwargs):
        calls.append(1)
        raise HTTPError(
            "https://www.twse.com.tw/", 429, "Too Many Requests", {"Retry-After": "120"}, None
        )

    monkeypatch.setattr(h, "urlopen", limited)
    with pytest.raises(h.RateLimited, match="120"):
        h.download(date(1999, 1, 1), date(2026, 10, 21))
    assert len(calls) == 1


def test_leap_year_month_iteration_and_retry_date():
    assert [m.isoformat() for m in h.iter_months(date(2023, 12, 1), date(2024, 3, 1))] == [
        "2023-12-01",
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
    ]
    now = datetime(2026, 10, 22, 0, 0, tzinfo=ZoneInfo("UTC"))
    assert h.retry_seconds("Thu, 22 Oct 2026 00:02:00 GMT", now) == 120


def test_partial_archive_never_claims_missing_month_complete(tmp_path, monkeypatch):
    body = json.dumps(payload()).encode()

    def fetch(folder, month, asof, interval):
        if month.month == 2:
            raise h.RateLimited("HTTP 429")
        return body, {"response_sha256": h.sha256(body)}, False

    monkeypatch.setattr(h, "load_or_fetch", fetch)
    result = h.archive(tmp_path, date(1999, 2, 28), interval=1)
    assert result["status"] == "partial" and result["completed_months"] == 1
    assert result["missing_months"] == ["1999-02"]
    assert (tmp_path / "daily.csv").read_text().splitlines()[1].startswith("1999-01-05,")
