from datetime import date

import pytest

from tw_scenario.finmind_history import normalize


def row(**changes):
    return dict(
        date="1994-10-01", stock_id="TAIEX", open=100, max=102, min=99, close=101, **changes
    )


def test_old_history_and_precision():
    assert normalize([row()], 1994, date(1994, 12, 31))[0]["date"] == "1994-10-01"


@pytest.mark.parametrize(
    "field,value",
    [("stock_id", "0050"), ("date", "1995-01-01"), ("max", 99), ("close", float("nan"))],
)
def test_reject_wrong_instrument_date_and_bad_prices(field, value):
    r = row()
    r[field] = value
    with pytest.raises(ValueError):
        normalize([r], 1994, date(1994, 12, 31))


def test_duplicate_and_cutoff():
    with pytest.raises(ValueError):
        normalize([row(), row()], 1994, date(1994, 12, 31))
    with pytest.raises(ValueError):
        normalize([row()], 1994, date(1994, 9, 30))


def test_quota_denial_stops_without_requesting_later_years(tmp_path, monkeypatch):
    import json
    from urllib.error import HTTPError

    from tw_scenario import finmind_history as h

    calls = []

    def denied(*args):
        calls.append(args)
        raise HTTPError("https://api.finmindtrade.com", 402, "quota", {}, None)

    monkeypatch.setattr(h, "cached", denied)
    assert h.archive(tmp_path, date(2001, 12, 31), start=1999) == []
    assert len(calls) == 1
    manifest = json.loads((tmp_path / "finmind/taiex/manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["outcomes"][0]["http_status"] == 402
