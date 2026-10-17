"""Synthetic safeguards: these are not market accuracy evidence."""

from datetime import date, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from tw_scenario.research import baselines, tej
from tw_scenario.research.data import (
    Bar,
    canonical_hash,
    indices,
    load_dataset,
    reconstruct,
    sample_arrays,
    transform,
    write_new,
)
from tw_scenario.research.experiment import acceptable


def fixture():
    return [
        Bar(
            market="TX",
            contract="SYNTHETIC",
            trading_date=date(2025, 1, 1) + timedelta(days=i),
            open=100 + i,
            high=103 + i,
            low=99 + i,
            close=102 + i,
        )
        for i in range(90)
    ]


def test_reject_invalid_ohlc_nonfinite_and_wrong_contract():
    for kw in [{"open": float("nan")}, {"high": 100}, {"contract": None}, {"low": -1}]:
        raw = fixture()[0].model_dump()
        raw.update(kw)
        with pytest.raises(ValidationError):
            Bar.model_validate(raw)


@pytest.mark.parametrize("representation", ["relative", "gbud"])
def test_transforms_reconstruct_actual_prices(representation):
    bars = fixture()
    v = transform(bars, representation)
    back = reconstruct(v, np.array([b.close for b in bars[:-1]]), representation)
    actual = np.array([[b.open, b.high, b.low, b.close] for b in bars[1:]])
    np.testing.assert_allclose(back, actual, rtol=1e-6)


def test_future_targets_never_enter_context():
    bars = fixture()
    x, _, _, _ = sample_arrays(bars, "relative", 32, [40])
    changed = list(bars)
    changed[41] = Bar(
        **{**changed[41].model_dump(), "open": 200, "high": 300, "low": 190, "close": 250}
    )
    x2, y2, _, _ = sample_arrays(changed, "relative", 32, [40])
    np.testing.assert_array_equal(x, x2)
    assert y2[0, 3] != transform(bars, "relative")[40, 3]


def test_baselines_do_not_use_target_or_future():
    bars = fixture()
    original = baselines.forecast(bars, [60], "historical_shape_20")
    changed = list(bars)
    changed[61] = Bar(
        **{**changed[61].model_dump(), "open": 500, "high": 520, "low": 490, "close": 510}
    )
    np.testing.assert_array_equal(
        original, baselines.forecast(changed, [60], "historical_shape_20")
    )


def test_cross_contract_returns_rejected():
    bars = fixture()
    bars[10] = Bar(**{**bars[10].model_dump(), "contract": "OTHER"})
    with pytest.raises(ValueError, match="cross-contract"):
        transform(bars, "gbud")


def test_dataset_integrity_and_no_overwrite(tmp_path):
    payload = {"bars": [b.model_dump(mode="json") for b in fixture()]}
    payload["checksum"] = canonical_hash(payload)
    path = tmp_path / "data.json"
    write_new(path, payload)
    assert len(load_dataset(path)[1]) == 90
    with pytest.raises(FileExistsError):
        write_new(path, payload)
    payload["bars"][0]["close"] = 101
    import json

    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path)


def test_chronological_split_excludes_final_dates():
    bars = fixture()
    selected = indices(bars, 32, "2025-02-01", "2025-02-28")
    assert selected and all(
        date(2025, 2, 1) <= bars[i + 1].trading_date <= date(2025, 2, 28) for i in selected
    )


def test_gate_requires_all_three_ohl_limits_and_close():
    base = {f: {"mae_points": 100.0} for f in ["open", "high", "low", "close"]}
    candidate = {f: {"mae_points": 90.0} for f in base}
    assert acceptable(candidate, base, 0.95)
    candidate["low"]["mae_points"] = 106
    assert not acceptable(candidate, base, 0.95)
    candidate["low"]["mae_points"] = 90
    candidate["close"]["mae_points"] = 96
    assert not acceptable(candidate, base, 0.95)


def test_tej_permission_error_and_pagination_failure(monkeypatch):
    monkeypatch.setenv("TEJ_API_KEY", "SYNTHETIC")

    class Response:
        def json(self):
            return {"error": {"code": "PDB003", "message": "private message"}}

    monkeypatch.setattr(tej.Connection, "request", lambda *a, **k: Response())
    with pytest.raises(tej.AccessError, match="PDB003") as caught:
        tej.metadata("synthetic")
    assert "private message" not in str(caught.value)
    monkeypatch.setattr(
        tej,
        "request",
        lambda *a, **k: {
            "datatable": {"columns": [{"name": "date"}], "data": [["2025-01-01"]]},
            "meta": {"next_cursor_id": "more"},
        },
    )
    with pytest.raises(tej.AccessError, match="incomplete"):
        tej.get_rows("synthetic", "x", ["date"], max_pages=1)


def test_night_parser_rejects_wrong_day_session_and_contract():
    from tw_scenario.research.taifex import parse_night

    html = (
        '<input name="queryDate" value="2025/10/01">'
        "<table><tr><th>契約</th><th>到期月份(週別)</th><th>最後成交價</th><th>*成交量</th></tr>"
        "<tr><td>TX</td><td>202603</td><td>28000</td><td>3</td></tr></table>"
    )
    assert parse_night(html, "2025-10-01")["close"] == 28000
    with pytest.raises(ValueError, match="date"):
        parse_night(html, "2025-10-02")
    with pytest.raises(ValueError, match="night session"):
        parse_night(html.replace("*成交量", "*一般交易時段成交量"), "2025-10-01")
    assert parse_night(html.replace("202603", "202606"), "2025-10-01")["close"] is None
    assert parse_night(html.replace("28000", "-"), "2025-10-01")["close"] is None


def test_known_night_alignment_and_explicit_missing_covariates():
    from tw_scenario.research.night import covariates

    bars = fixture()
    target_day = str(bars[41].trading_date)
    values = covariates(bars, [40], 32, {target_day: 0.01})
    assert values.shape == (1, 1, 33)
    assert np.isnan(values[0, 0, :-1]).all()
    assert values[0, 0, -1] == pytest.approx(0.01)


def test_night_regression_does_not_fit_validation_labels():
    from tw_scenario.research.night import regression

    bars = fixture()
    training = list(range(32, 55))
    validation = [60]
    series = {str(b.trading_date): float(i % 7) / 1000 for i, b in enumerate(bars)}
    first = regression(bars, training, validation, series)
    changed = list(bars)
    changed[61] = Bar(
        **{**changed[61].model_dump(), "open": 500, "high": 520, "low": 490, "close": 510}
    )
    np.testing.assert_array_equal(first, regression(changed, training, validation, series))


def test_official_long_retry_after_stops_without_retry_or_sleep(monkeypatch, tmp_path):
    from tw_scenario.research import taifex

    class Response:
        status_code = 429

        def __init__(self):
            self.headers = {"Retry-After": "120"}

    class Session:
        calls = 0

        def mount(self, *args):
            pass

        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

        def close(self):
            pass

    session = Session()
    monkeypatch.setattr(taifex.requests, "Session", lambda: session)
    monkeypatch.setattr(taifex.time, "sleep", lambda _: pytest.fail("must not sleep beyond budget"))
    with pytest.raises(RuntimeError, match="Retry-After"):
        taifex.fetch_night(tmp_path, "2025-10-01")
    assert session.calls == 1
