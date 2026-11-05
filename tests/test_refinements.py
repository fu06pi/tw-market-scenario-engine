"""Synthetic isolation tests, never empirical accuracy evidence."""

from copy import deepcopy
from datetime import date, timedelta

import pytest

from tw_scenario.intervals import bands, calibrate, interval_score
from tw_scenario.refinements import METHODS, fit, predict
from tw_scenario.split_forecast import Components


def samples(market="TAIEX", n=140):
    result = []
    for i in range(n):
        day = date(2025, 1, 1) + timedelta(days=i)
        night = ((i % 7) - 3) * 20
        result.append({
            "date": day.isoformat(), "market": market, "previous_close": 100,
            "feature_latest_date": (day - timedelta(days=1)).isoformat(),
            "features": [i % 4 * 10, i % 5 * 20, i % 3 * 10, 50 + i % 6 * 10],
            "night_return_bps": night,
            "target": Components(g=night / 15000, b=-night / 20000, u=0.003, d=0.004),
        })
    return result


@pytest.mark.parametrize("method", METHODS)
def test_frozen_predictions_ignore_current_future_labels_and_preserve_ohlc(method):
    rows = samples()
    checkpoint = fit(rows[:120], method, "TAIEX")
    first = predict(checkpoint, rows[120])
    changed = deepcopy(rows[120])
    changed["target"] = Components(g=1, b=1, u=1, d=1)
    assert predict(checkpoint, changed) == first
    assert first[1].high >= max(first[1].open, first[1].close)
    assert first[1].low <= min(first[1].open, first[1].close)


def test_body_ablation_holds_gap_fixed_and_calibration_holds_close_fixed():
    rows = samples()
    outputs = {m: predict(fit(rows[:120], m, "TAIEX"), rows[120])
               for m in ("body_control", "body_night", "gap_calibrated")}
    assert outputs["body_control"][0].g == outputs["body_night"][0].g
    assert outputs["body_control"][0].b != outputs["body_night"][0].b
    assert outputs["gap_calibrated"][1].close == pytest.approx(outputs["body_night"][1].close)
    assert outputs["gap_calibrated"][0].g != outputs["body_night"][0].g


def test_tx_is_not_silently_given_cash_calibrator():
    rows = samples("TX_selected_monthly")
    assert predict(fit(rows[:120], "gap_calibrated", rows[0]["market"]), rows[120]) == predict(
        fit(rows[:120], "body_night", rows[0]["market"]), rows[120])


def test_training_temporal_identity_and_checkpoint_gates():
    rows = samples()
    with pytest.raises(ValueError, match="chronological"):
        fit(rows[:119] + [rows[118]], "night_residual", "TAIEX")
    with pytest.raises(ValueError, match="market"):
        fit(rows[:120], "night_residual", "TX_selected_monthly")
    with pytest.raises(ValueError, match="past checkpoint"):
        predict(fit(rows[:120], "night_residual", "TAIEX"), rows[119])
    rows[10]["features"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit(rows[:120], "night_residual", "TAIEX")


def test_previous_return_on_roll_day_uses_current_contract_lag():
    rows = samples("TX_selected_monthly")
    query = {**rows[120], "features": [50, 80, 30, 100]}
    components, prices = predict(fit(rows[:120], "previous_return", rows[0]["market"]), query)
    assert prices.open == pytest.approx(100.5)
    assert prices.close == pytest.approx(100.8)
    assert components.g != rows[119]["target"].g


def calibration_rows():
    return [{"date": str(date(2025, 1, 1) + timedelta(days=i)), "market": "TAIEX",
             "method": "night_carry", **{f"{k}_ae_bps": float(i + 1)
                                         for k in ("open", "high", "low", "close")}}
            for i in range(30)]


def test_intervals_finite_sample_rank_market_model_and_future_isolation():
    rows = calibration_rows()
    cal = calibrate(rows, "2026-01-01")
    assert cal["radius_bps"]["close"] == 25
    prices = Components(g=0, b=0, u=0.01, d=0.01).reconstruct(100)
    result = bands(cal, prices, 100, "TAIEX", "night_carry", "2026-01-01")
    assert result["close"] == {"lower": 99.75, "upper": 100.25}
    with pytest.raises(ValueError, match="mismatch"):
        bands(cal, prices, 100, "TX_selected_monthly", "night_carry", "2026-01-01")
    with pytest.raises(ValueError, match="strictly past"):
        calibrate(rows, rows[-1]["date"])
    with pytest.raises(ValueError, match="30"):
        calibrate(rows[:29], "2026-01-01")
    assert interval_score(90, 110, 100) == 20
    assert interval_score(90, 110, 120) == 120
