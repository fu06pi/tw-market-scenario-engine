from copy import deepcopy
from math import exp

import pytest

from tw_scenario.baselines import (
    DIRECTION_BASELINES,
    PRICE_BASELINES,
    OvernightFeatures,
    direction_baseline,
    overnight_gap_baseline,
    predict_preopen_baseline,
    price_baseline,
)
from tw_scenario.models import Direction
from tw_scenario.split_forecast import Components, TrainingExample


def history():
    return [Components(g=0.001 * i, b=-0.002 * i, u=0.003, d=0.004) for i in range(1, 31)]


@pytest.mark.parametrize("name", PRICE_BASELINES)
def test_price_baselines_return_coherent_prices_in_reference_scale(name):
    result = price_baseline(name, history(), 100)
    scaled = price_baseline(name, history(), 200)
    assert result.prices.high >= max(result.prices.open, result.prices.close)
    assert result.prices.low <= min(result.prices.open, result.prices.close)
    assert scaled.prices.close == pytest.approx(result.prices.close * 2)
    assert scaled.directions == result.directions


def test_zero_return_and_previous_return_have_distinct_price_meanings():
    past = history()
    assert price_baseline("zero_return", past, 100).prices.close == 100
    previous = price_baseline("previous_return", past, 100)
    assert previous.prices.close == pytest.approx(100 * exp(past[-1].g + past[-1].b))


def test_fixed_rolling_window_ignores_older_return_shock():
    past = history()
    changed = deepcopy(past)
    changed[0] = Components(g=1, b=1, u=0.003, d=0.004)
    assert price_baseline("rolling_mean_20", past, 100) == price_baseline(
        "rolling_mean_20", changed, 100
    )
    assert (
        price_baseline("expanding_mean", past, 100).prices
        != price_baseline("expanding_mean", changed, 100).prices
    )


@pytest.mark.parametrize("name", DIRECTION_BASELINES)
def test_direction_baselines_do_not_claim_price_forecasts(name):
    result = direction_baseline(name, history())
    assert result.prices is None and result.components is None
    assert set(result.directions) == {"gap", "body", "close"}


def test_majority_tie_order_and_previous_direction_use_only_history():
    tied = [Components(g=0.01, b=0, u=0, d=0), Components(g=-0.01, b=0, u=0, d=0)]
    assert direction_baseline("historical_majority", tied).directions["close"] == Direction.UP
    assert direction_baseline("previous_direction", tied).directions["close"] == Direction.DOWN
    tied.append(Components(g=0, b=0, u=0, d=0))
    assert direction_baseline("historical_majority", tied).directions["close"] == Direction.FLAT


def test_insufficient_rolling_history_and_nonfinite_reference_rejected():
    with pytest.raises(ValueError, match="20 past"):
        price_baseline("rolling_mean_20", history()[:19], 100)
    with pytest.raises(ValueError, match="finite positive"):
        price_baseline("zero_return", history(), float("nan"))
    with pytest.raises(ValueError, match="require past"):
        direction_baseline("historical_majority", [])
    with pytest.raises(ValueError):
        OvernightFeatures(float("nan"), 0, 0)


def test_preopen_baseline_rejects_same_day_and_unavailable_labels(split_data):
    examples, request, _ = split_data
    assert (
        predict_preopen_baseline("zero_return", request, examples).prices.open
        == request.previous_close
    )
    payload = examples[0].model_dump(mode="json")
    payload["actual"]["available_at"] = "2026-10-01T09:00:00+08:00"
    late = TrainingExample.model_validate(payload)
    with pytest.raises(ValueError, match="unavailable at cutoff"):
        predict_preopen_baseline("zero_return", request, [late])
    from conftest import bar_for

    same_day = TrainingExample(inputs=request, actual=bar_for(request))
    with pytest.raises(ValueError, match="strictly precede"):
        predict_preopen_baseline("zero_return", request, [same_day])


def test_preopen_baseline_rejects_duplicate_dates_wrong_market_and_missing_night(split_data):
    examples, request, _ = split_data
    with pytest.raises(ValueError, match="duplicate"):
        predict_preopen_baseline("zero_return", request, examples + examples[:1])
    from conftest import bar_for, inputs_for

    tx = inputs_for(examples[0].inputs.trading_date, market="TX", contract="202610")
    with pytest.raises(ValueError, match="market/contract"):
        predict_preopen_baseline(
            "zero_return", request, [TrainingExample(inputs=tx, actual=bar_for(tx))]
        )
    with pytest.raises(ValueError, match="missing overnight"):
        predict_preopen_baseline("overnight_gap", request, examples)


def test_original_overnight_rule_keeps_its_open_only_threshold():
    assert overnight_gap_baseline(OvernightFeatures(1, 0, 0)) == Direction.UP
    assert overnight_gap_baseline(OvernightFeatures(-1, 0, 0)) == Direction.DOWN
    assert overnight_gap_baseline(OvernightFeatures(0, 0, 0)) == Direction.FLAT


def test_available_overnight_features_produce_only_open_direction(split_data):
    from tw_scenario.split_forecast import PreopenInputs

    examples, request, _ = split_data
    payload = request.model_dump(mode="json")
    template = dict(payload["features"][0])
    payload["features"] = [
        dict(template, name=name, value=value)
        for name, value in (
            ("taifex_night_return_pct", 1),
            ("sox_return_pct", 0),
            ("tsmc_adr_return_pct", 0),
        )
    ]
    prediction = predict_preopen_baseline(
        "overnight_gap", PreopenInputs.model_validate(payload), examples
    )
    assert prediction.directions == {"gap": Direction.UP}
    assert prediction.prices is None and prediction.information_set == "night_and_overseas"
