from datetime import date
from math import exp

import pytest
from conftest import bar_for, inputs_for

from tw_scenario.policy import ScoringPolicy
from tw_scenario.split_forecast import (
    OHLC,
    Components,
    PreopenInputs,
    SplitForecast,
    TrainingExample,
    encode_components,
    evaluate_split,
    forecast_split,
)


def test_open_can_rise_while_close_falls(split_data):
    forecast = forecast_split(*split_data)
    assert forecast.prices.open > forecast.inputs.previous_close > forecast.prices.close
    assert forecast.components.g > 0 > forecast.components.b
    assert set(forecast.head_parameters) == {"g", "b"}
    assert forecast.prices.high >= max(forecast.prices.open, forecast.prices.close)
    assert forecast.prices.low <= min(forecast.prices.open, forecast.prices.close)


def test_heads_respond_to_separate_feature_sets(split_data):
    examples, request, config = split_data
    first = forecast_split(examples, request, config)
    other = inputs_for(request.trading_date, night=-5, body=-4)
    second = forecast_split(examples, other, config)
    assert first.components.g != second.components.g
    assert first.components.b == second.components.b
    assert first.training_digest == second.training_digest


@pytest.mark.parametrize(
    "g,b,u,d", [(0, 0, 0, 0), (0.03, -0.06, 0.01, 0.02), (-0.04, 0.07, 0.03, 0)]
)
def test_log_components_roundtrip(g, b, u, d):
    components = Components(g=g, b=b, u=u, d=d)
    bar = components.reconstruct(100)
    actual = encode_components(100, bar)
    assert actual.model_dump() == pytest.approx(components.model_dump(), abs=1e-14)
    assert bar.close == pytest.approx(100 * exp(g + b))


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_cutoff_at", "2026-10-01T08:30:01+08:00"),
        ("generated_at", "2026-10-01T08:40:00+08:00"),
        ("generated_at", "2026-10-01T08:29:59+08:00"),
        ("data_cutoff_at", "2026-10-01T08:30:00"),
        ("previous_close_date", "2026-10-01"),
        ("previous_close_available_at", "2026-10-01T08:31:00+08:00"),
        ("previous_close", 0),
        ("previous_close", float("nan")),
    ],
)
def test_invalid_time_or_reference_rejected(split_data, field, value):
    payload = split_data[1].model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValueError):
        PreopenInputs.model_validate(payload)


@pytest.mark.parametrize(
    "mutation", ["late_available", "unfinished", "same_day_actual", "duplicate"]
)
def test_feature_leakage_rejected(split_data, mutation):
    payload = split_data[1].model_dump(mode="json")
    feature = payload["features"][0]
    if mutation == "late_available":
        feature["available_at"] = "2026-10-01T08:31:00+08:00"
    elif mutation == "unfinished":
        feature["observed_at"] = "2026-10-01T09:00:00+08:00"
    elif mutation == "same_day_actual":
        feature.update(kind="historical", observed_at="2026-10-01T08:00:00+08:00")
    else:
        payload["features"].append(dict(feature))
    with pytest.raises(ValueError):
        PreopenInputs.model_validate(payload)


def test_utc_cutoff_is_aligned_to_taipei(split_data):
    payload = split_data[1].model_dump(mode="json")
    payload["data_cutoff_at"] = "2026-10-01T00:30:00+00:00"
    PreopenInputs.model_validate(payload)


def test_same_day_training_and_unavailable_outcomes_rejected(split_data):
    examples, request, config = split_data
    same_day = TrainingExample(inputs=request, actual=bar_for(request))
    with pytest.raises(ValueError, match="precede"):
        forecast_split(examples + [same_day], request, config)
    payload = examples[0].model_dump(mode="json")
    payload["actual"]["available_at"] = "2026-10-01T09:00:00+08:00"
    changed = TrainingExample.model_validate(payload)
    with pytest.raises(ValueError, match="outcome was not yet available"):
        forecast_split([changed, *examples[1:]], request, config)


def test_duplicate_dates_and_insufficient_history_rejected(split_data):
    examples, request, config = split_data
    with pytest.raises(ValueError, match="duplicate"):
        forecast_split(examples + [examples[0]], request, config)
    with pytest.raises(ValueError, match="training samples"):
        forecast_split(examples[:2], request, config)


def test_future_training_feature_cannot_be_hidden_in_a_past_row(split_data):
    payload = split_data[0][0].model_dump(mode="json")
    payload["inputs"]["features"][0]["available_at"] = "2026-09-02T05:00:00+08:00"
    with pytest.raises(ValueError, match="unavailable at cutoff"):
        TrainingExample.model_validate(payload)


def test_tx_reference_requires_same_contract_and_market():
    request = inputs_for(date(2026, 10, 1), market="TX", contract="202610")
    payload = request.model_dump(mode="json")
    payload["previous_close_contract"] = "202609"
    with pytest.raises(ValueError, match="same-contract"):
        PreopenInputs.model_validate(payload)
    payload = request.model_dump(mode="json")
    payload["market"] = "TAIEX"
    with pytest.raises(ValueError):
        PreopenInputs.model_validate(payload)


def test_market_and_feature_version_isolation(split_data):
    examples, request, config = split_data
    tx = inputs_for(request.trading_date, market="TX", contract="202610")
    with pytest.raises(ValueError, match="market independently"):
        forecast_split(examples, tx, config)
    payload = request.model_dump(mode="json")
    payload["feature_version"] = "new-unmatched-version"
    with pytest.raises(ValueError, match="feature versions"):
        forecast_split(examples, PreopenInputs.model_validate(payload), config)


def test_nonfinite_incoherent_prices_and_negative_wicks_rejected():
    with pytest.raises(ValueError):
        OHLC(open=100, high=99, low=98, close=100)
    with pytest.raises(ValueError):
        Components(g=0, b=0, u=-0.01, d=0)
    with pytest.raises(ValueError):
        Components(g=float("inf"), b=0, u=0, d=0)


def test_missing_head_feature_and_unexpected_actual_input_rejected(split_data):
    examples, request, config = split_data
    payload = request.model_dump(mode="json")
    payload["features"] = payload["features"][1:]
    with pytest.raises(ValueError, match="missing features"):
        forecast_split(examples, PreopenInputs.model_validate(payload), config)
    payload = request.model_dump(mode="json")
    payload["actual_open"] = 123
    with pytest.raises(ValueError):
        PreopenInputs.model_validate(payload)


def test_evaluation_separates_gap_body_close_and_checks_identity(split_data):
    forecast = forecast_split(*split_data)
    actual = bar_for(forecast.inputs, forecast.components.g, forecast.components.b)
    scores = evaluate_split(forecast, actual)
    for target in ("open_gap", "session_body", "close_return"):
        assert scores[f"{target}_mae_bps"] == pytest.approx(0, abs=1e-9)
        assert scores[f"{target}_direction_hit"] == 1
    wrong = bar_for(inputs_for(date(2026, 10, 2)))
    with pytest.raises(ValueError, match="trading_date"):
        evaluate_split(forecast, wrong)


def test_snapshot_cannot_disagree_with_its_components(split_data):
    payload = forecast_split(*split_data).model_dump(mode="json")
    payload["prices"]["high"] *= 2
    with pytest.raises(ValueError, match="disagree"):
        SplitForecast.model_validate(payload)


def test_frozen_scoring_policy_cannot_be_changed_after_forecast(split_data):
    forecast = forecast_split(*split_data)
    with pytest.raises(ValueError, match="policy frozen"):
        evaluate_split(forecast, bar_for(forecast.inputs), ScoringPolicy(flat_threshold_bps=10))


def test_tx_rollover_uses_each_rows_own_contract():
    inputs = inputs_for(date(2026, 10, 1), market="TX", contract="202610")
    payload = TrainingExample(inputs=inputs, actual=bar_for(inputs)).model_dump(mode="json")
    payload["actual"]["contract"] = "202611"
    with pytest.raises(ValueError, match="contract do not match"):
        TrainingExample.model_validate(payload)
