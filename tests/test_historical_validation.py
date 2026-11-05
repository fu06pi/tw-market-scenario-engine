"""Research boundary checks using synthetic bars, not performance evidence."""

from copy import deepcopy
from datetime import date, timedelta

import pytest

from scripts.run_historical_validation import bootstrap_improvement, build_samples, predict_day


def history():
    rows = []
    previous = 100.0
    for i in range(95):
        opening = previous * (1 + ((i % 5) - 2) * 0.002)
        closing = opening * (1 + ((i % 7) - 3) * 0.003)
        rows.append(
            {
                "market": "TAIEX",
                "contract": None,
                "session": "day",
                "trading_date": (date(2025, 1, 1) + timedelta(days=i)).isoformat(),
                "prices": {
                    "open": opening,
                    "close": closing,
                    "high": max(opening, closing) * 1.005,
                    "low": min(opening, closing) * 0.995,
                },
            }
        )
        previous = closing
    return rows


@pytest.mark.parametrize(
    "model", ["split_ridge", "direct_close_ridge", "expanding_mean", "zero_return"]
)
def test_future_prices_cannot_change_earlier_forecast(model):
    original = history()
    changed = deepcopy(original)
    for row in changed[75:]:
        row["prices"] = {k: v * 3 for k, v in row["prices"].items()}
    a = predict_day(build_samples(original), 65, model, 1)
    b = predict_day(build_samples(changed), 65, model, 1)
    assert a == b


def test_forecast_cannot_use_its_own_target():
    original = history()
    changed = deepcopy(original)
    target_index = 65
    # Sample index 65 corresponds to raw bar 67 after two lag-building rows.
    changed[target_index + 2]["prices"]["close"] *= 1.001
    first, second = build_samples(original), build_samples(changed)
    assert first[target_index]["target"] != second[target_index]["target"]
    assert predict_day(first, target_index, "split_ridge", 1) == predict_day(
        second, target_index, "split_ridge", 1
    )


def test_duplicate_dates_and_contract_mixing_rejected():
    rows = history()
    with pytest.raises(ValueError, match="duplicate"):
        build_samples(rows + [rows[0]])
    rows[-1]["contract"] = "different-contract"
    with pytest.raises(ValueError, match="one market"):
        build_samples(rows)


def test_past_training_minimum_and_positive_alpha():
    samples = build_samples(history())
    with pytest.raises(ValueError, match="insufficient"):
        predict_day(samples, 59, "split_ridge", 1)
    with pytest.raises(ValueError, match="positive"):
        predict_day(samples, 65, "split_ridge", -1)


def test_paired_bootstrap_preserves_identical_errors():
    rows = [{"date": str(i), "close_ae_bps": i + 1} for i in range(20)]
    result = bootstrap_improvement(rows, rows, {"block_length": 5, "draws": 100, "seed": 3})
    assert result["paired_n"] == 20
    assert result["improvement"] == 0
    assert result["time_block_95pct_interval"] == [0, 0]
