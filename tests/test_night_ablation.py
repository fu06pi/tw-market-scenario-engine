"""Synthetic isolation checks, never market-performance evidence."""

from copy import deepcopy
from datetime import date, timedelta
from math import log1p

import pytest

from scripts.run_night_ablation import paired_interval, predict
from tw_scenario.split_forecast import Components


def samples():
    result = []
    for i in range(90):
        day = date(2025, 1, 1) + timedelta(days=i)
        night = ((i % 5) - 2) * 30
        result.append(
            {
                "date": day.isoformat(),
                "feature_latest_date": (day - timedelta(days=1)).isoformat(),
                "previous_close": 100,
                "features": [0, 0, 0, 50],
                "night_return_bps": night,
                "target": Components(g=night / 10000, b=-0.002, u=0.003, d=0.004),
            }
        )
    return result


def test_current_night_changes_augmented_gap_but_not_body_or_control():
    original = samples()
    changed = deepcopy(original)
    changed[65]["night_return_bps"] = 300
    before = predict(original, 65, "without_night")[0]
    control = predict(changed, 65, "without_night")[0]
    after = predict(original, 65, "with_night_gap_only")[0]
    augmented = predict(changed, 65, "with_night_gap_only")[0]
    assert before == control
    assert after.g != augmented.g
    assert before.b == after.b == augmented.b
    assert before.u == after.u and before.d == after.d


@pytest.mark.parametrize(
    "model", ["without_night", "with_night_gap_only", "night_carry_with_mean_body"]
)
def test_future_and_current_targets_do_not_change_current_prediction(model):
    original = samples()
    changed = deepcopy(original)
    for row in changed[65:]:
        row["target"] = Components(g=1, b=1, u=1, d=1)
    assert predict(original, 65, model) == predict(changed, 65, model)


def test_copy_baseline_has_no_fitted_night_multiplier():
    rows = samples()
    components, _ = predict(rows, 65, "night_carry_with_mean_body")
    assert components.g == pytest.approx(log1p(rows[65]["night_return_bps"] / 10000))


def test_minimum_past_and_duplicate_training_dates_rejected():
    rows = samples()
    with pytest.raises(ValueError, match="insufficient"):
        predict(rows, 59, "with_night_gap_only")
    rows[0]["date"] = rows[65]["date"]
    with pytest.raises(ValueError, match="leakage"):
        predict(rows, 65, "with_night_gap_only")


def test_paired_statistics_reject_dropped_or_duplicate_days():
    settings = {"seed": 1, "draws": 100, "block_length": 2}
    first = [{"date": str(i), "open_ae_bps": i + 1} for i in range(10)]
    half = [{"date": str(i), "open_ae_bps": (i + 1) / 2} for i in range(10)]
    stats = paired_interval(first, half, "open_ae_bps", settings)
    assert stats["improvement"] == pytest.approx(0.5)
    assert stats["interval_95"] == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="identical"):
        paired_interval(first, half[:9], "open_ae_bps", settings)
    with pytest.raises(ValueError, match="identical"):
        paired_interval(first, half + half[:1], "open_ae_bps", settings)
