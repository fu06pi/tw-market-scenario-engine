"""Reject misleading report comparisons, without running or selecting a model."""

from copy import deepcopy

import pytest

from scripts.prepare_academic_results import METHODS, validate_market
from scripts.run_historical_validation import score_day, summarize
from tw_scenario.split_forecast import OHLC, encode_components


def fixture():
    samples, arms = {}, {m: [] for m in METHODS}
    for i, day in enumerate(("2026-01-02", "2026-01-05")):
        actual = OHLC(open=100 + i, high=104 + i, low=98 + i, close=102 + i)
        sample = {"date": day, "previous_close": 100, "actual": actual,
                  "target": encode_components(100, actual), "features": [0, 0, 0, 80],
                  "contract": "202601", "rollover": False}
        samples[day] = sample
        predicted = OHLC(open=101 + i, high=105 + i, low=99 + i, close=103 + i)
        for method in METHODS:
            arms[method].append(score_day(sample, encode_components(100, predicted), predicted, 5))
    published = {m: summarize(rows, 2) for m, rows in arms.items()}
    return arms, samples, list(samples), published


def test_report_recomputes_prices_and_adds_point_rmse_without_fitting():
    arms, samples, dates, published = fixture()
    verified, metrics = validate_market(arms, samples, dates, published)
    assert len(verified[METHODS[0]]) == 2
    assert metrics[METHODS[0]]["rmse_points"]["close"] == pytest.approx(1)
    assert metrics[METHODS[0]]["missing_or_failed"] == 0


@pytest.mark.parametrize("change", ["drop", "duplicate", "reorder"])
def test_report_refuses_subset_or_duplicate_or_reordered_comparisons(change):
    arms, samples, dates, published = fixture()
    arms[METHODS[0]] = {"drop": arms[METHODS[0]][:1], "duplicate": arms[METHODS[0]][:1] * 2,
                        "reorder": arms[METHODS[0]][::-1]}[change]
    with pytest.raises(ValueError, match="dates"):
        validate_market(arms, samples, dates, published)


@pytest.mark.parametrize("change", ["actual", "error", "published", "source"])
def test_report_refuses_tampered_sources_daily_metrics_or_public_reference(change):
    arms, samples, dates, published = deepcopy(fixture())
    if change == "actual":
        arms[METHODS[0]][0]["actual"]["open"] += .25
    elif change == "error":
        arms[METHODS[0]][0]["close_ae_bps"] += 1
    elif change == "published":
        published[METHODS[0]]["close_mae_bps"] += 1
    else:
        samples.pop(dates[0])
    with pytest.raises(ValueError):
        validate_market(arms, samples, dates, published)
