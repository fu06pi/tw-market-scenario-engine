from copy import deepcopy
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from tw_scenario.quant.head_blend import causal_blend_rows

CONFIG = {
    "version": "causal-head-baseline-blend-v1",
    "candidate_context": 64,
    "window": 3,
    "minimum": 2,
    "alpha_grid": [0.0, 0.5, 1.0],
    "objective": "prior_published_close_mae",
    "update_policy": "previous_labels_available_by_0830_only",
}


def _rows():
    candidate, baseline, times = [], [], {}
    timezone = ZoneInfo("Asia/Taipei")
    for index in range(5):
        day = date(2026, 1, 5) + timedelta(days=index)
        actual_close = 102.0 + index
        common = {
            "date": str(day),
            "instrument": "TAIEX",
            "contract": "",
            "status": "OK",
            "previous_close": 100.0,
            "actual": {
                "open": actual_close - 1,
                "high": actual_close + 1,
                "low": actual_close - 2,
                "close": actual_close,
            },
        }
        candidate.append(
            {
                **deepcopy(common),
                "predicted": {
                    "open": actual_close - 1,
                    "high": actual_close + 1,
                    "low": actual_close - 2,
                    "close": actual_close,
                },
                "interval": {
                    field: [common["actual"][field] - 1, common["actual"][field] + 1]
                    for field in ("open", "high", "low", "close")
                },
            }
        )
        baseline.append(
            {
                **deepcopy(common),
                "predicted": {
                    "open": actual_close - 3,
                    "high": actual_close - 1,
                    "low": actual_close - 4,
                    "close": actual_close - 2,
                },
                "interval": {
                    field: [common["actual"][field] - 3, common["actual"][field] + 3]
                    for field in ("open", "high", "low", "close")
                },
            }
        )
        times[str(day)] = datetime.combine(day, time(18), tzinfo=timezone)
    return candidate, baseline, times


def test_causal_blend_warms_up_on_baseline_then_uses_only_prior_labels():
    candidate, baseline, times = _rows()
    rows = causal_blend_rows(candidate, baseline, times, CONFIG)
    assert [row["blend"]["alpha"] for row in rows[:2]] == [0.0, 0.0]
    assert rows[2]["blend"]["alpha"] == 1.0
    assert rows[2]["blend"]["samples"] == 2
    assert rows[2]["predicted"] == candidate[2]["predicted"]

    changed = deepcopy(candidate)
    changed[-1]["actual"]["close"] = 10000.0
    changed_baseline = deepcopy(baseline)
    changed_baseline[-1]["actual"]["close"] = 10000.0
    counterfactual = causal_blend_rows(changed, changed_baseline, times, CONFIG)
    assert [row["predicted"] for row in counterfactual[:-1]] == [
        row["predicted"] for row in rows[:-1]
    ]
    assert counterfactual[-1]["blend"] == rows[-1]["blend"]


def test_causal_blend_rejects_misaligned_dates_and_bad_availability():
    candidate, baseline, times = _rows()
    baseline[1]["date"] = "2026-02-01"
    with pytest.raises(ValueError, match="identical unique chronological"):
        causal_blend_rows(candidate, baseline, times, CONFIG)

    candidate, baseline, times = _rows()
    times[candidate[0]["date"]] = datetime.combine(
        date.fromisoformat(candidate[0]["date"]), time(8), tzinfo=ZoneInfo("Asia/Taipei")
    )
    with pytest.raises(ValueError, match="after forecast cutoff"):
        causal_blend_rows(candidate, baseline, times, CONFIG)
