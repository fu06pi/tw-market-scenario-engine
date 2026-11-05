import pytest

from scripts.compare_simple_baselines import (
    align_reference,
    complete_comparison,
    direction_summary,
    select_complete,
)
from scripts.run_historical_validation import build_samples
from tests.test_historical_validation import history


def test_reference_alignment_rejects_missing_duplicate_and_wrong_actuals():
    samples = build_samples(history())
    indices = [65, 66]
    rows = [
        {
            "date": samples[i]["date"],
            "actual": samples[i]["actual"].model_dump(),
            "predicted": samples[i]["actual"].model_dump(),
        }
        for i in indices
    ]
    assert set(align_reference(rows, samples, indices)) == {samples[i]["date"] for i in indices}
    with pytest.raises(ValueError, match="exactly"):
        align_reference(rows[:1], samples, indices)
    with pytest.raises(ValueError, match="exactly"):
        align_reference(rows + rows[:1], samples, indices)
    rows[0]["actual"]["high"] *= 2
    with pytest.raises(ValueError, match="disagree"):
        align_reference(rows, samples, indices)


def test_selection_excludes_incomplete_and_failed_favorable_results():
    rows = {
        "incomplete": {
            "metrics": {"expected": 10, "scored": 1, "close_mae_bps": 0},
            "failures": [],
        },
        "failed": {
            "metrics": {"expected": 10, "scored": 10, "close_mae_bps": 0},
            "failures": [{"date": "one failed sample"}],
        },
        "complete": {
            "metrics": {"expected": 10, "scored": 10, "close_mae_bps": 20},
            "failures": [],
        },
    }
    assert select_complete(rows, "close_mae_bps") == "complete"
    with pytest.raises(ValueError, match="no complete"):
        select_complete({"incomplete": rows["incomplete"]}, "close_mae_bps")


def test_validation_direction_selection_and_ties_are_deterministic():
    rows = {
        name: {"metrics": {"expected": 4, "scored": 4, "close_accuracy": hit}, "failures": []}
        for name, hit in [("first", 0.75), ("second", 0.75), ("third", 0.25)]
    }
    assert select_complete(rows, "close_accuracy", maximize=True) == "first"


def test_direction_only_report_does_not_invent_mae_or_hide_missing_days():
    rows = [
        {
            f"{target}_{kind}": value
            for target in ("gap", "body", "close")
            for kind, value in [
                ("hit", 1),
                ("direction_predicted", "up"),
                ("direction_actual", "up"),
            ]
        }
    ]
    result = direction_summary(rows, 2)
    assert result["scored"] == 1 and result["expected"] == 2
    assert result["coverage_of_available_rows"] == 0.5
    assert result["price_metrics"] is None
    assert "close_mae_bps" not in result


@pytest.mark.parametrize(
    "baseline",
    [
        {"metrics": {"expected": 10, "scored": 9, "close_mae_bps": 100}, "failures": []},
        {"metrics": {"expected": 9, "scored": 9, "close_mae_bps": 100}, "failures": []},
        {"metrics": {"expected": 10, "scored": 10, "close_mae_bps": 0}, "failures": []},
    ],
)
def test_no_improvement_claim_for_incomplete_unpaired_or_zero_error_baseline(baseline):
    candidate = {"metrics": {"expected": 10, "scored": 10, "close_mae_bps": 90}, "failures": []}
    assert complete_comparison(candidate, baseline) is None
