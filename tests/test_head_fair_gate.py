from copy import deepcopy

import pytest

from tw_scenario.quant.head_fair_gate import fixed_comparator_rows

CONFIG = {
    "version": "fixed-point-comparator-v1",
    "primary_model": "night_regression",
    "fallback_model": "range",
    "representation": "gbud",
    "selection_policy": "predeclared_primary_before_evaluation",
    "failure_policy": "same_date_fallback_preserve_primary_error",
}


def _row(day: str, model: str) -> dict:
    return {
        "date": day,
        "instrument": "TAIEX",
        "contract": "",
        "status": "OK",
        "previous_close": 100.0,
        "predicted": {field: 100.0 for field in ("open", "high", "low", "close")},
        "actual": {field: 101.0 for field in ("open", "high", "low", "close")},
        "interval": {field: [99.0, 102.0] for field in ("open", "high", "low", "close")},
        "test_model": model,
    }


def test_fixed_comparator_uses_primary_then_same_date_fallback():
    primary = [_row("2026-01-05", "primary"), _row("2026-01-06", "primary")]
    fallback = [_row("2026-01-05", "fallback"), _row("2026-01-06", "fallback")]
    primary[1] = {
        "date": "2026-01-06",
        "instrument": "TAIEX",
        "contract": "",
        "status": "FAILED",
        "error": "invalid primary price",
    }

    rows = fixed_comparator_rows(primary, fallback, CONFIG)

    assert rows[0]["test_model"] == "primary"
    assert rows[0]["comparator"] == {
        "model": "night_regression",
        "fallback_used": False,
    }
    assert rows[1]["test_model"] == "fallback"
    assert rows[1]["comparator"] == {
        "model": "range",
        "fallback_used": True,
        "primary_error": "invalid primary price",
    }


def test_fixed_comparator_preserves_double_failure_and_rejects_misalignment():
    primary = [_row("2026-01-05", "primary")]
    fallback = [_row("2026-01-05", "fallback")]
    primary[0].update(status="FAILED", error="primary failed")
    fallback[0].update(status="FAILED", error="fallback failed")

    rows = fixed_comparator_rows(primary, fallback, CONFIG)
    assert rows[0]["status"] == "FAILED"
    assert rows[0]["comparator"]["primary_error"] == "primary failed"
    assert rows[0]["comparator"]["fallback_error"] == "fallback failed"

    changed = deepcopy(fallback)
    changed[0]["date"] = "2026-01-06"
    with pytest.raises(ValueError, match="identical unique ordered dates"):
        fixed_comparator_rows(primary, changed, CONFIG)
