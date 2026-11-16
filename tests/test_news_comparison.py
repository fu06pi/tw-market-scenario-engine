"""Synthetic direction experiment, not historical or forward news performance."""

from copy import deepcopy
from datetime import date

import pytest

from tests.test_morning_adapter import actuals, paired
from tests.test_shadow import stamp
from tw_scenario.news_comparison import evaluate_pairs, freeze_decision


def payload(forecast):
    return {"forecast_hash": forecast["snapshot_hash"], "market": "TAIEX", "contract": None,
            "quantitative_method": "night_carry", "direction": "down", "model_version": "synthetic-only",
            "prompt_sha256": "a" * 64, "rationale": "synthetic research decision",
            "news": [{"news_id": "fixture-1", "source_url": "https://example.com/synthetic",
                      "published_at": "2026-11-05T07:00:00+08:00", "fetched_at": "2026-11-05T08:20:00+08:00",
                      "evidence_text": "Synthetic news, never a real market source", "availability_basis": "synthetic"}]}


def test_direction_pair_does_not_change_prices_and_missing_b_is_retained(tmp_path):
    forecast = paired(); before = deepcopy(forecast)
    frozen = freeze_decision(forecast, payload(forecast), stamp(date(2026, 11, 5), 8, 26), tmp_path / "b.json")
    result = evaluate_pairs([forecast], [frozen], actuals(forecast))
    assert forecast == before and not result["price_predictions_modified"]
    assert result["expected_pairs"] == 2 and result["paired_n"] == 1
    assert len(result["missing"]) == 1
    assert result["groups"][0]["block5_interval_95"] is None
    assert not result["formal_accuracy_eligible"]


def test_no_news_late_news_late_generation_or_new_prices_cannot_pass(tmp_path):
    forecast = paired()
    for change in ("empty", "late_news", "new_prices", "backdated"):
        p = payload(forecast)
        if change == "empty": p["news"] = []
        if change == "late_news": p["news"][0]["fetched_at"] = "2026-11-05T08:31:00+08:00"
        if change == "new_prices": p["close"] = 100
        if change == "backdated": p["generated_at"] = "2026-11-05T08:00:00+08:00"
        with pytest.raises(ValueError):
            freeze_decision(forecast, p, stamp(date(2026, 11, 5), 8, 35), tmp_path / f"{change}.json")
    with pytest.raises(ValueError, match="08:40"):
        freeze_decision(forecast, payload(forecast), stamp(date(2026, 11, 5), 9), tmp_path / "late.json")


def test_duplicate_b_decisions_cannot_be_selected_after_outcomes(tmp_path):
    forecast = paired()
    b = freeze_decision(forecast, payload(forecast), stamp(date(2026, 11, 5), 8, 26), tmp_path / "b.json")
    result = evaluate_pairs([forecast], [b, b], actuals(forecast))
    assert result["paired_n"] == 0 and result["failures"]
