from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from tw_scenario.quant import backtest
from tw_scenario.quant.backtest import DatasetQualityError, _select_baseline, walk_forward
from tw_scenario.quant.contracts import Bar, Calendar, Prices, at, digest
from tw_scenario.quant.evaluation import metrics
from tw_scenario.quant.quality import quality_report


def _actual(day: date, available_at) -> Bar:
    return Bar(
        instrument="TAIEX",
        contract="",
        trading_date=day,
        session="day",
        open=10_000,
        high=10_100,
        low=9_900,
        close=10_050,
        session_start_at=at(day, "08:45"),
        session_end_at=at(day, "13:45"),
        available_at=available_at,
        fetched_at=at(day + timedelta(days=7), "12:00"),
        availability_basis="synthetic fixture",
        source="synthetic-test",
        source_hash="0" * 64,
    )


def _result(expected: list[str], successes: dict[str, float]) -> dict:
    rows = []
    for day in expected:
        if day not in successes:
            rows.append({"date": day, "status": "FAILED", "error": "synthetic"})
            continue
        error = successes[day]
        actual = {name: 10_000.0 for name in ("open", "high", "low", "close")}
        predicted = {name: 10_000.0 + error for name in actual}
        rows.append({
            "date": day,
            "status": "OK",
            "previous_close": 10_000.0,
            "actual": actual,
            "predicted": predicted,
            "interval": None,
        })
    return {"rows": rows, "metrics": metrics(rows)}


def test_quality_report_fails_empty_requested_coverage_and_manifest_failures():
    start, end = date(2026, 1, 1), date(2026, 3, 31)
    calendar = Calendar(
        days=[],
        expiries={},
        source="synthetic-test",
        available_at=at(start),
    )
    manifest = {
        "start": str(start),
        "end": str(end),
        "rows": 0,
        "failures": [{"source": "TAIFEX", "error": "synthetic failure"}],
    }

    report = quality_report([], calendar, start, end, manifest=manifest)

    assert report["status"] == "FAIL"
    assert {issue["issue"] for issue in report["issues"]} == {"no_bars_loaded"}
    assert report["requested_coverage"]["missing_months"] == [
        "2026-01",
        "2026-02",
        "2026-03",
    ]
    assert any(
        issue["issue"] == "manifest_source_failures"
        for issue in report["manifest_issues"]
    )


def test_quality_report_detects_month_missing_from_observed_only_calendar():
    start, end = date(2026, 1, 1), date(2026, 3, 31)
    calendar = Calendar(
        days=[date(2026, 1, 5), date(2026, 3, 2)],
        expiries={},
        source="synthetic-test",
        available_at=at(start),
    )

    report = quality_report([], calendar, start, end)

    assert report["status"] == "FAIL"
    assert "2026-02" in report["requested_coverage"]["missing_months"]
    assert any(
        issue["issue"] == "unexplained_exchange_calendar_gap"
        for issue in report["calendar_gaps"]
    )


def test_quality_report_accepts_only_manifest_verified_no_night_session():
    days = [date(2026, 1, 1) + timedelta(days=index) for index in range(20)]
    target = days[4]
    calendar = Calendar(
        days=days,
        expiries={"202601": days[14], "202602": days[19]},
        source="synthetic-test",
        available_at=at(days[0]),
    )
    taiex = _actual(target, at(target, "18:00"))
    tx_day = Bar(
        instrument="TX",
        contract="202601",
        trading_date=target,
        session="day",
        open=20_000,
        high=20_100,
        low=19_900,
        close=20_050,
        session_start_at=at(target, "08:45"),
        session_end_at=at(target, "13:45"),
        available_at=at(target, "18:00"),
        fetched_at=at(target + timedelta(days=1)),
        availability_basis="ASSUMPTION: synthetic fixture",
        source="synthetic-test",
        source_hash="1" * 64,
    )
    base_manifest = {
        "start": str(target),
        "end": str(target),
        "rows": 2,
        "failures": [],
    }

    rejected = quality_report([taiex, tx_day], calendar, target, target, base_manifest)
    assert rejected["status"] == "FAIL"
    assert rejected["missing"] == [["TX", "202601", "night", str(target)]]

    certified = {
        **base_manifest,
        "verified_session_exceptions": [{
            "instrument": "TX",
            "session": "night",
            "trading_date": str(target),
            "basis": "two independent synthetic sources confirm no market night session",
        }],
    }
    accepted = quality_report([taiex, tx_day], calendar, target, target, certified)
    assert accepted["status"] == "PASS_RETROSPECTIVE"
    assert accepted["missing"] == []
    assert accepted["verified_session_exceptions"] == [["TX", "night", str(target)]]


def test_walk_forward_only_uses_residual_after_label_is_available(monkeypatch):
    days = [date(2026, 1, day) for day in (5, 6, 7)]
    actuals = {
        days[0]: _actual(days[0], at(days[2], "08:00")),
        days[1]: _actual(days[1], at(days[1], "18:00")),
        days[2]: _actual(days[2], at(days[2], "18:00")),
    }

    class Data:
        calendar = type("CalendarStub", (), {"days": days})()

        @staticmethod
        def actual(_instrument, day):
            return actuals[day]

        @staticmethod
        def contract(_instrument, _day):
            return ""

    residual_counts = []

    def fake_predict(_data, _instrument, _day, _model, _representation, _config):
        return Prices(open=10_000, high=10_000, low=10_000, close=10_000), np.zeros(4), 10_000, []

    def fake_interval(_target, _previous, _representation, residuals, _minimum):
        residual_counts.append(len(residuals))

    monkeypatch.setattr(backtest, "predict", fake_predict)
    monkeypatch.setattr(backtest, "calibrated_interval", fake_interval)

    rows = walk_forward(
        Data(),
        "TAIEX",
        "zero",
        "relative",
        days[0],
        days[-1],
        {"calibration_window": 10, "min_calibration": 0, "cutoff": "08:30"},
    )

    assert [row["status"] for row in rows] == ["OK", "OK", "OK"], [
        row.get("error") for row in rows
    ]
    assert residual_counts == [0, 0, 2]


def test_walk_forward_limits_pre_output_calibration_warmup(monkeypatch):
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(300)]

    class Data:
        calendar = type("CalendarStub", (), {"days": days})()

        @staticmethod
        def actual(_instrument, day):
            return _actual(day, at(day, "18:00"))

        @staticmethod
        def contract(_instrument, _day):
            return ""

    predicted_days = []

    def fake_predict(_data, _instrument, day, _model, _representation, _config):
        predicted_days.append(day)
        prices = Prices(open=10_000, high=10_000, low=10_000, close=10_000)
        return prices, np.zeros(4), 10_000, []

    monkeypatch.setattr(backtest, "predict", fake_predict)
    monkeypatch.setattr(backtest, "calibrated_interval", lambda *_args: None)

    rows = walk_forward(
        Data(),
        "TAIEX",
        "zero",
        "relative",
        days[250],
        days[251],
        {"calibration_window": 10, "min_history": 5, "cutoff": "08:30"},
    )

    assert len(predicted_days) == 17
    assert predicted_days[0] == days[235]
    assert [row["date"] for row in rows] == [str(days[250]), str(days[251])]


def test_baseline_selection_rejects_low_coverage_winner():
    expected = [f"2026-01-{day:02d}" for day in range(5, 9)]
    validation = {
        "TAIEX/relative/sparse": _result(expected, {expected[0]: 0.0}),
        "TAIEX/relative/complete": _result(
            expected, {day: 5.0 for day in expected}
        ),
    }

    selected, audit = _select_baseline(validation, "TAIEX", expected, 0.75, 5.0)

    assert selected["model"] == "complete"
    assert not audit["TAIEX/relative/sparse"]["eligible"]
    assert audit["TAIEX/relative/sparse"]["ineligible_reason"] == (
        "below_min_validation_coverage"
    )


def test_baseline_selection_scores_candidates_on_same_paired_dates():
    expected = [f"2026-01-{day:02d}" for day in range(5, 9)]
    validation = {
        "TAIEX/relative/a": _result(
            expected,
            {expected[0]: 0.0, expected[1]: 10.0, expected[2]: 10.0},
        ),
        "TAIEX/relative/b": _result(
            expected,
            {expected[1]: 1.0, expected[2]: 1.0, expected[3]: 100.0},
        ),
    }

    selected, audit = _select_baseline(validation, "TAIEX", expected, 0.5, 5.0)

    assert selected["model"] == "b"
    assert selected["validation_paired_days"] == 2
    assert audit["TAIEX/relative/a"]["paired_close_mae"] == 10.0
    assert audit["TAIEX/relative/b"]["paired_close_mae"] == 1.0


def test_backtest_stops_at_manifest_and_data_quality_gate(tmp_path, monkeypatch):
    start, end = date(2026, 1, 1), date(2026, 1, 31)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "bars.jsonl").write_text("", encoding="utf-8")
    calendar = Calendar(
        days=[],
        expiries={},
        source="synthetic-test",
        available_at=at(start),
    )
    (dataset_dir / "calendar.json").write_text(
        calendar.model_dump_json(), encoding="utf-8"
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps({
            "dataset_id": digest([]),
            "start": str(start),
            "end": str(end),
            "rows": 0,
            "failures": [{"source": "TAIFEX", "error": "synthetic"}],
        }),
        encoding="utf-8",
    )
    config_path = tmp_path / "config" / "quant" / "research.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"train_start": str(start), "test_end": str(end)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backtest,
        "walk_forward",
        lambda *_args, **_kwargs: pytest.fail("model evaluation ran before DQ gate"),
    )

    with pytest.raises(DatasetQualityError, match="dataset quality gate failed"):
        backtest.run_baseline_suite(dataset_dir, config_path, tmp_path / "output")
