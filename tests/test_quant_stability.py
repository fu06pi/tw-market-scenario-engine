from tw_scenario.quant.evaluation import stability_breakdown


def _row(day, contract, predicted_close, actual_close, high, low, status="OK"):
    previous_close = 100.0
    predicted = {"open": 100.0, "high": high, "low": low, "close": predicted_close}
    actual = {"open": 100.0, "high": high, "low": low, "close": actual_close}
    row = {
        "date": day,
        "instrument": "TX",
        "contract": contract,
        "status": status,
        "previous_close": previous_close,
        "predicted": predicted,
        "actual": actual,
        "interval": {
            field: [value - 1, value + 1] for field, value in predicted.items()
        },
    }
    if status != "OK":
        row["error"] = "preserved failure"
    return row


def test_stability_breakdown_preserves_failures_and_marks_posthoc_groups():
    candidate = [
        _row("2025-12-31", "202601", 101, 100, 100.5, 99.5),
        _row("2026-01-02", "202602", 104, 102, 103, 99),
        _row("2026-01-03", "202602", 100, 100, 101, 99, status="FAIL"),
    ]
    baseline = [
        _row("2025-12-31", "202601", 102, 100, 100.5, 99.5),
        _row("2026-01-02", "202602", 106, 102, 103, 99),
        _row("2026-01-03", "202602", 100, 100, 101, 99),
    ]

    result = stability_breakdown(candidate, baseline)

    assert result["paired_days"] == 2
    assert result["candidate_failed_days"] == [
        {"date": "2026-01-03", "reason": "preserved failure"}
    ]
    assert set(result["groups"]["calendar_year"]) == {"2025", "2026"}
    assert result["groups"]["contract_rollover"]["rollover"]["paired_days"] == 1
    assert result["major_event"]["status"].startswith("NOT_EVALUATED")


def test_stability_breakdown_uses_supplied_event_labels_without_changing_rows():
    candidate = [_row("2026-01-02", "202602", 101, 100, 103, 99)]
    baseline = [_row("2026-01-02", "202602", 102, 100, 103, 99)]

    result = stability_breakdown(
        candidate,
        baseline,
        event_labels={"2026-01-02": "verified_macro_event"},
    )

    assert result["major_event"]["status"] == "EVALUATED_POST_HOC"
    assert result["major_event"]["groups"]["verified_macro_event"]["paired_days"] == 1
