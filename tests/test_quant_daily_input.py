from __future__ import annotations

from datetime import date, datetime, timedelta

from tw_scenario.quant.contracts import Bar, Calendar, at
from tw_scenario.quant.daily_input import audit_preopen_dataset, data_quality_evidence
from tw_scenario.quant.forecast import Dataset

DAY = date(2026, 11, 10)


def _bar(
    instrument: str,
    trading_date: date,
    session: str = "day",
    fetched_at: datetime | None = None,
) -> Bar:
    contract = "" if instrument == "TAIEX" else "202611"
    if session == "night":
        start = at(trading_date - timedelta(days=1), "15:00")
        end = at(trading_date, "05:00")
        available = at(trading_date, "07:30")
    elif instrument == "TAIEX":
        start, end, available = (
            at(trading_date, "09:00"),
            at(trading_date, "13:30"),
            at(trading_date, "18:00"),
        )
    else:
        start, end, available = (
            at(trading_date, "08:45"),
            at(trading_date, "13:45"),
            at(trading_date, "18:00"),
        )
    return Bar(
        instrument=instrument,
        contract=contract,
        trading_date=trading_date,
        session=session,
        open=20_000,
        high=20_100,
        low=19_900,
        close=20_050,
        volume=100,
        session_start_at=start,
        session_end_at=end,
        available_at=available,
        fetched_at=fetched_at or at(DAY, "08:20"),
        availability_basis="live test observation",
        source="synthetic",
        source_hash="1" * 64,
    )


def _input(late: bool = False, current_daytime: bool = False):
    days = [DAY - timedelta(days=2), DAY - timedelta(days=1), DAY]
    days.extend(DAY + timedelta(days=offset) for offset in range(1, 11))
    calendar = Calendar(
        days=days,
        expiries={"202611": days[-1]},
        source="synthetic verified calendar",
        available_at=at(DAY - timedelta(days=1), "18:00"),
    )
    bars = []
    for historical_day in days[:2]:
        bars.extend([
            _bar("TAIEX", historical_day),
            _bar("TX", historical_day),
            _bar("TX", historical_day, "night"),
        ])
    fetched_at = at(DAY, "08:31") if late else at(DAY, "08:20")
    bars.append(_bar("TX", DAY, "night", fetched_at))
    if current_daytime:
        bars.append(_bar("TAIEX", DAY, fetched_at=fetched_at))
    manifest = {
        "dataset_id": "2" * 64,
        "parent_dataset_id": "3" * 64,
        "start": str(days[0]),
        "incremental_start": str(days[1]),
        "source_requests": [
            {"fetched_at": fetched_at.isoformat()},
            {"fetched_at": fetched_at.isoformat()},
        ],
        "official_current_night": {
            "fetched_at": fetched_at.isoformat(),
            "ohlc_mismatch_fields": [],
        },
        "completed_incremental_reconciliation": {
            "status": "PASS_RETROSPECTIVE_VALUE_AGREEMENT",
            "missing": 0,
            "mismatches": 0,
        },
    }
    return Dataset(bars, calendar), manifest


def test_preopen_input_audit_passes_only_cutoff_safe_inputs():
    data, manifest = _input()
    report = audit_preopen_dataset(data, manifest, DAY, at(DAY, "08:30"))

    assert report["status"] == "PASS"
    assert report["issues"] == []
    assert [item["session"] for item in report["required_sources"]] == [
        "day",
        "day",
        "night",
    ]
    evidence = data_quality_evidence(report)
    assert evidence.status == "PASS"
    assert evidence.report_hash == report["report_hash"]


def test_preopen_input_audit_rejects_late_fetch_and_prediction_period_daytime():
    data, manifest = _input(late=True, current_daytime=True)
    report = audit_preopen_dataset(data, manifest, DAY, at(DAY, "08:30"))

    assert report["status"] == "FAIL"
    assert "prediction-period daytime rows are present" in report["issues"]
    assert "incremental provider response was fetched after the 08:30 cutoff" in report["issues"]
    assert "source request was not captured by cutoff" in report["issues"]
    evidence = data_quality_evidence(report)
    assert evidence.status == "FAIL"
