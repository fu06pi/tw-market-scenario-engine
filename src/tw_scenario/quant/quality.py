from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from itertools import pairwise

from .contracts import Bar, Calendar

MAX_UNEXPLAINED_CALENDAR_GAP_DAYS = 14


def _months(start: date, end: date) -> list[str]:
    month = start.replace(day=1)
    result = []
    while month <= end:
        result.append(month.strftime("%Y-%m"))
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result


def _manifest_checks(
    manifest: dict | None, bars: list[Bar], start: date, end: date
) -> list[dict]:
    if manifest is None:
        return []
    issues = []
    failures = manifest.get("failures", [])
    if not isinstance(failures, list):
        issues.append({"issue": "manifest_failures_invalid", "value": repr(failures)})
    elif failures:
        issues.append({"issue": "manifest_source_failures", "count": len(failures)})
    if manifest.get("rows") != len(bars):
        issues.append({
            "issue": "manifest_row_count_mismatch",
            "manifest_rows": manifest.get("rows"),
            "loaded_rows": len(bars),
        })
    try:
        declared_start = date.fromisoformat(str(manifest["start"]))
        declared_end = date.fromisoformat(str(manifest["end"]))
    except (KeyError, TypeError, ValueError):
        issues.append({"issue": "manifest_requested_coverage_invalid"})
    else:
        if declared_start > start or declared_end < end:
            issues.append({
                "issue": "manifest_requested_coverage_mismatch",
                "manifest": [str(declared_start), str(declared_end)],
                "requested": [str(start), str(end)],
            })
    return issues


def _verified_session_exceptions(manifest: dict | None) -> tuple[set[tuple[str, str, str]], list[dict]]:
    if manifest is None:
        return set(), []
    raw = manifest.get("verified_session_exceptions", [])
    if not isinstance(raw, list):
        return set(), [{"issue": "verified_session_exceptions_invalid"}]
    result = set()
    issues = []
    for item in raw:
        if not isinstance(item, dict):
            issues.append({"issue": "verified_session_exception_invalid", "value": repr(item)})
            continue
        instrument = item.get("instrument")
        session = item.get("session")
        trading_date = item.get("trading_date")
        basis = item.get("basis")
        try:
            parsed_date = date.fromisoformat(str(trading_date))
        except ValueError:
            parsed_date = None
        if (
            instrument != "TX"
            or session != "night"
            or parsed_date is None
            or not isinstance(basis, str)
            or not basis.strip()
        ):
            issues.append({"issue": "verified_session_exception_invalid", "value": item})
            continue
        key = (instrument, session, str(parsed_date))
        if key in result:
            issues.append({"issue": "verified_session_exception_duplicate", "key": list(key)})
        result.add(key)
    return result, issues


def quality_report(
    bars: list[Bar],
    calendar: Calendar,
    start: date,
    end: date,
    manifest: dict | None = None,
) -> dict:
    if start > end:
        raise ValueError("quality-report start must not be after end")

    keys = [(b.instrument, b.contract, b.session, str(b.trading_date)) for b in bars]
    duplicates = [list(key) for key, count in Counter(keys).items() if count > 1]
    issues = []
    if not bars:
        issues.append({"issue": "no_bars_loaded"})

    for bar in bars:
        if not 1000 <= bar.low <= bar.high <= 200000:
            issues.append({
                "date": str(bar.trading_date),
                "contract": bar.contract,
                "issue": "price_scale",
            })
        if bar.trading_date not in calendar.days:
            issues.append({"date": str(bar.trading_date), "issue": "non_exchange_day"})

    requested_days = [day for day in calendar.days if start <= day <= end]
    requested_months = _months(start, end)
    observed_months = {day.strftime("%Y-%m") for day in requested_days}
    missing_months = [month for month in requested_months if month not in observed_months]
    calendar_gaps = [
        {"issue": "calendar_month_missing", "month": month} for month in missing_months
    ]
    if not requested_days:
        calendar_gaps.append({
            "issue": "no_exchange_calendar_days_in_requested_range",
            "requested": [str(start), str(end)],
        })
    for previous, current in pairwise(requested_days):
        gap_days = (current - previous).days
        if gap_days > MAX_UNEXPLAINED_CALENDAR_GAP_DAYS:
            calendar_gaps.append({
                "issue": "unexplained_exchange_calendar_gap",
                "after": str(previous),
                "before": str(current),
                "calendar_days": gap_days - 1,
            })

    verified_exceptions, exception_issues = _verified_session_exceptions(manifest)
    expected = []
    keyset = set(keys)
    for day in requested_days:
        expected.append(("TAIEX", "", "day", str(day)))
        try:
            contract = calendar.contract_for(day)
        except ValueError as exc:
            issues.append({
                "date": str(day),
                "issue": "roll_calendar_coverage",
                "detail": str(exc),
            })
        else:
            expected.append(("TX", contract, "day", str(day)))
            if ("TX", "night", str(day)) not in verified_exceptions:
                expected.append(("TX", contract, "night", str(day)))
    missing = [list(key) for key in expected if key not in keyset]
    assumptions = sum("ASSUMPTION" in bar.availability_basis for bar in bars)
    manifest_issues = _manifest_checks(manifest, bars, start, end) + exception_issues
    for instrument, session, trading_date in verified_exceptions:
        if any(
            key[0] == instrument and key[2] == session and key[3] == trading_date
            for key in keyset
        ):
            manifest_issues.append({
                "issue": "verified_session_exception_has_observation",
                "key": [instrument, session, trading_date],
            })
    blockers = duplicates or missing or issues or calendar_gaps or manifest_issues

    return {
        "status": "FAIL" if blockers else "PASS_RETROSPECTIVE",
        "rows": len(bars),
        "expected_rows": len(expected),
        "missing": missing,
        "duplicates": duplicates,
        "issues": issues,
        "calendar_gaps": calendar_gaps,
        "manifest_checked": manifest is not None,
        "manifest_failures": manifest.get("failures", []) if manifest else [],
        "manifest_issues": manifest_issues,
        "verified_session_exceptions": [
            list(key) for key in sorted(verified_exceptions)
        ],
        "requested_coverage": {
            "start": str(start),
            "end": str(end),
            "months": requested_months,
            "calendar_days": len(requested_days),
            "calendar_start": str(requested_days[0]) if requested_days else None,
            "calendar_end": str(requested_days[-1]) if requested_days else None,
            "missing_months": missing_months,
        },
        "assumed_availability_rows": assumptions,
        "point_in_time_verified": bool(bars) and assumptions == 0,
        "source_counts": dict(Counter(bar.source for bar in bars)),
        "coverage": {
            instrument: {
                "start": min(
                    (
                        str(bar.trading_date)
                        for bar in bars
                        if bar.instrument == instrument and start <= bar.trading_date <= end
                    ),
                    default=None,
                ),
                "end": max(
                    (
                        str(bar.trading_date)
                        for bar in bars
                        if bar.instrument == instrument and start <= bar.trading_date <= end
                    ),
                    default=None,
                ),
                "rows": sum(
                    bar.instrument == instrument and start <= bar.trading_date <= end
                    for bar in bars
                ),
            }
            for instrument in ("TAIEX", "TX")
        },
    }
