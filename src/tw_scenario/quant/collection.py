from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from .contracts import TZ, Calendar, digest, write_once
from .providers import FINMIND_URL, FinMindProvider, OfficialProvider


def _bounded_year_ranges(start: date, end: date):
    """Yield non-overlapping requests that remain within FinMind's one-year bound."""
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(cursor.year, 12, 31))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _next_month(month: date) -> date:
    return (month.replace(day=28) + timedelta(days=4)).replace(day=1)


def _expiry_calendar(days: list[date]) -> dict[str, date]:
    """Apply the predeclared TAIFEX third-Wednesday/next-session rule."""
    if not days:
        return {}
    expiries = {}
    month = days[0].replace(day=1)
    last_month = days[-1].replace(day=1)
    while month <= last_month:
        following = _next_month(month)
        third_wednesday = month + timedelta(days=(2 - month.weekday()) % 7 + 14)
        candidates = [
            day for day in days
            if third_wednesday <= day < following
        ]
        if candidates:
            expiries[month.strftime("%Y%m")] = candidates[0]
        month = following
    return expiries


def _cache_evidence(provider: FinMindProvider, path: Path, dataset: str,
                    data_id: str | None) -> dict:
    cached = provider._read_cache(path, dataset, data_id)
    return {
        "cache_file": path.name,
        "request": cached["request"],
        "fetched_at": cached["fetched_at"],
        "sha256": cached["sha256"],
        "records": len(cached["records"]),
    }


def _write_processed_dataset(root: Path, bars, calendar: Calendar, manifest: dict) -> Path:
    payload = [bar.model_dump(mode="json") for bar in bars]
    bars_hash = digest(payload)
    calendar_payload = calendar.model_dump(mode="json")
    calendar_hash = digest(calendar_payload)
    manifest_identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"dataset_id", "bars_hash", "calendar_hash", "rows"}
    }
    dataset_id = digest({
        "bars_hash": bars_hash,
        "calendar_hash": calendar_hash,
        "manifest_hash": digest(manifest_identity),
    })
    complete_manifest = {
        **manifest,
        "dataset_id": dataset_id,
        "bars_hash": bars_hash,
        "calendar_hash": calendar_hash,
        "rows": len(bars),
    }
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    output = processed / dataset_id
    if output.exists():
        try:
            existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError("existing processed dataset is incomplete or invalid") from None
        if digest(existing) != digest(complete_manifest):
            raise ValueError("existing processed dataset manifest does not match content")
        return output

    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=processed))
    try:
        bars_path = temporary / "bars.jsonl"
        with bars_path.open("x", encoding="utf-8") as target:
            for row in payload:
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        write_once(temporary / "calendar.json", calendar_payload)
        write_once(temporary / "manifest.json", complete_manifest)
        temporary.rename(output)
        directory_fd = os.open(processed, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def _official_bars_for_range(
    provider: OfficialProvider,
    calendar: Calendar,
    start: date,
    end: date,
):
    bars = []
    month = start.replace(day=1)
    while month <= end:
        bars.extend(provider.twse_month(month.year, month.month))
        month = _next_month(month)

    current_year = datetime.now(TZ).year
    for year in range(start.year, end.year + 1):
        year_start = max(start, date(year, 1, 1))
        year_end = min(end, date(year, 12, 31))
        if year < current_year:
            bars.extend(provider.tx_year(year, calendar))
            continue
        cursor = year_start.replace(day=1)
        while cursor <= year_end:
            following = _next_month(cursor)
            bars.extend(provider.tx_range(
                max(year_start, cursor),
                min(year_end, following - timedelta(days=1)),
                calendar,
            ))
            cursor = following
    return [bar for bar in bars if start <= bar.trading_date <= end]


def _selected_reconciliation_bars(bars, calendar: Calendar, start: date, end: date):
    selected = []
    for bar in bars:
        if not start <= bar.trading_date <= end:
            continue
        if bar.instrument == "TAIEX":
            selected.append(bar)
            continue
        if bar.contract == calendar.contract_for(bar.trading_date):
            selected.append(bar)
    return selected


def certify_finmind_with_official(dataset_dir: Path, root: Path) -> tuple[Path, dict]:
    """Create an immutable certified view after selected-series official reconciliation."""
    from .backtest import load_dataset
    from .reconciliation import reconcile_bars

    data, parent_dataset_id = load_dataset(dataset_dir)
    parent_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if parent_manifest.get("source") != "FinMind v4":
        raise ValueError("only an uncertified FinMind v4 dataset can be certified")
    start = date.fromisoformat(str(parent_manifest["start"]))
    end = date.fromisoformat(str(parent_manifest["end"]))
    authority = _official_bars_for_range(
        OfficialProvider(root / "raw" / "official"), data.calendar, start, end
    )
    primary_selected = _selected_reconciliation_bars(data.bars, data.calendar, start, end)
    authority_selected = _selected_reconciliation_bars(
        authority, data.calendar, start, end
    )
    report = reconcile_bars(primary_selected, authority_selected)
    if report["status"] != "PASS_RETROSPECTIVE_VALUE_AGREEMENT":
        raise ValueError(
            "FinMind/official selected-series reconciliation failed: "
            f"missing={len(report['missing'])}, mismatches={len(report['mismatch_rows'])}"
        )

    requested_days = [day for day in data.calendar.days if start <= day <= end]
    primary_tx_day = {
        bar.trading_date for bar in data.bars
        if bar.instrument == "TX" and bar.session == "day"
    }
    primary_tx_night = {
        bar.trading_date for bar in data.bars
        if bar.instrument == "TX" and bar.session == "night"
    }
    authority_tx_day = {
        bar.trading_date for bar in authority
        if bar.instrument == "TX" and bar.session == "day"
    }
    authority_tx_night = {
        bar.trading_date for bar in authority
        if bar.instrument == "TX" and bar.session == "night"
    }
    exceptions = [
        {
            "instrument": "TX",
            "session": "night",
            "trading_date": str(day),
            "basis": (
                "FinMind and TAIFEX official records both contain daytime TX rows "
                "and no night rows across any explicit TX contract"
            ),
        }
        for day in requested_days
        if day in primary_tx_day
        and day in authority_tx_day
        and day not in primary_tx_night
        and day not in authority_tx_night
    ]
    report.update({
        "parent_dataset_id": parent_dataset_id,
        "requested_start": str(start),
        "requested_end": str(end),
        "verified_session_exceptions": exceptions,
        "official_source_hashes": sorted({bar.source_hash for bar in authority}),
    })
    reconciliation_hash = digest(report)
    manifest = {
        key: value
        for key, value in parent_manifest.items()
        if key not in {"dataset_id", "bars_hash", "calendar_hash", "rows"}
    }
    manifest.update({
        "source": "FinMind v4 reconciled with TWSE/TAIFEX official",
        "parent_dataset_id": parent_dataset_id,
        "reconciliation_hash": reconciliation_hash,
        "verified_session_exceptions": exceptions,
        "authority_source_hashes": report["official_source_hashes"],
    })
    output = _write_processed_dataset(root, data.bars, data.calendar, manifest)
    report["certified_dataset_id"] = output.name
    report["reconciliation_hash"] = reconciliation_hash
    return output, report


def collect_finmind(
    start: date,
    end: date,
    root: Path,
    include_tx: bool = True,
) -> Path:
    """Build one immutable FinMind research dataset with a verified roll-calendar tail."""
    if start > end:
        raise ValueError("collection start must not be after end")
    if end >= datetime.now(TZ).date():
        raise ValueError("only complete prior dates may be collected")

    provider = FinMindProvider(root / "raw" / "finmind")
    # The prior exchange date is required to timestamp the first night session. Two
    # successor months keep the fixed roll rule verifiable near the requested tail.
    calendar_start = start - timedelta(days=45)
    calendar_end = end + timedelta(days=62)
    calendar_days = []
    source_requests = []
    fetched_at = []
    for chunk_start, chunk_end in _bounded_year_ranges(calendar_start, calendar_end):
        path = provider.download("TaiwanStockTradingDate", None, chunk_start, chunk_end)
        evidence = _cache_evidence(provider, path, "TaiwanStockTradingDate", None)
        source_requests.append(evidence)
        fetched_at.append(datetime.fromisoformat(evidence["fetched_at"]))
        calendar_days.extend(provider.parse_trading_dates(path))
    calendar_days = sorted(set(calendar_days))
    calendar = Calendar(
        days=calendar_days,
        expiries=_expiry_calendar(calendar_days),
        source=(
            "FinMind TaiwanStockTradingDate; TAIFEX monthly expiry derived by the "
            "predeclared third-Wednesday/next-exchange-session rule; retrospective "
            "API, not an archived point-in-time calendar"
        ),
        available_at=max(fetched_at),
    )

    bars = []
    for chunk_start, chunk_end in _bounded_year_ranges(start, end):
        path = provider.download("TaiwanStockPrice", "TAIEX", chunk_start, chunk_end)
        source_requests.append(_cache_evidence(
            provider, path, "TaiwanStockPrice", "TAIEX"
        ))
        bars.extend(provider.parse_taiex(path))
        if include_tx:
            path = provider.download("TaiwanFuturesDaily", "TX", chunk_start, chunk_end)
            source_requests.append(_cache_evidence(
                provider, path, "TaiwanFuturesDaily", "TX"
            ))
            bars.extend(provider.parse_tx(path, calendar))
    bars.sort(key=lambda bar: (
        bar.trading_date,
        bar.instrument,
        bar.contract,
        0 if bar.session == "night" else 1,
    ))

    requested_days = [day for day in calendar.days if start <= day <= end]
    manifest = {
        "start": start,
        "end": end,
        "requested_coverage": {"start": start, "end": end},
        "calendar_coverage": {
            "start": calendar.days[0] if calendar.days else None,
            "end": calendar.days[-1] if calendar.days else None,
            "requested_exchange_days": len(requested_days),
            "support_exchange_days": len(calendar.days),
        },
        "source": "FinMind v4",
        "source_url": FINMIND_URL,
        "source_requests": source_requests,
        "failures": [],
        "source_failure_count": 0,
        "include_tx": include_tx,
        "usage": (
            "retrospective non-commercial research; FinMind attribution required; "
            "raw data is not redistributed; availability timestamps are assumptions, "
            "not an archived point-in-time feed"
        ),
    }
    return _write_processed_dataset(root, bars, calendar, manifest)


def collect_official(start: date, end: date, root: Path, include_tx: bool = True):
    if end >= datetime.now(TZ).date():
        raise ValueError('only complete prior dates may be collected')
    provider = OfficialProvider(root/'raw'/'official')
    month = start.replace(day=1)
    bars, failures = [], []
    while month <= end:
        try:
            bars.extend(provider.twse_month(month.year, month.month))
        except Exception as exc:  # noqa: BLE001 - source failures belong in the manifest
            failures.append({'month': str(month), 'source': 'TWSE', 'error': str(exc)})
        month = (month.replace(day=28)+timedelta(days=4)).replace(day=1)
    bars = [b for b in bars if start <= b.trading_date <= end]
    days = sorted({b.trading_date for b in bars})
    # Official rule: third Wednesday, postponed if not a trading day.
    # Dates derived from downloaded TWSE history: retrospective calendar only.
    expiries = {}
    for d in days:
        first = d.replace(day=1)
        third_wed = first+timedelta(days=(2-first.weekday()) % 7+14)
        expiry = next((x for x in days if x >= third_wed), None)
        if expiry is not None:
            expiries[d.strftime('%Y%m')] = expiry
    calendar = Calendar(days=days, expiries=expiries,
                        source='TWSE observed sessions + TAIFEX third-Wednesday rule; retrospective calendar, not archived PIT',
                        available_at=datetime.now(TZ))
    if include_tx:
        for year in range(start.year, min(end.year, 2025) + 1):
            try:
                bars.extend(provider.tx_year(year, calendar))
                print(json.dumps({'taifex_year': year, 'rows': len(bars)}), flush=True)
            except Exception as exc:  # noqa: BLE001 - source failures belong in the manifest
                failures.append({'year': year, 'source': 'TAIFEX', 'error': str(exc)})
        if end.year >= 2026:
            cursor = max(start, date(2026, 1, 1)).replace(day=1)
            while cursor <= end:
                next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
                range_start = max(start, cursor)
                range_end = min(end, next_month - timedelta(days=1))
                try:
                    bars.extend(provider.tx_range(range_start, range_end, calendar))
                except Exception as exc:  # noqa: BLE001 - source failures belong in the manifest
                    failures.append({
                        'start': str(range_start), 'end': str(range_end),
                        'source': 'TAIFEX', 'error': str(exc),
                    })
                cursor = next_month
    payload = [b.model_dump(mode='json') for b in bars]
    bars_hash = digest(payload)
    dataset_id = bars_hash
    out = root/'processed'/dataset_id
    out.mkdir(parents=True, exist_ok=True)
    path = out/'bars.jsonl'
    if not path.exists():
        with path.open('x') as f:
            for b in payload:
                f.write(json.dumps(b, ensure_ascii=False)+'\n')
        write_once(out/'calendar.json', calendar.model_dump(mode='json'))
        write_once(out/'manifest.json', {
            'dataset_id': dataset_id,
            'bars_hash': bars_hash,
            'start': start,
            'end': end,
            'requested_coverage': {'start': start, 'end': end},
            'calendar_coverage': {
                'start': days[0] if days else None,
                'end': days[-1] if days else None,
                'exchange_days': len(days),
            },
            'rows': len(bars),
            'failures': failures,
            'source_failure_count': len(failures),
            'usage': 'retrospective research; availability assumptions, no PIT archive',
        })
    return out
