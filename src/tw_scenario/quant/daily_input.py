from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from .backtest import _read_manifest, load_dataset
from .collection import (
    _cache_evidence,
    _official_bars_for_range,
    _selected_reconciliation_bars,
    _write_processed_dataset,
)
from .contracts import OHLC, TZ, Bar, digest
from .daily import DataQualityEvidence, OHLCForecastSnapshotV1
from .forecast import Dataset
from .providers import FinMindProvider, OfficialProvider
from .quality import quality_report
from .reconciliation import reconcile_bars


def _capture_requests(
    provider: FinMindProvider,
    start: date,
    trading_date: date,
    calendar,
) -> tuple[list[Bar], list[dict]]:
    requests = []
    taiex_path = provider.download("TaiwanStockPrice", "TAIEX", start, trading_date)
    requests.append(_cache_evidence(provider, taiex_path, "TaiwanStockPrice", "TAIEX"))
    tx_path = provider.download("TaiwanFuturesDaily", "TX", start, trading_date)
    requests.append(_cache_evidence(provider, tx_path, "TaiwanFuturesDaily", "TX"))
    bars = provider.parse_taiex(taiex_path) + provider.parse_tx(tx_path, calendar)
    return bars, requests


def _selected_capture_bars(bars: list[Bar], trading_date: date) -> list[Bar]:
    """Exclude prediction-period daytime rows even if a late provider response contains them."""
    selected = []
    for bar in bars:
        if bar.trading_date < trading_date or (
            bar.trading_date == trading_date
            and bar.instrument == "TX"
            and bar.session == "night"
        ):
            selected.append(bar)
    return selected


def _merge_bars(base: list[Bar], incremental: list[Bar]) -> list[Bar]:
    merged = {
        (bar.instrument, bar.contract, bar.session, bar.trading_date): bar for bar in base
    }
    if len(merged) != len(base):
        raise ValueError("base daily dataset contains duplicate bars")
    for bar in incremental:
        key = (bar.instrument, bar.contract, bar.session, bar.trading_date)
        if key in merged and merged[key] != bar:
            raise ValueError(f"incremental bar conflicts with immutable base: {key}")
        merged[key] = bar
    return sorted(
        merged.values(),
        key=lambda bar: (
            bar.trading_date,
            bar.instrument,
            bar.contract,
            0 if bar.session == "night" else 1,
        ),
    )


def audit_preopen_dataset(
    data: Dataset,
    manifest: dict,
    trading_date: date,
    cutoff_at: datetime,
) -> dict:
    issues = []
    if cutoff_at.utcoffset() is None or cutoff_at.astimezone(TZ).date() != trading_date:
        issues.append("cutoff timestamp/date mismatch")
    if trading_date not in data.calendar.days:
        issues.append("prediction date absent from verified exchange calendar")
        previous_day = None
    else:
        try:
            previous_day = data.calendar.previous(trading_date)
        except ValueError as exc:
            issues.append(str(exc))
            previous_day = None

    historical_quality = None
    if previous_day is not None:
        historical_quality = quality_report(
            data.bars,
            data.calendar,
            date.fromisoformat(str(manifest["start"])),
            previous_day,
        )
        if historical_quality["status"] != "PASS_RETROSPECTIVE":
            issues.append(
                "historical selected-series quality failed: "
                f"missing={len(historical_quality['missing'])}, "
                f"duplicates={len(historical_quality['duplicates'])}, "
                f"issues={len(historical_quality['issues'])}, "
                f"calendar_gaps={len(historical_quality['calendar_gaps'])}"
            )

    required = []
    if previous_day is not None and trading_date in data.calendar.days:
        try:
            taiex = data.reference("TAIEX", trading_date)
            tx = data.reference("TX", trading_date)
            contract = data.contract("TX", trading_date)
            night = data.get("TX", contract, "night", trading_date, cutoff_at)
            required = [taiex, tx, night]
        except ValueError as exc:
            issues.append(f"required pre-open observation failed: {exc}")

    current_daytime = [
        bar for bar in data.bars
        if bar.trading_date == trading_date and bar.session == "day"
    ]
    if current_daytime:
        issues.append("prediction-period daytime rows are present")

    incremental_start = date.fromisoformat(str(manifest["incremental_start"]))
    incremental_bars = [bar for bar in data.bars if bar.trading_date >= incremental_start]
    late_fetches = sorted({
        bar.fetched_at.isoformat() for bar in incremental_bars if bar.fetched_at > cutoff_at
    })
    if late_fetches:
        issues.append("incremental provider response was fetched after the 08:30 cutoff")
    if not incremental_bars:
        issues.append("incremental capture contains no bars")
    if required and any(bar.available_at > cutoff_at or bar.session_end_at > cutoff_at for bar in required):
        issues.append("required source was not available and complete by cutoff")

    source_requests = manifest.get("source_requests")
    if not isinstance(source_requests, list) or len(source_requests) != 2:
        issues.append("pre-open capture must contain exactly two FinMind request records")
    else:
        for request in source_requests:
            try:
                fetched_at = datetime.fromisoformat(request["fetched_at"])
            except (KeyError, TypeError, ValueError):
                issues.append("source request has invalid fetched_at")
                continue
            if fetched_at.utcoffset() is None or fetched_at > cutoff_at:
                issues.append("source request was not captured by cutoff")

    official_night = manifest.get("official_current_night")
    if not isinstance(official_night, dict):
        issues.append("official current-night verification evidence is missing")
    else:
        try:
            official_fetched_at = datetime.fromisoformat(str(official_night["fetched_at"]))
        except (KeyError, TypeError, ValueError):
            issues.append("official current-night fetched_at is invalid")
        else:
            if official_fetched_at.utcoffset() is None or official_fetched_at > cutoff_at:
                issues.append("official current-night verification was fetched after cutoff")
        if official_night.get("ohlc_mismatch_fields") != []:
            issues.append("official current-night OHLC reconciliation is not clean")

    incremental_reconciliation = manifest.get("completed_incremental_reconciliation")
    if not isinstance(incremental_reconciliation, dict) or incremental_reconciliation.get(
        "status"
    ) not in {"PASS_RETROSPECTIVE_VALUE_AGREEMENT", "NO_COMPLETED_INCREMENT"}:
        issues.append("completed incremental FinMind/official reconciliation is not clean")
    elif (
        incremental_reconciliation.get("missing") != 0
        or incremental_reconciliation.get("mismatches") != 0
    ):
        issues.append("completed incremental reconciliation contains differences")

    report = {
        "schema_version": "daily-preopen-input-audit-v1",
        "status": "PASS" if not issues else "FAIL",
        "trading_date": str(trading_date),
        "cutoff_at": cutoff_at.isoformat(),
        "base_dataset_id": manifest.get("parent_dataset_id"),
        "daily_dataset_id": manifest.get("dataset_id"),
        "incremental_start": str(incremental_start),
        "required_sources": [
            {
                "instrument": bar.instrument,
                "contract": bar.contract or None,
                "session": bar.session,
                "trading_date": str(bar.trading_date),
                "available_at": bar.available_at.isoformat(),
                "fetched_at": bar.fetched_at.isoformat(),
                "source_hash": bar.source_hash,
            }
            for bar in required
        ],
        "current_daytime_rows": len(current_daytime),
        "incremental_rows": len(incremental_bars),
        "issues": issues,
        "historical_quality": historical_quality,
    }
    report["report_hash"] = digest(report)
    return report


def data_quality_evidence(report: dict) -> DataQualityEvidence:
    return DataQualityEvidence(
        status="PASS" if report.get("status") == "PASS" else "FAIL",
        report_hash=str(report["report_hash"]),
        issues=tuple(str(issue) for issue in report.get("issues", [])),
    )


def capture_preopen_dataset(
    base_dataset_dir: Path,
    trading_date: date,
    cutoff_at: datetime,
    root: Path,
) -> tuple[Path, dict]:
    """Capture a write-once pre-open increment and bind it to a certified base dataset."""
    if cutoff_at.utcoffset() is None:
        raise ValueError("cutoff_at must be timezone-aware")
    base, base_dataset_id = load_dataset(base_dataset_dir)
    base_manifest = _read_manifest(base_dataset_dir)
    if base_manifest.get("source") != "FinMind v4 reconciled with TWSE/TAIFEX official":
        raise ValueError("daily capture requires the certified FinMind/official base dataset")
    if trading_date not in base.calendar.days:
        raise ValueError("trading date is absent from the certified exchange calendar support")
    base_end = date.fromisoformat(str(base_manifest["end"]))
    incremental_start = min(base_end + timedelta(days=1), trading_date)

    cache = root / "daily" / str(trading_date) / "preopen" / "raw" / "finmind"
    provider = FinMindProvider(cache)
    captured, source_requests = _capture_requests(
        provider, incremental_start, trading_date, base.calendar
    )
    selected = _selected_capture_bars(captured, trading_date)
    contract = base.calendar.contract_for(trading_date)
    finmind_night = [
        bar for bar in selected
        if bar.instrument == "TX"
        and bar.contract == contract
        and bar.session == "night"
        and bar.trading_date == trading_date
    ]
    official_provider = OfficialProvider(
        root / "daily" / str(trading_date) / "preopen" / "raw" / "official"
    )
    previous_day = base.calendar.previous(trading_date)
    if incremental_start <= previous_day:
        official_incremental = _official_bars_for_range(
            official_provider, base.calendar, incremental_start, previous_day
        )
        primary_incremental = _selected_reconciliation_bars(
            captured, base.calendar, incremental_start, previous_day
        )
        authority_incremental = _selected_reconciliation_bars(
            official_incremental, base.calendar, incremental_start, previous_day
        )
        incremental_reconciliation = reconcile_bars(
            primary_incremental, authority_incremental
        )
        if incremental_reconciliation["status"] != "PASS_RETROSPECTIVE_VALUE_AGREEMENT":
            raise ValueError(
                "FinMind/official completed incremental reconciliation failed: "
                f"missing={len(incremental_reconciliation['missing'])}, "
                f"mismatches={len(incremental_reconciliation['mismatch_rows'])}"
            )
    else:
        incremental_reconciliation = {
            "status": "NO_COMPLETED_INCREMENT",
            "missing": [],
            "mismatch_rows": [],
        }
    official_night = [
        bar for bar in official_provider.tx_day(trading_date, "night")
        if bar.contract == contract
    ]
    if len(official_night) != 1:
        raise ValueError(
            f"official pre-open verification expected one {contract} night row, "
            f"found {len(official_night)}"
        )
    if len(finmind_night) > 1:
        raise ValueError("FinMind pre-open capture contains duplicate selected night rows")
    mismatch = []
    if finmind_night:
        mismatch = [
            field for field in OHLC
            if getattr(finmind_night[0], field) != getattr(official_night[0], field)
        ]
        if mismatch:
            raise ValueError(
                "FinMind/TAIFEX current-night OHLC mismatch: " + ",".join(mismatch)
            )
        selected_night = finmind_night[0]
        primary = "FinMind"
    else:
        selected_night = official_night[0]
        primary = "TAIFEX official fallback because FinMind current night was absent"
    selected = [
        bar for bar in selected
        if not (
            bar.instrument == "TX"
            and bar.contract == contract
            and bar.session == "night"
            and bar.trading_date == trading_date
        )
    ]
    selected.append(selected_night)
    merged = _merge_bars(base.bars, selected)
    manifest = {
        "start": base_manifest["start"],
        "end": trading_date,
        "source": "certified base plus FinMind write-once pre-open increment",
        "parent_dataset_id": base_dataset_id,
        "parent_manifest_hash": digest(base_manifest),
        "trading_date": trading_date,
        "cutoff_at": cutoff_at,
        "incremental_start": incremental_start,
        "source_requests": source_requests,
        "official_current_night": {
            "contract": contract,
            "source_hash": official_night[0].source_hash,
            "fetched_at": official_night[0].fetched_at,
            "primary_row": primary,
            "ohlc_mismatch_fields": mismatch,
        },
        "completed_incremental_reconciliation": {
            "status": incremental_reconciliation["status"],
            "report_hash": digest(incremental_reconciliation),
            "missing": len(incremental_reconciliation["missing"]),
            "mismatches": len(incremental_reconciliation["mismatch_rows"]),
        },
        "failures": [],
        "source_failure_count": 0,
        "prediction_period_daytime_excluded": True,
    }
    output = _write_processed_dataset(
        root / "daily" / str(trading_date) / "preopen", merged, base.calendar, manifest
    )
    complete_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    data, _ = load_dataset(output)
    report = audit_preopen_dataset(data, complete_manifest, trading_date, cutoff_at)
    return output, report


def capture_postclose_bars(
    daily_dataset_dir: Path,
    snapshot: OHLCForecastSnapshotV1,
    root: Path,
) -> tuple[dict[str, Bar], dict]:
    """Capture current-day cash/futures daytime actuals without mutating pre-open data."""
    data, daily_dataset_id = load_dataset(daily_dataset_dir)
    if snapshot.trading_date not in data.calendar.days:
        raise ValueError("snapshot trading date is absent from daily dataset calendar")
    cache = (
        root
        / "daily"
        / str(snapshot.trading_date)
        / "postclose"
        / "raw"
        / "finmind"
    )
    provider = FinMindProvider(cache)
    captured, requests = _capture_requests(
        provider,
        snapshot.trading_date,
        snapshot.trading_date,
        data.calendar,
    )
    forecast_contract = next(
        forecast.contract for forecast in snapshot.forecasts if forecast.instrument == "TX"
    )
    matches = {
        "TAIEX": [
            bar for bar in captured
            if bar.instrument == "TAIEX"
            and bar.session == "day"
            and bar.trading_date == snapshot.trading_date
        ],
        "TX": [
            bar for bar in captured
            if bar.instrument == "TX"
            and bar.contract == forecast_contract
            and bar.session == "day"
            and bar.trading_date == snapshot.trading_date
        ],
    }
    issues = []
    official_provider = OfficialProvider(
        root
        / "daily"
        / str(snapshot.trading_date)
        / "postclose"
        / "raw"
        / "official"
    )
    official_matches = {
        "TAIEX": [
            bar for bar in official_provider.twse_month(
                snapshot.trading_date.year, snapshot.trading_date.month
            )
            if bar.trading_date == snapshot.trading_date
        ],
        "TX": [
            bar for bar in official_provider.tx_day(snapshot.trading_date, "day")
            if bar.contract == forecast_contract
        ],
    }
    for instrument, rows in matches.items():
        if len(rows) != 1:
            issues.append(f"expected one {instrument} daytime realization, found {len(rows)}")
            continue
        bar = rows[0]
        if bar.fetched_at < bar.available_at or bar.fetched_at < bar.session_end_at:
            issues.append(f"{instrument} post-close response was fetched before availability")
        official_rows = official_matches[instrument]
        if len(official_rows) != 1:
            issues.append(
                f"expected one official {instrument} realization, found {len(official_rows)}"
            )
        else:
            official_bar = official_rows[0]
            if (
                official_bar.fetched_at < official_bar.available_at
                or official_bar.fetched_at < official_bar.session_end_at
            ):
                issues.append(f"official {instrument} response was fetched before availability")
            if any(getattr(bar, field) != getattr(official_bar, field) for field in OHLC):
                issues.append(f"FinMind/official {instrument} daytime OHLC mismatch")
    report = {
        "schema_version": "daily-postclose-input-audit-v1",
        "status": "PASS" if not issues else "FAIL",
        "trading_date": str(snapshot.trading_date),
        "forecast_id": snapshot.forecast_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "daily_dataset_id": daily_dataset_id,
        "source_requests": requests,
        "rows": {
            instrument: [
                {
                    "contract": bar.contract or None,
                    "session_end_at": bar.session_end_at.isoformat(),
                    "available_at": bar.available_at.isoformat(),
                    "fetched_at": bar.fetched_at.isoformat(),
                    "source_hash": bar.source_hash,
                    "settlement_separate": bar.settlement is not None,
                }
                for bar in rows
            ]
            for instrument, rows in matches.items()
        },
        "official_rows": {
            instrument: [
                {
                    "contract": bar.contract or None,
                    "fetched_at": bar.fetched_at.isoformat(),
                    "source_hash": bar.source_hash,
                }
                for bar in rows
            ]
            for instrument, rows in official_matches.items()
        },
        "issues": issues,
    }
    report["report_hash"] = digest(report)
    return {
        instrument: rows[0] for instrument, rows in matches.items() if len(rows) == 1
    }, report
