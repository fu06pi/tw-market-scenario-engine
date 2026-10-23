from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .backtest import _git_state, _read_manifest, load_dataset
from .contracts import TZ, digest, write_once
from .daily import (
    V0_HEADERS,
    OHLCRealizationV1,
    OHLCValues,
    V0SheetDryRunBatch,
    append_evaluation,
    append_realization,
    build_v0_narrative,
    build_v0_postclose_sheet_batch,
    build_v0_sheet_dry_run,
    freeze_snapshot,
    load_snapshot,
    save_postclose_sheet_dry_run,
    save_sheet_dry_run,
    seal_snapshot,
)
from .daily_input import (
    audit_preopen_dataset,
    capture_postclose_bars,
    capture_preopen_dataset,
    data_quality_evidence,
)
from .daily_runner import build_locked_baseline_body
from .sheets import StagingSheetWriter, save_receipt


class DailyRunFailed(RuntimeError):
    def __init__(self, message: str, evidence_path: Path):
        super().__init__(message)
        self.evidence_path = evidence_path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"invalid required JSON: {path}") from None


def _write_once_or_same(path: Path, value: Any) -> tuple[Path, bool]:
    if path.exists():
        if _read_json(path) != value:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return path, True
    write_once(path, value)
    return path, False


def _attempt_path(
    root: Path, trading_date: date, started_at: datetime, phase: str = "preopen"
) -> Path:
    attempt_id = digest({
        "trading_date": trading_date,
        "started_at": started_at.isoformat(),
        "phase": phase,
    })[:16]
    return root / "attempts" / str(trading_date) / f"{phase}-{attempt_id}.json"


def _failure(
    root: Path,
    trading_date: date,
    started_at: datetime,
    stage: str,
    exc: Exception,
    phase: str = "preopen",
) -> DailyRunFailed:
    path = _attempt_path(root, trading_date, started_at, phase)
    evidence = {
        "schema_version": "daily-run-attempt-v1",
        "status": "FAIL",
        "phase": phase,
        "trading_date": str(trading_date),
        "started_at": started_at.isoformat(),
        "failed_at": datetime.now(TZ).isoformat(),
        "failed_stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "production_switch": False,
        "production_sheet_written": False,
    }
    _write_once_or_same(path, evidence)
    return DailyRunFailed(f"{stage} failed; evidence={path}: {exc}", path)


def run_preopen_staging(
    *,
    daily_dataset_dir: Path,
    baseline_path: Path,
    final_summary_path: Path,
    config: dict,
    config_path: Path,
    trading_date: date,
    started_at: datetime,
    output_root: Path,
    start_row: int,
    observed_headers: tuple[str, ...] = V0_HEADERS,
    news: list[dict] | None = None,
) -> Path:
    """Run the cutoff-safe local pre-open chain; no Google Sheet write occurs here."""
    stage = "validate_run_window"
    try:
        cutoff = datetime.combine(
            trading_date,
            datetime.strptime(config["cutoff"], "%H:%M").time(),
            TZ,
        )
        if started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")
        if started_at < cutoff:
            raise ValueError("forecast phase cannot start before the 08:30 cutoff")

        stage = "load_and_audit_daily_dataset"
        data, daily_dataset_id = load_dataset(daily_dataset_dir)
        manifest = _read_manifest(daily_dataset_dir)
        report = audit_preopen_dataset(data, manifest, trading_date, cutoff)
        audit_path = output_root / "input-audits" / str(trading_date) / f"{daily_dataset_id}.json"
        _write_once_or_same(audit_path, report)
        quality = data_quality_evidence(report)
        if quality.status != "PASS":
            raise ValueError("pre-open input audit is FAIL")

        stage = "validate_locked_model_policy"
        baseline = _read_json(baseline_path)
        final_summary = _read_json(final_summary_path)
        if final_summary.get("status") != "FAIL":
            raise ValueError("baseline fallback requires the recorded final acceptance failure")
        if baseline.get("dataset_id") != manifest.get("parent_dataset_id"):
            raise ValueError("baseline lock does not match the certified parent dataset")

        stage = "forecast_and_seal"
        body = build_locked_baseline_body(
            data=data,
            manifest=manifest,
            baseline_lock=baseline,
            config=config,
            trading_date=trading_date,
            generated_at=started_at,
            persisted_at=None,
            code_version=digest(_git_state(config_path.parent.parent.parent)),
            final_summary_hash=digest(final_summary),
            data_quality=quality,
        )
        snapshot = seal_snapshot(body)
        snapshot_path = output_root / "snapshots" / f"{snapshot.forecast_id}.json"
        if snapshot_path.exists():
            existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if existing != snapshot.model_dump(mode="json"):
                raise FileExistsError(f"different snapshot already exists: {snapshot_path}")
            snapshot_reused = True
        else:
            freeze_snapshot(snapshot, snapshot_path)
            snapshot_reused = False
        if snapshot.timeliness != "ON_TIME":
            raise ValueError("snapshot missed the 08:40 persistence deadline")

        stage = "narrative"
        narrative = build_v0_narrative(snapshot, news or [])
        narrative_path = output_root / "narratives" / f"{snapshot.forecast_id}.json"
        _, narrative_reused = _write_once_or_same(
            narrative_path, narrative.model_dump(mode="json")
        )

        stage = "sheet_write_plan"
        batch = build_v0_sheet_dry_run(
            snapshot, narrative, observed_headers, start_row
        )
        batch_path = output_root / "sheet-dry-runs" / f"{snapshot.forecast_id}.json"
        if batch_path.exists():
            if _read_json(batch_path) != batch.model_dump(mode="json"):
                raise FileExistsError(f"different Sheet plan already exists: {batch_path}")
            batch_reused = True
        else:
            save_sheet_dry_run(batch, batch_path)
            batch_reused = False

        stage = "complete"
        attempt_path = _attempt_path(output_root, trading_date, started_at)
        evidence = {
            "schema_version": "daily-run-attempt-v1",
            "status": "READY_FOR_STAGING_SHEET_WRITE",
            "phase": "preopen",
            "trading_date": str(trading_date),
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(TZ).isoformat(),
            "forecast_id": snapshot.forecast_id,
            "snapshot_hash": snapshot.snapshot_hash,
            "timeliness": snapshot.timeliness,
            "daily_dataset_id": daily_dataset_id,
            "input_audit": str(audit_path),
            "snapshot": str(snapshot_path),
            "narrative": str(narrative_path),
            "sheet_write_plan": str(batch_path),
            "sheet_write_range": batch.write_range,
            "reused": {
                "snapshot": snapshot_reused,
                "narrative": narrative_reused,
                "sheet_write_plan": batch_reused,
            },
            "staging_sheet_written": False,
            "production_switch": False,
            "production_sheet_written": False,
        }
        _write_once_or_same(attempt_path, evidence)
        return attempt_path
    except Exception as exc:
        if isinstance(exc, DailyRunFailed):
            raise
        raise _failure(output_root, trading_date, started_at, stage, exc) from exc


def run_live_preopen(
    *,
    trading_date: date,
    base_dataset_dir: Path,
    baseline_path: Path,
    final_summary_path: Path,
    config: dict,
    config_path: Path,
    data_root: Path,
    output_root: Path,
    writer: StagingSheetWriter,
    news: list[dict] | None = None,
    clock=None,
    sleeper=None,
) -> Path:
    """One staging-only entrypoint: capture before cutoff, then freeze and write after it."""
    clock = clock or (lambda: datetime.now(TZ))
    sleeper = sleeper or time.sleep
    initiated_at = clock()
    stage = "validate_live_window"
    try:
        if initiated_at.utcoffset() is None:
            raise ValueError("live clock must be timezone-aware")
        if initiated_at.astimezone(TZ).date() != trading_date:
            raise ValueError("live pre-open runs only for the current Taipei trading date")
        cutoff = datetime.combine(
            trading_date,
            datetime.strptime(config["cutoff"], "%H:%M").time(),
            TZ,
        )
        capture_not_before = datetime.combine(
            trading_date,
            datetime.strptime(config.get("capture_not_before", "08:20"), "%H:%M").time(),
            TZ,
        )
        if not capture_not_before <= initiated_at < cutoff:
            raise ValueError("live pre-open capture must start from 08:20 through 08:29 Taipei")

        stage = "capture_preopen_inputs"
        daily_dataset_dir, audit = capture_preopen_dataset(
            base_dataset_dir=base_dataset_dir,
            trading_date=trading_date,
            cutoff_at=cutoff,
            root=data_root,
        )
        audit_path = (
            output_root / "input-audits" / str(trading_date) / f"{daily_dataset_dir.name}.json"
        )
        _write_once_or_same(audit_path, audit)
        if audit["status"] != "PASS":
            raise ValueError("pre-open capture failed the cutoff/data-quality audit")

        stage = "staging_sheet_preflight"
        writer.verify_headers("'每日預測與覆盤'!A1:P1")
        start_row = writer.next_start_row("每日預測與覆盤")

        stage = "wait_for_cutoff"
        remaining = (cutoff - clock()).total_seconds()
        if remaining > 0:
            sleeper(remaining)
        generated_at = clock()
        if generated_at < cutoff:
            raise RuntimeError("clock did not reach the 08:30 forecast cutoff")

        stage = "forecast_freeze_narrative_plan"
        attempt_path = run_preopen_staging(
            daily_dataset_dir=daily_dataset_dir,
            baseline_path=baseline_path,
            final_summary_path=final_summary_path,
            config=config,
            config_path=config_path,
            trading_date=trading_date,
            started_at=generated_at,
            output_root=output_root,
            start_row=start_row,
            observed_headers=V0_HEADERS,
            news=news,
        )
        attempt = _read_json(attempt_path)
        batch = V0SheetDryRunBatch.model_validate(_read_json(Path(attempt["sheet_write_plan"])))

        stage = "staging_sheet_write_readback"
        sheet_result = writer.write_morning(batch)
        sheet_receipt_path = (
            output_root / "sheet-receipts" / f"{batch.forecast_id}-morning.json"
        )
        save_receipt(sheet_result, sheet_receipt_path)

        stage = "complete"
        e2e = {
            "schema_version": "daily-live-preopen-e2e-v1",
            "status": "STAGING_PREOPEN_E2E_VERIFIED",
            "trading_date": str(trading_date),
            "initiated_at": initiated_at.isoformat(),
            "capture_completed_by_cutoff": True,
            "generated_at": generated_at.isoformat(),
            "completed_at": clock().isoformat(),
            "daily_dataset_dir": str(daily_dataset_dir),
            "input_audit": str(audit_path),
            "forecast_attempt": str(attempt_path),
            "forecast_id": batch.forecast_id,
            "snapshot_hash": batch.snapshot_hash,
            "sheet_receipt": str(sheet_receipt_path),
            "staging_sheet_written": True,
            "production_switch": False,
            "production_sheet_written": False,
        }
        path = output_root / "e2e-receipts" / f"{batch.forecast_id}-preopen.json"
        _write_once_or_same(path, e2e)
        return path
    except Exception as exc:
        if isinstance(exc, DailyRunFailed):
            raise
        raise _failure(output_root, trading_date, initiated_at, stage, exc) from exc


def _append_or_same(record, root: Path, kind: str):
    contract = record.contract or "cash"
    path = root / kind / record.forecast_id / f"{record.instrument}-{contract}.json"
    if path.exists():
        if _read_json(path) != record.model_dump(mode="json"):
            raise FileExistsError(f"different append-only record already exists: {path}")
        return path, True
    if kind == "realizations-v1":
        append_realization(record, root)
    else:
        append_evaluation(record, root)
    return path, False


def run_live_postclose(
    *,
    snapshot_path: Path,
    daily_dataset_dir: Path,
    morning_batch_path: Path,
    data_root: Path,
    output_root: Path,
    writer: StagingSheetWriter,
    config: dict,
    clock=None,
) -> Path:
    """Capture actual OHLC, append evaluation, and update only staging G:N,P."""
    clock = clock or (lambda: datetime.now(TZ))
    initiated_at = clock()
    snapshot = load_snapshot(snapshot_path)
    trading_date = snapshot.trading_date
    stage = "validate_postclose_window"
    try:
        if initiated_at.utcoffset() is None:
            raise ValueError("post-close clock must be timezone-aware")
        if initiated_at.astimezone(TZ).date() != trading_date:
            raise ValueError("live post-close runs only on the snapshot trading date")
        earliest = datetime.combine(
            trading_date,
            datetime.strptime(config["postclose_not_before"], "%H:%M").time(),
            TZ,
        )
        if initiated_at < earliest:
            raise ValueError("post-close capture cannot begin before the configured Taipei time")

        stage = "capture_postclose_inputs"
        bars, audit = capture_postclose_bars(daily_dataset_dir, snapshot, data_root)
        audit_path = (
            output_root / "postclose-input-audits" / str(trading_date) / f"{snapshot.forecast_id}.json"
        )
        _write_once_or_same(audit_path, audit)
        if audit["status"] != "PASS":
            raise ValueError("post-close input audit is FAIL")

        stage = "append_realizations_and_evaluations"
        recorded_at = clock()
        realizations = []
        evaluations = []
        reused = {"realizations": [], "evaluations": []}
        for forecast in snapshot.forecasts:
            bar = bars[forecast.instrument]
            realization = OHLCRealizationV1(
                realization_id=(
                    f"realization-{forecast.instrument.lower()}-"
                    f"{digest({'forecast_id': snapshot.forecast_id, 'source': bar.source_hash})[:16]}"
                ),
                forecast_id=snapshot.forecast_id,
                snapshot_hash=snapshot.snapshot_hash,
                trading_date=trading_date,
                instrument=forecast.instrument,
                contract=forecast.contract,
                actual=OHLCValues(
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                ),
                source_id=f"finmind:{forecast.instrument}:day:{trading_date}",
                source_hash=bar.source_hash,
                observed_at=bar.fetched_at,
                recorded_at=recorded_at,
            )
            from .daily import evaluate_realization

            evaluation = evaluate_realization(snapshot, realization, recorded_at)
            _, realization_reused = _append_or_same(
                realization, output_root, "realizations-v1"
            )
            _, evaluation_reused = _append_or_same(
                evaluation, output_root, "evaluations-v1"
            )
            realizations.append(realization)
            evaluations.append(evaluation)
            reused["realizations"].append(realization_reused)
            reused["evaluations"].append(evaluation_reused)

        stage = "build_postclose_sheet_plan"
        morning_batch = V0SheetDryRunBatch.model_validate(_read_json(morning_batch_path))
        postclose = build_v0_postclose_sheet_batch(
            snapshot,
            morning_batch,
            tuple(realizations),
            tuple(evaluations),
            recorded_at,
        )
        plan_path = (
            output_root / "sheet-postclose-dry-runs" / f"{snapshot.forecast_id}.json"
        )
        if plan_path.exists():
            if _read_json(plan_path) != postclose.model_dump(mode="json"):
                raise FileExistsError(f"different post-close plan already exists: {plan_path}")
            plan_reused = True
        else:
            save_postclose_sheet_dry_run(postclose, plan_path)
            plan_reused = False

        stage = "staging_postclose_write_readback"
        sheet_result = writer.write_postclose(postclose)
        sheet_receipt_path = (
            output_root / "sheet-receipts" / f"{snapshot.forecast_id}-postclose.json"
        )
        save_receipt(sheet_result, sheet_receipt_path)

        stage = "complete"
        evidence = {
            "schema_version": "daily-live-postclose-e2e-v1",
            "status": "STAGING_POSTCLOSE_E2E_VERIFIED",
            "trading_date": str(trading_date),
            "initiated_at": initiated_at.isoformat(),
            "completed_at": clock().isoformat(),
            "forecast_id": snapshot.forecast_id,
            "snapshot_hash": snapshot.snapshot_hash,
            "postclose_input_audit": str(audit_path),
            "postclose_sheet_plan": str(plan_path),
            "sheet_receipt": str(sheet_receipt_path),
            "reused": {**reused, "sheet_plan": plan_reused},
            "staging_sheet_written": True,
            "production_switch": False,
            "production_sheet_written": False,
        }
        path = output_root / "e2e-receipts" / f"{snapshot.forecast_id}-postclose.json"
        _write_once_or_same(path, evidence)
        return path
    except Exception as exc:
        if isinstance(exc, DailyRunFailed):
            raise
        raise _failure(
            output_root,
            trading_date,
            initiated_at,
            stage,
            exc,
            phase="postclose",
        ) from exc
