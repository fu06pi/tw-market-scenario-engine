from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from tw_scenario.quant import daily_orchestration
from tw_scenario.quant.contracts import TZ, Bar, write_once
from tw_scenario.quant.daily import (
    DataQualityEvidence,
    InstrumentOHLCForecast,
    MarginalInterval,
    ModelIdentity,
    OHLCForecastSnapshotBodyV1,
    OHLCMarginalIntervals80,
    OHLCValues,
    SourceEvidence,
    V0MorningSheetRow,
    V0SheetDryRunBatch,
    deterministic_forecast_id,
    freeze_snapshot,
    save_sheet_dry_run,
    seal_snapshot,
)
from tw_scenario.quant.daily_orchestration import (
    DailyRunFailed,
    run_head_live_preopen,
    run_live_postclose,
    run_live_preopen,
)

DAY = date(2026, 10, 16)
FORECAST_ID = f"ohlc-v1-{DAY}-" + "1" * 64


def _at(clock: str) -> datetime:
    return datetime.combine(DAY, datetime.strptime(clock, "%H:%M").time(), TZ)


def _batch() -> V0SheetDryRunBatch:
    rows = []
    for instrument in ("TAIEX", "TX"):
        rows.append(V0MorningSheetRow(values=(
            "2026/10/16",
            instrument,
            "hypothesis",
            "direction",
            "confidence",
            "evidence",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "immutable note",
            "",
        )))
    return V0SheetDryRunBatch(
        forecast_id=FORECAST_ID,
        snapshot_hash="2" * 64,
        write_range="'每日預測與覆盤'!A42:P43",
        rows=tuple(rows),
    )


class FakeWriter:
    def verify_headers(self, _range):
        return None

    def next_start_row(self, _name):
        return 42

    def write_morning(self, batch):
        return {
            "operation": "MORNING_STAGING_WRITE",
            "forecast_id": batch.forecast_id,
            "snapshot_hash": batch.snapshot_hash,
            "sheet_id_hash": "3" * 64,
            "write_range": batch.write_range,
            "reused": False,
            "row_fingerprints": ["4" * 64, "5" * 64],
            "production_sheet_written": False,
        }

    def write_postclose(self, batch):
        return {
            "operation": "POSTCLOSE_STAGING_WRITE",
            "forecast_id": batch.forecast_id,
            "snapshot_hash": batch.snapshot_hash,
            "sheet_id_hash": "3" * 64,
            "write_range": "'每日預測與覆盤'!A42:P43",
            "reused": False,
            "preopen_fingerprints": [row.preopen_fingerprint for row in batch.rows],
            "production_sheet_written": False,
        }


def _forecast(instrument: str) -> InstrumentOHLCForecast:
    contract = None if instrument == "TAIEX" else "202610"
    point = OHLCValues(open=20_000, high=20_200, low=19_800, close=20_100)
    intervals = OHLCMarginalIntervals80(
        open=MarginalInterval(lower=19_500, upper=20_500),
        high=MarginalInterval(lower=19_900, upper=20_800),
        low=MarginalInterval(lower=19_200, upper=20_000),
        close=MarginalInterval(lower=19_500, upper=20_600),
        method="test",
    )
    return InstrumentOHLCForecast(
        instrument=instrument,
        contract=contract,
        reference_contract=contract,
        reference_date=date(2026, 10, 15),
        previous_close=20_000,
        point=point,
        intervals=intervals,
        model=ModelIdentity(
            model_name="locked-baseline",
            model_version="1",
            checkpoint_id="test",
            representation="gbud",
        ),
    )


def _snapshot(path):
    forecasts = (_forecast("TAIEX"), _forecast("TX"))
    quality = DataQualityEvidence(status="PASS", report_hash="6" * 64)
    sources = (
        SourceEvidence(
            source_id="test-source",
            content_hash="7" * 64,
            available_at=_at("08:20"),
        ),
    )
    forecast_id = deterministic_forecast_id(
        trading_date=DAY,
        cutoff_at=_at("08:30"),
        forecasts=forecasts,
        data_quality=quality,
        sources=sources,
        data_manifest_hash="8" * 64,
        config_hash="9" * 64,
        code_version="test",
    )
    snapshot = seal_snapshot(OHLCForecastSnapshotBodyV1(
        forecast_id=forecast_id,
        trading_date=DAY,
        cutoff_at=_at("08:30"),
        generated_at=_at("08:31"),
        persisted_at=_at("08:35"),
        timeliness="ON_TIME",
        forecasts=forecasts,
        data_quality=quality,
        sources=sources,
        data_manifest_hash="8" * 64,
        config_hash="9" * 64,
        code_version="test",
    ))
    freeze_snapshot(snapshot, path)
    return snapshot


def test_live_preopen_rejects_wrong_window_and_preserves_failure(tmp_path):
    with pytest.raises(DailyRunFailed) as raised:
        run_live_preopen(
            trading_date=DAY,
            base_dataset_dir=tmp_path / "base",
            baseline_path=tmp_path / "baseline.json",
            final_summary_path=tmp_path / "final.json",
            config={"cutoff": "08:30"},
            config_path=tmp_path / "config.json",
            data_root=tmp_path / "data",
            output_root=tmp_path / "daily",
            writer=FakeWriter(),
            clock=lambda: _at("08:10"),
        )
    evidence = json.loads(raised.value.evidence_path.read_text())
    assert evidence["status"] == "FAIL"
    assert evidence["failed_stage"] == "validate_live_window"
    assert evidence["production_sheet_written"] is False


def test_live_preopen_composes_capture_forecast_and_staging_write(monkeypatch, tmp_path):
    daily_dataset = tmp_path / "daily-dataset"
    audit = {"status": "PASS", "report_hash": "a" * 64, "issues": []}
    monkeypatch.setattr(
        daily_orchestration,
        "capture_preopen_dataset",
        lambda **_kwargs: (daily_dataset, audit),
    )

    def fake_forecast(**kwargs):
        batch = _batch()
        batch_path = kwargs["output_root"] / "sheet-dry-runs" / "batch.json"
        save_sheet_dry_run(batch, batch_path)
        attempt = kwargs["output_root"] / "attempt.json"
        write_once(attempt, {
            "status": "READY_FOR_STAGING_SHEET_WRITE",
            "sheet_write_plan": str(batch_path),
        })
        return attempt

    monkeypatch.setattr(daily_orchestration, "run_preopen_staging", fake_forecast)
    moments = iter((_at("08:25"), _at("08:25"), _at("08:30"), _at("08:31")))
    path = run_live_preopen(
        trading_date=DAY,
        base_dataset_dir=tmp_path / "base",
        baseline_path=tmp_path / "baseline.json",
        final_summary_path=tmp_path / "final.json",
        config={"cutoff": "08:30", "capture_not_before": "08:20"},
        config_path=tmp_path / "config.json",
        data_root=tmp_path / "data",
        output_root=tmp_path / "daily",
        writer=FakeWriter(),
        clock=lambda: next(moments),
        sleeper=lambda _seconds: None,
    )
    result = json.loads(path.read_text())
    assert result["status"] == "STAGING_PREOPEN_E2E_VERIFIED"
    assert result["capture_completed_by_cutoff"] is True
    assert result["staging_sheet_written"] is True
    assert result["production_sheet_written"] is False


def test_head_live_preopen_rejects_wrong_window_and_preserves_failure(tmp_path):
    with pytest.raises(DailyRunFailed) as raised:
        run_head_live_preopen(
            trading_date=DAY,
            base_dataset_dir=tmp_path / "base",
            bundle_path=tmp_path / "bundle.json",
            config={"cutoff": "08:30", "deadline": "08:40"},
            data_root=tmp_path / "data",
            output_root=tmp_path / "daily",
            cache_dir=tmp_path / "cache",
            clock=lambda: _at("08:10"),
        )
    evidence = json.loads(raised.value.evidence_path.read_text())
    assert evidence["status"] == "FAIL"
    assert evidence["failed_stage"] == "validate_head_live_window"
    assert evidence["production_sheet_written"] is False


def test_head_live_preopen_captures_and_freezes_before_deadline(monkeypatch, tmp_path):
    from tw_scenario.quant import head_daily, timesfm3

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({
        "TAIEX": {"result": "taiex.json", "checkpoint": "taiex.safetensors"},
        "TX": {"result": "tx.json", "checkpoint": "tx.safetensors"},
    }))
    daily_dataset = tmp_path / "daily-dataset"
    audit = {"status": "PASS", "report_hash": "a" * 64, "issues": []}
    monkeypatch.setattr(
        daily_orchestration,
        "capture_preopen_dataset",
        lambda **_kwargs: (daily_dataset, audit),
    )
    monkeypatch.setattr(head_daily, "read_head_bundle", lambda _path: {"validated": True})
    monkeypatch.setattr(
        timesfm3,
        "runtime_probe",
        lambda _path: type("Probe", (), {"status": "READY_TO_LOAD"})(),
    )

    def fake_head_daily(_dataset, _bundle, _day, output, _cache):
        _snapshot(output)
        return output

    monkeypatch.setattr(head_daily, "run_head_daily", fake_head_daily)
    moments = iter((_at("08:25"), _at("08:25"), _at("08:30"), _at("08:31")))
    path = run_head_live_preopen(
        trading_date=DAY,
        base_dataset_dir=tmp_path / "base",
        bundle_path=bundle_path,
        config={
            "cutoff": "08:30",
            "deadline": "08:40",
            "capture_not_before": "08:20",
        },
        data_root=tmp_path / "data",
        output_root=tmp_path / "daily",
        cache_dir=tmp_path / "cache",
        clock=lambda: next(moments),
        sleeper=lambda _seconds: None,
    )
    result = json.loads(path.read_text())
    assert result["status"] == "HEAD_FORWARD_SHADOW_FROZEN"
    assert result["capture_completed_by_cutoff"] is True
    assert result["timeliness"] == "ON_TIME"
    assert result["genuine_forward"] is True
    assert result["development_gate"] == "FAIL"
    assert result["production_switch"] is False
    assert result["production_sheet_written"] is False


def test_live_postclose_appends_records_and_writes_staging(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = _snapshot(snapshot_path)
    morning = _batch().model_copy(update={
        "forecast_id": snapshot.forecast_id,
        "snapshot_hash": snapshot.snapshot_hash,
    })
    morning_path = tmp_path / "morning.json"
    save_sheet_dry_run(morning, morning_path)

    def actual(instrument, contract=""):
        return Bar(
            instrument=instrument,
            contract=contract,
            trading_date=DAY,
            session="day",
            open=20_010,
            high=20_250,
            low=19_750,
            close=20_120,
            session_start_at=_at("08:45" if instrument == "TX" else "09:00"),
            session_end_at=_at("13:45" if instrument == "TX" else "13:30"),
            available_at=_at("18:00"),
            fetched_at=_at("18:05"),
            availability_basis="test live fetch",
            source="test",
            source_hash=("a" if instrument == "TAIEX" else "b") * 64,
        )

    monkeypatch.setattr(
        daily_orchestration,
        "capture_postclose_bars",
        lambda *_args: (
            {"TAIEX": actual("TAIEX"), "TX": actual("TX", "202610")},
            {"status": "PASS", "report_hash": "c" * 64, "issues": []},
        ),
    )
    moments = iter((_at("18:10"), _at("18:11"), _at("18:12")))
    evidence_path = run_live_postclose(
        snapshot_path=snapshot_path,
        daily_dataset_dir=tmp_path / "daily-dataset",
        morning_batch_path=morning_path,
        data_root=tmp_path / "data",
        output_root=tmp_path / "daily",
        writer=FakeWriter(),
        config={"postclose_not_before": "18:00"},
        clock=lambda: next(moments),
    )
    result = json.loads(evidence_path.read_text())
    assert result["status"] == "STAGING_POSTCLOSE_E2E_VERIFIED"
    assert result["staging_sheet_written"] is True
    assert len(list((tmp_path / "daily" / "realizations-v1").rglob("*.json"))) == 2
    assert len(list((tmp_path / "daily" / "evaluations-v1").rglob("*.json"))) == 2
