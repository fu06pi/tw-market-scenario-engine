from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from tw_scenario.quant import daily_orchestration
from tw_scenario.quant.contracts import TZ, write_once
from tw_scenario.quant.daily import V0MorningSheetRow, V0SheetDryRunBatch, save_sheet_dry_run
from tw_scenario.quant.daily_orchestration import DailyRunFailed, run_live_preopen

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
