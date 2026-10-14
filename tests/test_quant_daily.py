from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from tw_scenario.quant import daily_runner
from tw_scenario.quant.contracts import TZ, Prices
from tw_scenario.quant.daily import (
    V0_HEADER_FINGERPRINT,
    V0_HEADERS,
    DataQualityEvidence,
    InstrumentOHLCForecast,
    MarginalInterval,
    ModelIdentity,
    OHLCForecastSnapshotBodyV1,
    OHLCForecastSnapshotV1,
    OHLCMarginalIntervals80,
    OHLCRealizationV1,
    OHLCValues,
    SourceEvidence,
    V0NarrativePayload,
    append_evaluation,
    append_realization,
    build_v0_narrative,
    build_v0_sheet_dry_run,
    deterministic_forecast_id,
    evaluate_realization,
    freeze_snapshot,
    header_fingerprint,
    load_snapshot,
    save_sheet_dry_run,
    seal_snapshot,
    verify_narrative_snapshot_link,
)

DAY = date(2026, 10, 14)
ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64
TWO_HASH = "2" * 64


def _at(clock: str) -> datetime:
    return datetime.combine(DAY, datetime.strptime(clock, "%H:%M:%S").time(), TZ)


def _interval(lower: float, upper: float) -> MarginalInterval:
    return MarginalInterval(lower=lower, upper=upper)


def _forecast(instrument: str) -> InstrumentOHLCForecast:
    if instrument == "TAIEX":
        return InstrumentOHLCForecast(
            instrument="TAIEX",
            contract=None,
            reference_contract=None,
            reference_date=DAY - timedelta(days=1),
            previous_close=27_000.25,
            point=OHLCValues(
                open=27_110.5,
                high=27_380.75,
                low=26_955.25,
                close=27_225.5,
            ),
            intervals=OHLCMarginalIntervals80(
                open=_interval(26_900.25, 27_300.75),
                high=_interval(27_100.5, 27_600.25),
                low=_interval(26_700.25, 27_100.5),
                close=_interval(26_900.75, 27_500.25),
                method="walk-forward residual vector",
            ),
            model=ModelIdentity(
                model_name="night-regression",
                model_version="1",
                checkpoint_id="validation-best-001",
                representation="gbud",
            ),
        )
    return InstrumentOHLCForecast(
        instrument="TX",
        contract="202610",
        reference_contract="202610",
        reference_date=DAY - timedelta(days=1),
        previous_close=27_015.0,
        point=OHLCValues(open=27_125.0, high=27_410.0, low=26_940.0, close=27_240.0),
        intervals=OHLCMarginalIntervals80(
            open=_interval(26_880.0, 27_350.0),
            high=_interval(27_120.0, 27_700.0),
            low=_interval(26_650.0, 27_100.0),
            close=_interval(26_900.0, 27_520.0),
            method="walk-forward residual vector",
        ),
        model=ModelIdentity(
            model_name="historical-range",
            model_version="1",
            checkpoint_id="validation-best-002",
            representation="relative",
        ),
    )


def _body(
    *,
    persisted_at: datetime | None = None,
    timeliness: str = "ON_TIME",
    forecasts: tuple[InstrumentOHLCForecast, ...] | None = None,
    data_quality: DataQualityEvidence | None = None,
) -> OHLCForecastSnapshotBodyV1:
    forecasts = forecasts or (_forecast("TAIEX"), _forecast("TX"))
    data_quality = data_quality or DataQualityEvidence(
        status="PASS", report_hash=ZERO_HASH
    )
    sources = (
        SourceEvidence(
            source_id="finmind-cache-20261014",
            content_hash=ONE_HASH,
            available_at=_at("08:15:00"),
        ),
        SourceEvidence(
            source_id="taifex-official-cache-20261014",
            content_hash=TWO_HASH,
            available_at=_at("08:20:00"),
        ),
    )
    cutoff_at = _at("08:30:00")
    forecast_id = deterministic_forecast_id(
        trading_date=DAY,
        cutoff_at=cutoff_at,
        forecasts=forecasts,
        data_quality=data_quality,
        sources=sources,
        data_manifest_hash=ZERO_HASH,
        config_hash=ONE_HASH,
        code_version="test-commit",
    )
    return OHLCForecastSnapshotBodyV1(
        forecast_id=forecast_id,
        trading_date=DAY,
        cutoff_at=cutoff_at,
        generated_at=_at("08:35:00"),
        persisted_at=persisted_at or _at("08:39:59"),
        timeliness=timeliness,
        forecasts=forecasts,
        data_quality=data_quality,
        sources=sources,
        data_manifest_hash=ZERO_HASH,
        config_hash=ONE_HASH,
        code_version="test-commit",
    )


def test_snapshot_has_exact_markets_same_contract_reference_and_write_once(tmp_path):
    snapshot = seal_snapshot(_body())

    assert [forecast.instrument for forecast in snapshot.forecasts] == ["TAIEX", "TX"]
    tx = snapshot.forecasts[1]
    assert tx.contract == tx.reference_contract == "202610"
    assert tx.reference_date == DAY - timedelta(days=1)
    assert tx.previous_close == 27_015.0
    assert snapshot.timeliness == "ON_TIME"

    path = tmp_path / "snapshots" / f"{snapshot.forecast_id}.json"
    freeze_snapshot(snapshot, path)
    original_bytes = path.read_bytes()
    assert load_snapshot(path) == snapshot
    with pytest.raises(FileExistsError):
        freeze_snapshot(snapshot, path)
    assert path.read_bytes() == original_bytes


def test_snapshot_rejects_wrong_market_order_reference_and_interval():
    body = _body()
    payload = body.model_dump(mode="python")
    payload["forecasts"] = tuple(reversed(body.forecasts))
    with pytest.raises(ValidationError, match="exactly TAIEX then TX"):
        OHLCForecastSnapshotBodyV1.model_validate(payload)

    with pytest.raises(ValidationError, match="same contract"):
        InstrumentOHLCForecast.model_validate(
            {
                **body.forecasts[1].model_dump(mode="python"),
                "reference_contract": "202611",
            }
        )

    tx_payload = body.forecasts[1].model_dump(mode="python")
    tx_payload["intervals"]["close"] = {"lower": 27_300.0, "upper": 27_500.0}
    with pytest.raises(ValidationError, match="close point forecast"):
        InstrumentOHLCForecast.model_validate(tx_payload)

    with pytest.raises(ValidationError, match="greater than 0"):
        MarginalInterval(lower=0, upper=1)


def test_reference_date_and_source_cutoff_are_fail_closed():
    body = _body()
    forecast_payload = body.forecasts[0].model_dump(mode="python")
    forecast_payload["reference_date"] = DAY
    bad_forecast = InstrumentOHLCForecast.model_validate(forecast_payload)
    forecasts = (bad_forecast, body.forecasts[1])
    payload = body.model_dump(mode="python")
    payload["forecasts"] = forecasts
    payload["forecast_id"] = deterministic_forecast_id(
        trading_date=DAY,
        cutoff_at=body.cutoff_at,
        forecasts=forecasts,
        data_quality=body.data_quality,
        sources=body.sources,
        data_manifest_hash=body.data_manifest_hash,
        config_hash=body.config_hash,
        code_version=body.code_version,
    )
    with pytest.raises(ValidationError, match="reference_date must precede"):
        OHLCForecastSnapshotBodyV1.model_validate(payload)

    payload = body.model_dump(mode="python")
    payload["sources"][0]["available_at"] = _at("08:30:01")
    with pytest.raises(ValidationError, match="available by cutoff"):
        OHLCForecastSnapshotBodyV1.model_validate(payload)


def test_forecast_id_and_snapshot_hash_are_deterministic_and_tamper_evident():
    first = seal_snapshot(_body())
    second = seal_snapshot(_body())
    assert first.forecast_id == second.forecast_id
    assert first.snapshot_hash == second.snapshot_hash

    later_persistence = seal_snapshot(_body(persisted_at=_at("08:40:00")))
    assert later_persistence.forecast_id == first.forecast_id
    assert later_persistence.snapshot_hash != first.snapshot_hash

    body_payload = _body().model_dump(mode="python")
    body_payload["forecast_id"] = f"ohlc-v1-{DAY.isoformat()}-{'f' * 64}"
    with pytest.raises(ValidationError, match="deterministic forecast content"):
        OHLCForecastSnapshotBodyV1.model_validate(body_payload)

    snapshot_payload = first.model_dump(mode="python")
    snapshot_payload["forecasts"][0]["point"]["close"] = 27_226.0
    with pytest.raises(ValidationError):
        OHLCForecastSnapshotV1.model_validate(snapshot_payload)


def test_persistence_deadline_uses_persisted_at_and_includes_exact_0840():
    assert _body(persisted_at=_at("08:40:00"), timeliness="ON_TIME").timeliness == "ON_TIME"

    late = _body(persisted_at=_at("08:40:01"), timeliness="LATE")
    assert late.timeliness == "LATE"

    with pytest.raises(ValidationError, match="timeliness must be LATE"):
        _body(persisted_at=_at("08:40:01"), timeliness="ON_TIME")

    payload = _body().model_dump(mode="python")
    payload["generated_at"] = _at("08:40:00")
    payload["persisted_at"] = _at("08:39:59")
    with pytest.raises(ValidationError, match="cutoff <= generated <= persisted"):
        OHLCForecastSnapshotBodyV1.model_validate(payload)


def _realization(snapshot: OHLCForecastSnapshotV1) -> OHLCRealizationV1:
    return OHLCRealizationV1(
        realization_id="tx-20261014-official-v1",
        forecast_id=snapshot.forecast_id,
        snapshot_hash=snapshot.snapshot_hash,
        trading_date=DAY,
        instrument="TX",
        contract="202610",
        actual=OHLCValues(open=27_100.0, high=27_500.0, low=26_900.0, close=27_300.0),
        source_id="taifex-official-20261014",
        source_hash=TWO_HASH,
        observed_at=_at("13:45:00"),
        recorded_at=_at("14:15:00"),
    )


def test_realization_and_evaluation_append_without_changing_prediction(tmp_path):
    snapshot = seal_snapshot(_body())
    snapshot_path = tmp_path / "snapshot.json"
    freeze_snapshot(snapshot, snapshot_path)
    frozen_bytes = snapshot_path.read_bytes()
    frozen_payload = snapshot.model_dump(mode="json")

    realization = _realization(snapshot)
    evaluation = evaluate_realization(snapshot, realization, _at("14:20:00"))
    realization_path = append_realization(realization, tmp_path)
    evaluation_path = append_evaluation(evaluation, tmp_path)

    assert realization_path.exists()
    assert evaluation_path.exists()
    assert evaluation.forecast_id == snapshot.forecast_id
    assert evaluation.snapshot_hash == snapshot.snapshot_hash
    assert evaluation.absolute_error_points.close == 60.0
    assert snapshot.model_dump(mode="json") == frozen_payload
    assert snapshot_path.read_bytes() == frozen_bytes
    with pytest.raises(FileExistsError):
        append_realization(realization, tmp_path)
    with pytest.raises(FileExistsError):
        append_evaluation(evaluation, tmp_path)

    wrong_contract = OHLCRealizationV1.model_validate(
        {**realization.model_dump(mode="python"), "contract": "202611"}
    )
    with pytest.raises(ValueError, match="contract does not match"):
        evaluate_realization(snapshot, wrong_contract, _at("14:20:00"))


def test_narrative_is_deterministic_exact_and_rejects_late_news_or_edits():
    snapshot = seal_snapshot(_body())
    early_news = {
        "news_id": "b",
        "headline": "海外市場背景",
        "summary": "僅作文字背景",
        "source_url": "https://example.test/b",
        "published_at": _at("08:20:00"),
    }
    older_news = {
        "news_id": "a",
        "headline": "官方金融資料",
        "summary": "截止前已發布",
        "source_url": "https://example.test/a",
        "published_at": _at("08:10:00"),
    }
    narrative = build_v0_narrative(snapshot, [early_news, older_news])

    assert narrative == build_v0_narrative(snapshot, [early_news, older_news])
    assert [item.news_id for item in narrative.news_background] == ["a", "b"]
    for item, forecast in zip(narrative.items, snapshot.forecasts, strict=True):
        assert item.forecast_quote == forecast
        assert item.intraday_hypothesis.startswith("待驗證情境假說：")
        assert "上漲機率" not in item.open_hypothesis
        assert "下跌機率" not in item.close_hypothesis
        for value in (
            forecast.previous_close,
            forecast.point.open,
            forecast.point.high,
            forecast.point.low,
            forecast.point.close,
        ):
            assert str(value) in (
                item.open_hypothesis + item.intraday_hypothesis + item.close_hypothesis
            )
    assert verify_narrative_snapshot_link(snapshot, narrative)

    edited_payload = narrative.model_dump(mode="python")
    edited_payload["items"][0]["open_hypothesis"] += " 另編數字 99999。"
    edited = V0NarrativePayload.model_validate(edited_payload)
    assert not verify_narrative_snapshot_link(snapshot, edited)

    late_news = {**early_news, "news_id": "late", "published_at": _at("08:30:01")}
    with pytest.raises(ValueError, match="after cutoff"):
        build_v0_narrative(snapshot, [late_news])


def test_sheet_dry_run_preserves_ap_mapping_and_is_file_only_write_once(tmp_path):
    snapshot = seal_snapshot(_body())
    narrative = build_v0_narrative(snapshot)
    batch = build_v0_sheet_dry_run(snapshot, narrative, V0_HEADERS, start_row=42)

    assert header_fingerprint(V0_HEADERS) == V0_HEADER_FINGERPRINT
    assert batch.mode == "DRY_RUN_ONLY"
    assert batch.sheet_name == "每日預測與覆盤"
    assert batch.write_range == "'每日預測與覆盤'!A42:P43"
    assert batch.morning_write_columns == "A:F,O"
    assert batch.postclose_reserved_columns == "G:N,P"
    for row in batch.rows:
        assert len(row.values) == 16
        assert all(row.values[index] for index in range(6))
        assert all(row.values[index] == "" for index in range(6, 14))
        assert row.values[14]
        assert row.values[15] == ""

    path = tmp_path / "sheet-dry-run.json"
    save_sheet_dry_run(batch, path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_sheet_dry_run(batch, path)
    assert path.read_bytes() == original

    wrong_headers = list(V0_HEADERS)
    wrong_headers[5] = "自行改名"
    with pytest.raises(ValueError, match="header fingerprint mismatch"):
        build_v0_sheet_dry_run(snapshot, narrative, wrong_headers, start_row=42)
    edited_payload = narrative.model_dump(mode="python")
    edited_payload["items"][0]["close_hypothesis"] += " 另編收盤 99999。"
    edited = V0NarrativePayload.model_validate(edited_payload)
    with pytest.raises(ValueError, match="does not quote"):
        build_v0_sheet_dry_run(snapshot, edited, V0_HEADERS, start_row=42)


def test_data_quality_failure_cannot_generate_publishable_narrative():
    failed = DataQualityEvidence(status="FAIL", report_hash=ZERO_HASH, issues=("missing TX",))
    snapshot = seal_snapshot(_body(data_quality=failed))
    with pytest.raises(ValueError, match="DQ FAIL"):
        build_v0_narrative(snapshot)


def test_daily_runner_uses_causal_rows_and_expands_interval_to_point(monkeypatch):
    class Calendar:
        def __init__(self):
            self.days = [date(2026, 10, 6), date(2026, 10, 7), DAY]

    data = type("Data", (), {"calendar": Calendar()})()
    rows = [{
        "status": "OK",
        "previous_close": 100.0,
        "actual": {"open": 101.0, "high": 103.0, "low": 99.0, "close": 102.0},
        "predicted": {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0},
    }]
    monkeypatch.setattr(daily_runner, "walk_forward", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        daily_runner,
        "predict",
        lambda *_args, **_kwargs: (
            Prices(open=110.0, high=115.0, low=105.0, close=112.0),
            np.zeros(4),
            100.0,
            [],
        ),
    )
    monkeypatch.setattr(
        daily_runner,
        "calibrated_interval",
        lambda *_args, **_kwargs: {
            "open": [100.0, 105.0], "high": [100.0, 110.0],
            "low": [100.0, 104.0], "close": [100.0, 108.0],
        },
    )
    prices, interval, previous = daily_runner._daily_prediction(
        data,
        "TAIEX",
        {"model": "zero", "representation": "relative"},
        DAY,
        {"calibration_window": 126, "min_calibration": 1},
    )
    assert previous == 100.0
    for field in ("open", "high", "low", "close"):
        assert interval[field][0] <= getattr(prices, field) <= interval[field][1]
