from __future__ import annotations

from datetime import date, datetime, time

from .backtest import walk_forward
from .contracts import OHLC, TZ, Prices, at, digest, encode
from .daily import (
    DataQualityEvidence,
    FallbackInfo,
    InstrumentOHLCForecast,
    MarginalInterval,
    ModelIdentity,
    OHLCForecastSnapshotBodyV1,
    OHLCMarginalIntervals80,
    OHLCValues,
    SourceEvidence,
    deterministic_forecast_id,
)
from .forecast import calibrated_interval, predict


def _daily_prediction(data, instrument, choice, day, config):
    days = data.calendar.days
    index = days.index(day)
    start = days[max(0, index - config["calibration_window"] - 20)]
    previous_day = days[index - 1]
    rows = walk_forward(
        data,
        instrument,
        choice["model"],
        choice["representation"],
        start,
        previous_day,
        config,
    )
    residuals = []
    for row in rows:
        if row["status"] != "OK":
            continue
        previous = row["previous_close"]
        actual = Prices(**row["actual"])
        predicted = Prices(**row["predicted"])
        residuals.append(
            encode(actual, previous, choice["representation"])
            - encode(predicted, previous, choice["representation"])
        )
    residuals = residuals[-config["calibration_window"] :]
    prices, target, previous, _ = predict(
        data,
        instrument,
        day,
        choice["model"],
        choice["representation"],
        config,
    )
    interval = calibrated_interval(
        target,
        previous,
        choice["representation"],
        residuals,
        config["min_calibration"],
    )
    if interval is None:
        raise ValueError("daily interval calibration has insufficient causal residuals")
    # The published point must always be inside each marginal interval.
    for field in OHLC:
        point = getattr(prices, field)
        interval[field] = [min(point, interval[field][0]), max(point, interval[field][1])]
    return prices, interval, previous


def build_locked_baseline_body(
    data,
    manifest: dict,
    baseline_lock: dict,
    config: dict,
    trading_date: date,
    generated_at: datetime,
    persisted_at: datetime,
    code_version: str,
    final_summary_hash: str,
) -> OHLCForecastSnapshotBodyV1:
    cutoff = at(trading_date, config["cutoff"])
    for timestamp, name in ((generated_at, "generated_at"), (persisted_at, "persisted_at")):
        if timestamp.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
    forecasts = []
    sources = {}
    reference_date = data.calendar.previous(trading_date)
    for instrument in ("TAIEX", "TX"):
        choice = baseline_lock["selected_baselines"][instrument]
        prices, interval, previous = _daily_prediction(
            data, instrument, choice, trading_date, config
        )
        contract = data.contract(instrument, trading_date)
        reference = data.reference(instrument, trading_date)
        sources[f"{instrument}-reference-{reference_date}"] = reference
        forecasts.append(InstrumentOHLCForecast(
            instrument=instrument,
            contract=contract or None,
            reference_contract=contract or None,
            reference_date=reference_date,
            previous_close=previous,
            point=OHLCValues(**prices.model_dump()),
            intervals=OHLCMarginalIntervals80(
                **{
                    field: MarginalInterval(lower=interval[field][0], upper=interval[field][1])
                    for field in OHLC
                },
                method="causal trailing residual-vector reconstruction",
            ),
            model=ModelIdentity(
                model_name=choice["model"],
                model_version="locked-baseline-v1",
                checkpoint_id=digest(choice),
                representation=choice["representation"],
                fallback=FallbackInfo(
                    model_name="timesfm3-validation-locked-candidate",
                    reason="single final-test acceptance gate failed",
                ),
            ),
        ))
    night = data.get("TX", data.contract("TX", trading_date), "night", trading_date, cutoff)
    sources[f"TX-night-{trading_date}"] = night
    source_evidence = tuple(
        SourceEvidence(source_id=source_id, content_hash=bar.source_hash, available_at=bar.available_at)
        for source_id, bar in sorted(sources.items())
    )
    quality = DataQualityEvidence(
        status="WARN",
        report_hash=final_summary_hash,
        issues=("retrospective shadow; historical availability timestamps are assumptions",),
    )
    manifest_hash = digest(manifest)
    config_hash = digest(config)
    forecast_id = deterministic_forecast_id(
        trading_date=trading_date,
        cutoff_at=cutoff,
        forecasts=forecasts,
        data_quality=quality,
        sources=source_evidence,
        data_manifest_hash=manifest_hash,
        config_hash=config_hash,
        code_version=code_version,
    )
    deadline = datetime.combine(trading_date, time(8, 40), TZ)
    return OHLCForecastSnapshotBodyV1(
        forecast_id=forecast_id,
        trading_date=trading_date,
        cutoff_at=cutoff,
        generated_at=generated_at,
        persisted_at=persisted_at,
        timeliness="ON_TIME" if persisted_at <= deadline else "LATE",
        forecasts=tuple(forecasts),
        data_quality=quality,
        sources=source_evidence,
        data_manifest_hash=manifest_hash,
        config_hash=config_hash,
        code_version=code_version,
    )
