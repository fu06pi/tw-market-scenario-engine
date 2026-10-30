from __future__ import annotations

from datetime import date, datetime, time

from .backtest import walk_forward
from .contracts import OHLC, TZ, Prices, at, digest, encode
from .daily import (
    DataQualityEvidence,
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
from .head_fair_gate import fixed_comparator_rows


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


def _fixed_comparator_prediction(data, instrument, policy, day, config):
    """Run the frozen primary/fallback policy and calibrate its published path."""
    days = data.calendar.days
    index = days.index(day)
    start = days[max(0, index - config["calibration_window"] - 20)]
    previous_day = days[index - 1]
    primary_rows = walk_forward(
        data,
        instrument,
        policy["primary_model"],
        policy["representation"],
        start,
        previous_day,
        config,
    )
    fallback_rows = walk_forward(
        data,
        instrument,
        policy["fallback_model"],
        policy["representation"],
        start,
        previous_day,
        config,
    )
    rows = fixed_comparator_rows(primary_rows, fallback_rows, policy)
    residuals = []
    for row in rows:
        if row["status"] != "OK":
            continue
        previous = row["previous_close"]
        actual = Prices(**row["actual"])
        predicted = Prices(**row["predicted"])
        residuals.append(
            encode(actual, previous, policy["representation"])
            - encode(predicted, previous, policy["representation"])
        )
    residuals = residuals[-config["calibration_window"] :]

    fallback_used = False
    primary_error = None
    try:
        prices, target, previous, _ = predict(
            data,
            instrument,
            day,
            policy["primary_model"],
            policy["representation"],
            config,
        )
        model_name = policy["primary_model"]
    except Exception as exc:  # noqa: BLE001 - the frozen policy requires same-day fallback
        primary_error = f"{type(exc).__name__}: {exc}"
        prices, target, previous, _ = predict(
            data,
            instrument,
            day,
            policy["fallback_model"],
            policy["representation"],
            config,
        )
        model_name = policy["fallback_model"]
        fallback_used = True

    interval = calibrated_interval(
        target,
        previous,
        policy["representation"],
        residuals,
        config["min_calibration"],
    )
    if interval is None:
        raise ValueError("daily interval calibration has insufficient causal residuals")
    for field in OHLC:
        point = getattr(prices, field)
        interval[field] = [min(point, interval[field][0]), max(point, interval[field][1])]
    return prices, interval, previous, model_name, fallback_used, primary_error


def build_locked_baseline_body(
    data,
    manifest: dict,
    baseline_lock: dict,
    config: dict,
    trading_date: date,
    generated_at: datetime,
    persisted_at: datetime | None,
    code_version: str,
    final_summary_hash: str,
    data_quality: DataQualityEvidence | None = None,
) -> OHLCForecastSnapshotBodyV1:
    cutoff = at(trading_date, config["cutoff"])
    if generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    if persisted_at is not None and persisted_at.utcoffset() is None:
        raise ValueError("persisted_at must be timezone-aware")
    if generated_at < cutoff:
        raise ValueError("daily forecast generation cannot begin before the 08:30 cutoff")
    for timestamp, name in ((generated_at, "generated_at"),):
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
                selection_reason="locked TimesFM final acceptance gate failed",
                runtime_fallback_chain=(choice["model"],),
                fallback_used=False,
            ),
        ))
    night = data.get("TX", data.contract("TX", trading_date), "night", trading_date, cutoff)
    sources[f"TX-night-{trading_date}"] = night
    source_evidence = tuple(
        SourceEvidence(source_id=source_id, content_hash=bar.source_hash, available_at=bar.available_at)
        for source_id, bar in sorted(sources.items())
    )
    quality = data_quality or DataQualityEvidence(
        status="WARN",
        report_hash=final_summary_hash,
        issues=("retrospective shadow; historical availability timestamps are assumptions",),
    )
    manifest_hash = digest(manifest)
    config_hash = digest(config)
    persisted_at = persisted_at or datetime.now(TZ)
    if persisted_at < generated_at:
        raise ValueError("persisted_at must not precede generated_at")
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


def build_fixed_comparator_body(
    data,
    manifest: dict,
    comparator_policy: dict,
    config: dict,
    trading_date: date,
    generated_at: datetime,
    persisted_at: datetime | None,
    code_version: str,
    final_summary_hash: str,
    data_quality: DataQualityEvidence,
) -> OHLCForecastSnapshotBodyV1:
    """Build the research-forward snapshot from the frozen fair comparator policy."""
    cutoff = at(trading_date, config["cutoff"])
    if generated_at.utcoffset() is None or generated_at < cutoff:
        raise ValueError("daily forecast generation must be timezone-aware and after cutoff")
    if trading_date not in data.calendar.days:
        raise ValueError("forecast date is not an exchange trading day")
    reference_date = data.calendar.previous(trading_date)
    forecasts = []
    sources = {}
    for instrument in ("TAIEX", "TX"):
        prices, interval, previous, selected, fallback_used, primary_error = (
            _fixed_comparator_prediction(
                data, instrument, comparator_policy, trading_date, config
            )
        )
        contract = data.contract(instrument, trading_date)
        reference = data.reference(instrument, trading_date)
        sources[f"{instrument}-reference-{reference_date}"] = reference
        forecasts.append(
            InstrumentOHLCForecast(
                instrument=instrument,
                contract=contract or None,
                reference_contract=contract or None,
                reference_date=reference_date,
                previous_close=previous,
                point=OHLCValues(**prices.model_dump()),
                intervals=OHLCMarginalIntervals80(
                    **{
                        field: MarginalInterval(
                            lower=interval[field][0], upper=interval[field][1]
                        )
                        for field in OHLC
                    },
                    method="frozen comparator trailing residual-vector reconstruction",
                ),
                model=ModelIdentity(
                    model_name=selected,
                    model_version=comparator_policy["version"],
                    checkpoint_id=digest(comparator_policy),
                    representation=comparator_policy["representation"],
                    selection_reason=(
                        "fixed fair comparator after all trained candidates failed"
                    ),
                    runtime_fallback_chain=(
                        comparator_policy["primary_model"],
                        comparator_policy["fallback_model"],
                    ),
                    fallback_used=fallback_used,
                    fallback_reason=primary_error,
                ),
            )
        )
    night = data.get("TX", data.contract("TX", trading_date), "night", trading_date, cutoff)
    sources[f"TX-night-{trading_date}"] = night
    source_evidence = tuple(
        SourceEvidence(
            source_id=source_id,
            content_hash=bar.source_hash,
            available_at=bar.available_at,
        )
        for source_id, bar in sorted(sources.items())
    )
    manifest_hash = digest(manifest)
    config_hash = digest(
        {
            "research_config": config,
            "comparator_policy": comparator_policy,
            "failed_candidate_summary_hash": final_summary_hash,
        }
    )
    persisted_at = persisted_at or datetime.now(TZ)
    if persisted_at.utcoffset() is None or persisted_at < generated_at:
        raise ValueError("persisted_at must be aware and not precede generated_at")
    forecast_id = deterministic_forecast_id(
        trading_date=trading_date,
        cutoff_at=cutoff,
        forecasts=forecasts,
        data_quality=data_quality,
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
        data_quality=data_quality,
        sources=source_evidence,
        data_manifest_hash=manifest_hash,
        config_hash=config_hash,
        code_version=code_version,
    )
