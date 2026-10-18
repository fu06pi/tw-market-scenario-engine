"""Research-only checkpoint inference using the existing immutable OHLC contract."""
from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path

import numpy as np

from .backtest import _git_state, load_dataset
from .contracts import OHLC, TZ, at, digest
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
    freeze_snapshot,
    seal_snapshot,
)
from .timesfm3 import load_pinned_forecaster, runtime_probe
from .timesfm_head_training import (
    build_head_inference_sample,
    predict_head,
    predict_joint_head,
    restore_output_head,
)


def read_head_bundle(path: Path) -> dict:
    """Resolve portable result/checkpoint paths relative to the bundle file."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if set(bundle) not in ({"TAIEX", "TX"}, {"JOINT"}):
        raise ValueError("bundle requires TAIEX and TX artifacts, or one JOINT artifact")
    artifacts = {}
    for instrument, entry in bundle.items():
        if set(entry) != {"result", "checkpoint"}:
            raise ValueError("each artifact requires exactly result and checkpoint paths")
        result_path = path.parent / entry["result"]
        checkpoint = path.parent / entry["checkpoint"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity", {}).get("instrument") != instrument:
            raise ValueError("artifact instrument does not match bundle")
        artifacts[instrument] = {"result": result, "checkpoint": checkpoint}
    return artifacts


def build_head_body(data, manifest, artifacts, model, day, generated_at, code_version):
    """Forecast both markets; never access same-day labels or modify a schedule."""
    cutoff = at(day)
    if generated_at.utcoffset() is None or generated_at < cutoff:
        raise ValueError("generation must be timezone-aware and at/after 08:30 cutoff")
    if set(artifacts) not in ({"TAIEX", "TX"}, {"JOINT"}):
        raise ValueError("complete dual-market artifacts required")
    if not np.allclose(model.quantiles, np.arange(1, 10) / 10, rtol=0, atol=1e-7):
        raise ValueError("unexpected TimesFM quantile grid")
    forecasts = {}
    artifact_ids = {}
    for instrument, artifact in artifacts.items():
        result = artifact["result"]
        identity = result["identity"]
        if identity["instrument"] != instrument:
            raise ValueError("artifact instrument mismatch")
        expected_names = (
            {market: [f"{market.lower()}__{name}" for name in data.research_feature_names()]
             for market in ("TAIEX", "TX")}
            if instrument == "JOINT" else data.research_feature_names()
        )
        if identity["feature_names"] != expected_names:
            raise ValueError("checkpoint feature schema/order mismatch")
        mean, scale = restore_output_head(
            model, artifact["checkpoint"], result, forecast_day=day
        )
        sample = build_head_inference_sample(data, instrument, day, identity["spec"]["context"])
        predictions = (predict_joint_head(model, sample, mean, scale)
                       if instrument == "JOINT" else
                       {instrument: predict_head(model, sample, mean, scale)})
        artifact_ids[instrument] = digest(result)
        for market, (point, interval, _) in predictions.items():
            contract = data.contract(market, day) or None
            forecasts[market] = InstrumentOHLCForecast(
                instrument=market, contract=contract, reference_contract=contract,
                reference_date=data.calendar.previous(day),
                previous_close=float(data.reference(market, day).close),
                point=OHLCValues(**point.model_dump()),
                intervals=OHLCMarginalIntervals80(
                    **{field: MarginalInterval(lower=interval[field][0], upper=interval[field][1])
                       for field in OHLC},
                    method="uncalibrated transformed marginal quantile envelope; coverage unverified",
                ),
                model=ModelIdentity(
                    model_name="timesfm3-custom-output-head",
                    model_version=digest(identity),
                    checkpoint_id=result["checkpoint"]["sha256"], representation="gbud",
                    selection_reason="explicit research artifact; promotion and forward gate unverified",
                    runtime_fallback_chain=("timesfm3-custom-output-head",),
                ),
            )
    sources = []
    for market in ("TAIEX", "TX"):
        bar = data.reference(market, day)
        sources.append(SourceEvidence(source_id=f"{market}-same-contract-reference",
                                      content_hash=bar.source_hash, available_at=bar.available_at))
    night = data.get("TX", data.contract("TX", day), "night", day, cutoff)
    sources.append(SourceEvidence(source_id="TX-completed-night", content_hash=night.source_hash,
                                  available_at=night.available_at))
    quality = DataQualityEvidence(
        status="WARN", report_hash=digest({"manifest": manifest, "artifacts": artifact_ids}),
        issues=("Research checkpoint; development/forward acceptance not established.",
                "Nominal 80% intervals are uncalibrated; actual coverage is unverified.",
                "Historical availability assumptions require separate point-in-time verification.",
                "Context sessions exclude the additional rolling-feature raw-data lookback."),
    )
    common = {"trading_date": day, "cutoff_at": cutoff,
                  "forecasts": tuple(forecasts[market] for market in ("TAIEX", "TX")),
                  "data_quality": quality, "sources": tuple(sources), "data_manifest_hash": digest(manifest),
                  "config_hash": digest(artifact_ids), "code_version": code_version}
    persisted_at = datetime.now(TZ)
    return OHLCForecastSnapshotBodyV1(
        **common, forecast_id=deterministic_forecast_id(**common),
        generated_at=generated_at, persisted_at=persisted_at,
        timeliness="ON_TIME" if persisted_at <= datetime.combine(day, time(8, 40), TZ) else "LATE",
    )


def run_head_daily(dataset_dir: Path, bundle_path: Path, day: date,
                   output: Path, cache_dir: Path) -> Path:
    if output.exists():
        raise FileExistsError("frozen forecast already exists")
    artifacts = read_head_bundle(bundle_path)
    data, _ = load_dataset(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    generated_at = datetime.now(TZ)
    if generated_at < at(day):
        raise ValueError("cannot generate before trading-day 08:30 cutoff")
    probe = runtime_probe(cache_dir)
    if probe.status != "READY_TO_LOAD":
        raise RuntimeError(f"TimesFM load blocked: {probe.status}")
    forecaster = load_pinned_forecaster(cache_dir, local_files_only=True)
    body = build_head_body(data, manifest, artifacts, forecaster.model, day, generated_at,
                           digest(_git_state(Path(__file__).resolve().parents[3])))
    return freeze_snapshot(seal_snapshot(body), output)
