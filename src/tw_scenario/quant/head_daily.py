"""Research-only checkpoint inference using the existing immutable OHLC contract."""
from __future__ import annotations

import hashlib
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
from .head_calibration import (
    apply_head_calibration,
    load_head_calibration_artifact,
    verify_head_calibration_availability,
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
        required = {"result", "checkpoint"}
        optional = {"fold", "calibration"}
        if not required <= set(entry) <= required | optional:
            raise ValueError(
                "artifact requires result/checkpoint paths and optional fold/calibration"
            )
        result_path = path.parent / entry["result"]
        checkpoint = path.parent / entry["checkpoint"]
        source_result = json.loads(result_path.read_text(encoding="utf-8"))
        source_result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
        result = source_result
        if "fold" in entry:
            result = read_walk_forward_fold(result_path, result, checkpoint, entry["fold"])
        if result.get("identity", {}).get("instrument") != instrument:
            raise ValueError("artifact instrument does not match bundle")
        calibration = None
        if "calibration" in entry:
            if instrument == "JOINT":
                raise ValueError("joint daily calibration artifacts are not implemented")
            calibration_path = path.parent / entry["calibration"]
            calibration_artifact, calibration_sha256 = load_head_calibration_artifact(
                calibration_path,
                instrument=instrument,
                source_identity=source_result["identity"],
                source_result_sha256=source_result_sha256,
                source_dir=result_path.parent,
                source_fold_results=source_result.get("fold_results"),
            )
            calibration = {
                "artifact": calibration_artifact,
                "sha256": calibration_sha256,
            }
        artifacts[instrument] = {
            "result": result,
            "checkpoint": checkpoint,
            "calibration": calibration,
        }
    calibration_flags = [artifact["calibration"] is not None for artifact in artifacts.values()]
    if any(calibration_flags) and not all(calibration_flags):
        raise ValueError("dual-market bundle requires calibration artifacts for both markets")
    return artifacts



def read_walk_forward_fold(result_path, source, checkpoint, fold_index):
    """Expose a real fold as a research inference artifact, retaining its provenance."""
    if source.get("schema_version") != "timesfm3-output-head-walk-forward-v1":
        raise ValueError("requires walk-forward source result")
    if type(fold_index) is not int or not 1 <= fold_index <= len(source["fold_results"]):
        raise ValueError("invalid walk-forward fold index")
    path = result_path.parent / f"fold-{fold_index}.json"
    saved = json.loads(path.read_text())
    summary = source["fold_results"][fold_index - 1]
    if (saved["identity_hash"] != digest(source["identity"])
            or saved["fold"] != fold_index or saved["summary"] != summary):
        raise ValueError("walk-forward fold provenance mismatch")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != summary["checkpoint"]["sha256"]:
        raise ValueError("walk-forward checkpoint hash mismatch")
    declared = source["identity"]["folds"][fold_index - 1]
    if (summary["train_split"] != declared["train"]
            or summary["validation_split"] != declared["validation"]
            or summary["train_split"][1] >= summary["validation_split"][0]):
        raise ValueError("walk-forward temporal provenance mismatch")
    return {
        "schema_version": "timesfm3-walk-forward-fold-inference-v1",
        "identity": source["identity"], "weight_repo": source["weight_repo"],
        "checkpoint": summary["checkpoint"], "training": summary["training"],
        # Conservatively include all training/early-stop labels in the fold train range.
        "training_label_end": summary["train_split"][1],
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "source_fold_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_fold": fold_index, "source_development_status": source["status"],
        "research_only": True, "production_candidate": False,
    }

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
    calibration_usage = {}
    for instrument, artifact in artifacts.items():
        result = artifact["result"]
        identity = result["identity"]
        if identity["instrument"] != instrument:
            raise ValueError("artifact instrument mismatch")
        expected_names = (
            {market: [f"{market.lower()}__{name}" for name in data.research_feature_names()]
             for market in ("TAIEX", "TX")}
            if instrument == "JOINT" else list(data.research_feature_names())
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
        artifact_ids[instrument] = {
            "model_result": digest(result),
            "calibration_artifact": (
                artifact["calibration"]["sha256"] if artifact["calibration"] else None
            ),
        }
        for market, (point, interval, _) in predictions.items():
            contract = data.contract(market, day) or None
            previous_close = float(data.reference(market, day).close)
            interval_method = (
                "uncalibrated transformed marginal quantile envelope; coverage unverified"
            )
            if artifact["calibration"] is not None:
                verify_head_calibration_availability(
                    data,
                    market,
                    artifact["calibration"]["artifact"],
                    day,
                )
                interval, usage = apply_head_calibration(
                    point.model_dump(),
                    previous_close,
                    artifact["calibration"]["artifact"],
                    day,
                )
                usage["artifact_sha256"] = artifact["calibration"]["sha256"]
                calibration_usage[market] = usage
                interval_method = (
                    "causal signed price residual marginals v1; "
                    f"{usage['samples']} labels through {usage['latest_label_date']}; "
                    "marginal not joint OHLC; "
                    f"artifact sha256 {usage['artifact_sha256']}"
                )
            forecasts[market] = InstrumentOHLCForecast(
                instrument=market, contract=contract, reference_contract=contract,
                reference_date=data.calendar.previous(day),
                previous_close=previous_close,
                point=OHLCValues(**point.model_dump()),
                intervals=OHLCMarginalIntervals80(
                    **{field: MarginalInterval(lower=interval[field][0], upper=interval[field][1])
                       for field in OHLC},
                    method=interval_method,
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
    interval_issue = (
        "Nominal 80% marginal intervals use timestamp-filtered causal residual artifacts; "
        "historical coverage does not establish forward acceptance."
        if calibration_usage else
        "Nominal 80% intervals are uncalibrated; actual coverage is unverified."
    )
    quality = DataQualityEvidence(
        status="WARN", report_hash=digest({"manifest": manifest, "artifacts": artifact_ids}),
        issues=("Research checkpoint; development/forward acceptance not established.",
                "Checkpoint replay remains a separate verification; runtime metadata alone is insufficient.",
                *(('Legacy head lacks recorded numeric runtime; exact replay is unverified.',)
                  if any('numeric_runtime' not in a['result']['training']
                         for a in artifacts.values()) else ()),
                interval_issue,
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


def verify_head_replay(dataset_dir, bundle_path, day, output, cache_dir, report_path):
    """Replay fold predictions through the daily snapshot path; never claim live timeliness."""
    from .contracts import write_once
    from .daily import load_snapshot

    bundle = json.loads(bundle_path.read_text())
    if any('calibration' in entry for entry in bundle.values()):
        raise ValueError('raw checkpoint replay requires a bundle without calibration')
    expected = {}
    for market, entry in bundle.items():
        if market not in ('TAIEX', 'TX') or 'fold' not in entry:
            raise ValueError('replay requires independent market fold artifacts')
        result_path = bundle_path.parent / entry['result']
        fold = json.loads((result_path.parent / f"fold-{entry['fold']}.json").read_text())
        matches = [r for r in fold['candidate_rows'] if r['date'] == str(day)]
        if len(matches) != 1 or matches[0]['status'] != 'OK':
            raise ValueError('replay day has no unique successful saved prediction')
        expected[market] = matches[0]
    path = run_head_daily(dataset_dir, bundle_path, day, output, cache_dir)
    snapshot = load_snapshot(path)
    errors = {}
    for forecast in snapshot.forecasts:
        row = expected[forecast.instrument]
        point_error = max(abs(getattr(forecast.point, f) - row['predicted'][f]) for f in OHLC)
        interval_error = max(
            abs(getattr(getattr(forecast.intervals, f), bound) - row['interval'][f][index])
            for f in OHLC for index, bound in enumerate(('lower', 'upper')))
        errors[forecast.instrument] = {'max_point_error': point_error,
                                       'max_interval_error': interval_error}
    passed = all(max(v.values()) <= 0.001 for v in errors.values())
    report = {'schema_version': 'trained-head-daily-replay-v1',
              'status': 'PASS' if passed else 'FAIL', 'date': str(day),
              'comparison_tolerance_points': 0.001, 'errors': errors,
              'snapshot_hash': snapshot.snapshot_hash, 'forecast_id': snapshot.forecast_id,
              'timeliness': snapshot.timeliness, 'genuine_forward': False,
              'production_switch': False,
              'meaning': 'checkpoint daily replay consistency, not forecast acceptance',
              'checkpoints': {f.instrument: f.model.checkpoint_id for f in snapshot.forecasts}}
    write_once(report_path, report)
    if not passed:
        raise RuntimeError('daily replay differs from saved fold prediction; see report')
    return report
