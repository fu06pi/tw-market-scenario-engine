from __future__ import annotations

import json
import resource
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .backtest import _git_state, _read_manifest, _safe_contract, _select_baseline, load_dataset
from .contracts import TZ, at, decode, digest, encode, write_once
from .evaluation import block_improvement, metrics
from .forecast import calibrated_interval
from .quality import quality_report
from .timesfm3 import TimesFM3Adapter, load_pinned_forecaster

TIMESFM_MODES = ("timesfm_univariate", "timesfm_multivariate")
JOINT_MODE = "timesfm_joint_multivariate"


def _predict(adapter, data, instrument, day, mode, representation, config):
    with_features = mode == "timesfm_multivariate"
    history, features, used = data.history(
        instrument,
        day,
        representation,
        config["window"],
        config["min_history"],
        with_features,
    )
    reference = data.reference(instrument, day).close
    if mode == "timesfm_univariate":
        prediction = adapter.zero_shot_univariate(history)
    elif mode == "timesfm_multivariate":
        prediction = adapter.zero_shot_multivariate(
            history, features, data.features(day)
        )
    else:
        raise ValueError(f"unsupported TimesFM mode: {mode}")
    prices = decode(prediction.point, reference, representation)
    coherent_target = encode(prices, reference, representation)
    return prices, coherent_target, reference, used, prediction.quantiles


def walk_forward_timesfm(
    data,
    adapter,
    instrument: str,
    mode: str,
    representation: str,
    start: date,
    end: date,
    config: dict,
) -> list[dict]:
    if mode not in TIMESFM_MODES:
        raise ValueError("unknown TimesFM mode")
    residual_records = []
    output = []
    eligible_days = [day for day in data.calendar.days if day <= end]
    output_index = next(
        (index for index, day in enumerate(eligible_days) if day >= start),
        len(eligible_days),
    )
    warmup_sessions = config["calibration_window"] + config.get("min_history", 0)
    warmup_index = max(0, output_index - warmup_sessions)
    for day in eligible_days[warmup_index:]:
        try:
            cutoff = at(day, config.get("cutoff", "08:30"))
            eligible_residuals = [
                record["residual"]
                for record in residual_records
                if record["target_date"] < day and record["available_at"] <= cutoff
            ][-config["calibration_window"] :]
            prices, target, previous, _, raw_quantiles = _predict(
                adapter, data, instrument, day, mode, representation, config
            )
            interval = calibrated_interval(
                target,
                previous,
                representation,
                eligible_residuals,
                config["min_calibration"],
            )
            actual = data.actual(instrument, day)
            actual_target = encode(actual, previous, representation)
            if day >= start:
                output.append({
                    "date": str(day),
                    "instrument": instrument,
                    "contract": data.contract(instrument, day),
                    "status": "OK",
                    "previous_close": previous,
                    "predicted": prices.model_dump(),
                    "actual": {
                        name: getattr(actual, name)
                        for name in ("open", "high", "low", "close")
                    },
                    "interval": interval,
                    "interval_method": "joint_trailing_residual_reconstruction",
                    "raw_transformed_quantiles": raw_quantiles.tolist(),
                })
            residual_records.append({
                "target_date": day,
                "available_at": actual.available_at,
                "residual": actual_target - target,
            })
        except Exception as exc:  # noqa: BLE001 - retain every failed expected day
            if day >= start:
                output.append({
                    "date": str(day),
                    "instrument": instrument,
                    "contract": _safe_contract(data, instrument, day),
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return output


def _joint_history(data, day, representation, config):
    taiex_history, taiex_features, taiex_dates = data.history(
        "TAIEX",
        day,
        representation,
        config["window"],
        config["min_history"],
        True,
    )
    tx_history, tx_features, tx_dates = data.history(
        "TX",
        day,
        representation,
        config["window"],
        config["min_history"],
        True,
    )
    if taiex_dates != tx_dates:
        raise ValueError("joint TAIEX/TX history dates are not aligned")
    if taiex_features.shape != tx_features.shape or not (
        taiex_features == tx_features
    ).all():
        raise ValueError("joint TAIEX/TX history features are not aligned")
    return (
        np.column_stack([taiex_history, tx_history]),
        taiex_features,
        taiex_dates,
    )


def _predict_joint(adapter, data, day, representation, config):
    history, features, used = _joint_history(data, day, representation, config)
    prediction = adapter.zero_shot_joint_multivariate(
        history, features, data.features(day)
    )
    result = {}
    for index, instrument in enumerate(("TAIEX", "TX")):
        start = index * 4
        reference = data.reference(instrument, day).close
        prices = decode(prediction.point[start : start + 4], reference, representation)
        target = encode(prices, reference, representation)
        result[instrument] = {
            "prices": prices,
            "target": target,
            "reference": reference,
            "quantiles": prediction.quantiles[start : start + 4],
        }
    return result, used


def walk_forward_timesfm_joint(
    data,
    adapter,
    representation: str,
    start: date,
    end: date,
    config: dict,
) -> dict[str, list[dict]]:
    residual_records = {"TAIEX": [], "TX": []}
    output = {"TAIEX": [], "TX": []}
    eligible_days = [day for day in data.calendar.days if day <= end]
    output_index = next(
        (index for index, day in enumerate(eligible_days) if day >= start),
        len(eligible_days),
    )
    warmup_sessions = config["calibration_window"] + config.get("min_history", 0)
    warmup_index = max(0, output_index - warmup_sessions)
    for day in eligible_days[warmup_index:]:
        try:
            cutoff = at(day, config.get("cutoff", "08:30"))
            predictions, _ = _predict_joint(
                adapter, data, day, representation, config
            )
            pending_residuals = {}
            for instrument in ("TAIEX", "TX"):
                prediction = predictions[instrument]
                eligible_residuals = [
                    record["residual"]
                    for record in residual_records[instrument]
                    if record["target_date"] < day
                    and record["available_at"] <= cutoff
                ][-config["calibration_window"] :]
                interval = calibrated_interval(
                    prediction["target"],
                    prediction["reference"],
                    representation,
                    eligible_residuals,
                    config["min_calibration"],
                )
                actual = data.actual(instrument, day)
                actual_target = encode(
                    actual, prediction["reference"], representation
                )
                pending_residuals[instrument] = {
                    "target_date": day,
                    "available_at": actual.available_at,
                    "residual": actual_target - prediction["target"],
                }
                if day >= start:
                    output[instrument].append({
                        "date": str(day),
                        "instrument": instrument,
                        "contract": data.contract(instrument, day),
                        "status": "OK",
                        "previous_close": prediction["reference"],
                        "predicted": prediction["prices"].model_dump(),
                        "actual": {
                            name: getattr(actual, name)
                            for name in ("open", "high", "low", "close")
                        },
                        "interval": interval,
                        "interval_method": "joint_trailing_residual_reconstruction",
                        "raw_transformed_quantiles": prediction["quantiles"].tolist(),
                    })
            for instrument, record in pending_residuals.items():
                residual_records[instrument].append(record)
        except Exception as exc:  # noqa: BLE001 - retain every failed expected day
            if day >= start:
                for instrument in ("TAIEX", "TX"):
                    output[instrument].append({
                        "date": str(day),
                        "instrument": instrument,
                        "contract": _safe_contract(data, instrument, day),
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    return output


def _quality_gate(dataset_dir: Path, config: dict):
    data, dataset_id = load_dataset(dataset_dir)
    manifest = _read_manifest(dataset_dir)
    report = quality_report(
        data.bars,
        data.calendar,
        date.fromisoformat(config["train_start"]),
        date.fromisoformat(config["test_end"]),
        manifest=manifest,
    )
    if report["status"] != "PASS_RETROSPECTIVE":
        raise ValueError("TimesFM run blocked because dataset quality did not pass")
    return data, dataset_id, report


def run_timesfm_validation(
    dataset_dir: Path,
    config_path: Path,
    output_root: Path,
    cache_dir: Path,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data, dataset_id, data_quality = _quality_gate(dataset_dir, config)
    config_hash = digest(config)
    code_version = _git_state(config_path.parent.parent.parent)
    experiment_id = digest({
        "stage": "timesfm_validation",
        "dataset_id": dataset_id,
        "config_hash": config_hash,
        "code_version": code_version,
    })[:20]
    experiment_dir = output_root / experiment_id
    output = experiment_dir / "timesfm_validation.json"
    if output.exists():
        return output
    validation_start = date.fromisoformat(config["validation_start"])
    test_start = date.fromisoformat(config["test_start"])
    validation_dates = [
        day for day in data.calendar.days if validation_start <= day < test_start
    ]
    if not validation_dates:
        raise ValueError("validation split contains no verified exchange dates")

    started = datetime.now(TZ)
    clock = time.monotonic()
    validation = {}
    checkpoints = {}
    adapter = None
    for instrument in ("TAIEX", "TX"):
        for representation in config["representations"]:
            for mode in TIMESFM_MODES:
                key = f"{instrument}/{representation}/{mode}"
                checkpoint = experiment_dir / f"checkpoint-{key.replace('/', '__')}.json"
                if checkpoint.exists():
                    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if (
                        envelope.get("key") != key
                        or envelope.get("dataset_id") != dataset_id
                        or envelope.get("config_hash") != config_hash
                        or envelope.get("code_version") != code_version
                    ):
                        raise ValueError(f"TimesFM checkpoint identity mismatch: {checkpoint}")
                    validation[key] = envelope["result"]
                    checkpoints[key] = {
                        "path": str(checkpoint),
                        "reused": True,
                        "elapsed_seconds": envelope["elapsed_seconds"],
                    }
                    continue
                if adapter is None:
                    adapter = TimesFM3Adapter(
                        load_pinned_forecaster(cache_dir, local_files_only=True)
                    )
                candidate_clock = time.monotonic()
                rows = walk_forward_timesfm(
                    data,
                    adapter,
                    instrument,
                    mode,
                    representation,
                    validation_start,
                    validation_dates[-1],
                    config,
                )
                candidate_result = {
                    "rows": rows,
                    "metrics": metrics(rows, config["neutral_bps"]),
                }
                elapsed = time.monotonic() - candidate_clock
                write_once(checkpoint, {
                    "key": key,
                    "dataset_id": dataset_id,
                    "config_hash": config_hash,
                    "code_version": code_version,
                    "elapsed_seconds": elapsed,
                    "result": candidate_result,
                })
                validation[key] = candidate_result
                checkpoints[key] = {
                    "path": str(checkpoint),
                    "reused": False,
                    "elapsed_seconds": elapsed,
                }
                consumed = sum(
                    item["elapsed_seconds"] for item in checkpoints.values()
                )
                if consumed > config["budget_cpu_seconds"]:
                    raise TimeoutError("frozen TimesFM validation budget exhausted")

    expected_dates = [str(day) for day in validation_dates]
    selected = {}
    selection_audit = {}
    minimum_coverage = float(config.get("min_validation_coverage", 0.95))
    for instrument in ("TAIEX", "TX"):
        selected[instrument], selection_audit[instrument] = _select_baseline(
            validation,
            instrument,
            expected_dates,
            minimum_coverage,
            config["neutral_bps"],
        )
    finished = datetime.now(TZ)
    result = {
        "experiment_type": "timesfm_validation_only_candidate_lock",
        "dataset_id": dataset_id,
        "data_quality": data_quality,
        "config_hash": config_hash,
        "code_version": code_version,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": time.monotonic() - clock,
        "candidate_elapsed_seconds": sum(
            item["elapsed_seconds"] for item in checkpoints.values()
        ),
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "validation_split": [str(validation_start), str(validation_dates[-1])],
        "selected_candidates": selected,
        "selection_audit": selection_audit,
        "validation": validation,
        "candidate_checkpoints": checkpoints,
        "final_test_opened": False,
        "historical_evidence": "retrospective_with_possible_pretraining_overlap",
    }
    write_once(output, result)
    return output


def run_timesfm_joint_validation(
    independent_validation_path: Path,
    dataset_dir: Path,
    config_path: Path,
    output_root: Path,
    cache_dir: Path,
) -> Path:
    """Compare one predeclared joint GBUD candidate with locked independent winners."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_validation_path.read_text(encoding="utf-8"))
    data, dataset_id, data_quality = _quality_gate(dataset_dir, config)
    config_hash = digest(config)
    if (
        independent.get("dataset_id") != dataset_id
        or independent.get("config_hash") != config_hash
        or independent.get("final_test_opened") is not False
    ):
        raise ValueError("independent TimesFM lock does not match dataset and frozen config")
    if "gbud" not in config["representations"]:
        raise ValueError("joint validation requires the predeclared GBUD representation")

    code_version = _git_state(config_path.parent.parent.parent)
    parent_hash = digest(independent)
    experiment_id = digest({
        "stage": "timesfm_joint_validation",
        "dataset_id": dataset_id,
        "config_hash": config_hash,
        "code_version": code_version,
        "independent_validation_lock_hash": parent_hash,
    })[:20]
    experiment_dir = output_root / experiment_id
    output = experiment_dir / "timesfm_joint_validation.json"
    if output.exists():
        return output

    validation_start = date.fromisoformat(config["validation_start"])
    test_start = date.fromisoformat(config["test_start"])
    validation_dates = [
        day for day in data.calendar.days if validation_start <= day < test_start
    ]
    if not validation_dates:
        raise ValueError("validation split contains no verified exchange dates")

    checkpoint = experiment_dir / "checkpoint-joint__gbud__timesfm_joint_multivariate.json"
    started = datetime.now(TZ)
    clock = time.monotonic()
    if checkpoint.exists():
        envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            envelope.get("dataset_id") != dataset_id
            or envelope.get("config_hash") != config_hash
            or envelope.get("code_version") != code_version
            or envelope.get("independent_validation_lock_hash") != parent_hash
        ):
            raise ValueError(f"joint TimesFM checkpoint identity mismatch: {checkpoint}")
        joint = envelope["result"]
        checkpoint_reused = True
        candidate_elapsed = envelope["elapsed_seconds"]
    else:
        adapter = TimesFM3Adapter(
            load_pinned_forecaster(cache_dir, local_files_only=True)
        )
        candidate_clock = time.monotonic()
        rows = walk_forward_timesfm_joint(
            data,
            adapter,
            "gbud",
            validation_start,
            validation_dates[-1],
            config,
        )
        joint = {
            instrument: {
                "rows": instrument_rows,
                "metrics": metrics(instrument_rows, config["neutral_bps"]),
            }
            for instrument, instrument_rows in rows.items()
        }
        candidate_elapsed = time.monotonic() - candidate_clock
        write_once(checkpoint, {
            "dataset_id": dataset_id,
            "config_hash": config_hash,
            "code_version": code_version,
            "independent_validation_lock_hash": parent_hash,
            "elapsed_seconds": candidate_elapsed,
            "result": joint,
        })
        checkpoint_reused = False
    if candidate_elapsed > config["budget_cpu_seconds"]:
        raise TimeoutError("frozen joint TimesFM validation budget exhausted")

    expected_dates = [str(day) for day in validation_dates]
    validation = {}
    for instrument in ("TAIEX", "TX"):
        independent_choice = independent["selected_candidates"][instrument]
        independent_key = (
            f"{instrument}/{independent_choice['representation']}/"
            f"{independent_choice['model']}"
        )
        validation[independent_key] = independent["validation"][independent_key]
        validation[f"{instrument}/gbud/{JOINT_MODE}"] = joint[instrument]

    selected = {}
    selection_audit = {}
    minimum_coverage = float(config.get("min_validation_coverage", 0.95))
    for instrument in ("TAIEX", "TX"):
        selected[instrument], selection_audit[instrument] = _select_baseline(
            validation,
            instrument,
            expected_dates,
            minimum_coverage,
            config["neutral_bps"],
        )
    result = {
        "experiment_type": "timesfm_joint_validation_only_candidate_lock",
        "dataset_id": dataset_id,
        "data_quality": data_quality,
        "config_hash": config_hash,
        "code_version": code_version,
        "independent_validation_path": str(independent_validation_path),
        "independent_validation_lock_hash": parent_hash,
        "started_at": started,
        "finished_at": datetime.now(TZ),
        "elapsed_seconds": time.monotonic() - clock,
        "candidate_elapsed_seconds": candidate_elapsed,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "validation_split": [str(validation_start), str(validation_dates[-1])],
        "selected_candidates": selected,
        "selection_audit": selection_audit,
        "validation": validation,
        "joint_checkpoint": {
            "path": str(checkpoint),
            "reused": checkpoint_reused,
            "elapsed_seconds": candidate_elapsed,
        },
        "final_test_opened": False,
        "search_stopped_after_joint_comparison": True,
        "historical_evidence": "retrospective_with_possible_pretraining_overlap",
    }
    write_once(output, result)
    return output


def _paired_rows(candidate_rows: list[dict], baseline_rows: list[dict]):
    candidate = {row["date"]: row for row in candidate_rows if row["status"] == "OK"}
    baseline = {row["date"]: row for row in baseline_rows if row["status"] == "OK"}
    dates = sorted(candidate.keys() & baseline.keys())
    return [candidate[day] for day in dates], [baseline[day] for day in dates]


def _acceptance(candidate_rows, baseline_rows, config):
    candidate_paired, baseline_paired = _paired_rows(candidate_rows, baseline_rows)
    candidate_metrics = metrics(candidate_paired, config["neutral_bps"])
    baseline_metrics = metrics(baseline_paired, config["neutral_bps"])
    expected_days = len(candidate_rows)
    paired_coverage = len(candidate_paired) / expected_days if expected_days else 0.0
    if not candidate_paired:
        return {
            "status": "FAIL_NO_PAIRED_DAYS",
            "paired_days": 0,
            "paired_coverage": 0.0,
        }
    close_improvement = 1 - (
        candidate_metrics["ohlc"]["close"]["mae_points"]
        / baseline_metrics["ohlc"]["close"]["mae_points"]
    )
    degradation = {
        field: (
            candidate_metrics["ohlc"][field]["mae_points"]
            / baseline_metrics["ohlc"][field]["mae_points"]
            - 1
        )
        for field in ("open", "high", "low")
    }
    interval_coverage = {
        field: candidate_metrics.get("intervals", {}).get(field, {}).get("coverage")
        for field in ("open", "high", "low", "close")
    }
    interval_pass = all(
        value is not None and 0.75 <= value <= 0.85
        for value in interval_coverage.values()
    )
    passed = (
        paired_coverage >= float(config.get("min_validation_coverage", 0.95))
        and close_improvement >= 0.05
        and all(value <= 0.05 for value in degradation.values())
        and interval_pass
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "paired_days": len(candidate_paired),
        "paired_coverage": paired_coverage,
        "close_relative_improvement": close_improvement,
        "ohl_degradation": degradation,
        "interval_coverage": interval_coverage,
        "interval_coverage_pass": interval_pass,
        "candidate_paired_metrics": candidate_metrics,
        "baseline_paired_metrics": baseline_metrics,
        "block_improvement": block_improvement(
            candidate_paired,
            baseline_paired,
            config["bootstrap_block"],
            config["bootstrap_draws"],
            config["seed"],
        ),
    }


def run_timesfm_final(
    validation_path: Path,
    baseline_path: Path,
    dataset_dir: Path,
    config_path: Path,
    cache_dir: Path,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation_lock = json.loads(validation_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    data, dataset_id, data_quality = _quality_gate(dataset_dir, config)
    config_hash = digest(config)
    if (
        validation_lock.get("dataset_id") != dataset_id
        or baseline.get("dataset_id") != dataset_id
        or validation_lock.get("config_hash") != config_hash
        or baseline.get("config_hash") != config_hash
        or validation_lock.get("final_test_opened") is not False
    ):
        raise ValueError("validation/baseline lock does not match dataset and frozen config")
    output = validation_path.parent / "timesfm_final.json"
    if output.exists():
        return output

    started = datetime.now(TZ)
    clock = time.monotonic()
    adapter = TimesFM3Adapter(load_pinned_forecaster(cache_dir, local_files_only=True))
    test_start = date.fromisoformat(config["test_start"])
    test_end = date.fromisoformat(config["test_end"])
    final_test = {}
    acceptance = {}
    joint_rows = None
    if any(
        choice["model"] == JOINT_MODE
        for choice in validation_lock["selected_candidates"].values()
    ):
        joint_rows = walk_forward_timesfm_joint(
            data, adapter, "gbud", test_start, test_end, config
        )
    for instrument in ("TAIEX", "TX"):
        choice = validation_lock["selected_candidates"][instrument]
        if choice["model"] == JOINT_MODE:
            if choice["representation"] != "gbud" or joint_rows is None:
                raise ValueError("joint final candidate must use the locked GBUD path")
            rows = joint_rows[instrument]
        else:
            rows = walk_forward_timesfm(
                data,
                adapter,
                instrument,
                choice["model"],
                choice["representation"],
                test_start,
                test_end,
                config,
            )
        final_test[instrument] = {
            "selection": choice,
            "rows": rows,
            "metrics": metrics(rows, config["neutral_bps"]),
        }
        baseline_rows = baseline["final_test"][instrument]["rows"]
        acceptance[instrument] = _acceptance(rows, baseline_rows, config)
        if time.monotonic() - clock > config["budget_cpu_seconds"]:
            raise TimeoutError("frozen TimesFM final-test budget exhausted")
    overall = "PASS" if all(
        result["status"] == "PASS" for result in acceptance.values()
    ) else "FAIL"
    result = {
        "experiment_type": "locked_timesfm_candidate_single_final_test",
        "dataset_id": dataset_id,
        "data_quality": data_quality,
        "config_hash": config_hash,
        "validation_lock_hash": digest(validation_lock),
        "baseline_lock_hash": digest(baseline),
        "code_version": _git_state(config_path.parent.parent.parent),
        "started_at": started,
        "finished_at": datetime.now(TZ),
        "elapsed_seconds": time.monotonic() - clock,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "final_test_split": [str(test_start), str(test_end)],
        "final_test": final_test,
        "acceptance_gate": {
            "status": overall,
            "requirements": {
                "close_relative_improvement_minimum": 0.05,
                "open_high_low_degradation_maximum": 0.05,
                "marginal_interval_coverage_range": [0.75, 0.85],
            },
            "instruments": acceptance,
        },
        "historical_evidence": "retrospective_with_possible_pretraining_overlap",
        "final_test_must_not_be_reused_for_tuning": True,
    }
    write_once(output, result)
    return output
