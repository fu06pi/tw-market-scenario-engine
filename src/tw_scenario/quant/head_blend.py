"""Causal blend of a frozen TimesFM head with its locked stable baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .backtest import load_dataset
from .contracts import OHLC, at, digest
from .head_calibration import calibrate_head_intervals
from .timesfm_head_training import development_acceptance, paired_validation_metrics

BLEND_VERSION = "causal-head-baseline-blend-v1"


def _validate_blend_config(config: dict) -> None:
    required = {
        "version",
        "candidate_context",
        "window",
        "minimum",
        "alpha_grid",
        "objective",
        "update_policy",
    }
    if set(config) != required:
        raise ValueError("unexpected causal blend config keys")
    if config["version"] != BLEND_VERSION:
        raise ValueError("unsupported causal blend version")
    window = int(config["window"])
    minimum = int(config["minimum"])
    if not 2 <= minimum <= window:
        raise ValueError("causal blend requires 2 <= minimum <= window")
    if config["objective"] != "prior_published_close_mae":
        raise ValueError("unsupported causal blend objective")
    if config["update_policy"] != "previous_labels_available_by_0830_only":
        raise ValueError("unsupported causal blend update policy")
    grid = [float(value) for value in config["alpha_grid"]]
    if len(grid) < 2 or grid != sorted(set(grid)) or grid[0] < 0 or grid[-1] > 1:
        raise ValueError("alpha grid must be unique, ordered, and within [0, 1]")


def causal_blend_rows(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    label_available_at: dict[str, datetime],
    config: dict,
) -> list[dict]:
    """Choose a convex weight from prior published close errors only."""
    _validate_blend_config(config)
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("candidate and baseline row counts differ")
    dates = [row["date"] for row in candidate_rows]
    if dates != sorted(set(dates)) or dates != [row["date"] for row in baseline_rows]:
        raise ValueError("blend rows must have identical unique chronological dates")
    window = int(config["window"])
    minimum = int(config["minimum"])
    grid = np.asarray(config["alpha_grid"], dtype=float)
    history: list[dict] = []
    output = []
    for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
        day = date.fromisoformat(candidate["date"])
        if candidate.get("status") != baseline.get("status"):
            raise ValueError("candidate and baseline statuses differ")
        if candidate.get("status") != "OK":
            failed = deepcopy(candidate)
            failed["blend"] = {"status": "INPUT_FAILED", "alpha": None, "samples": 0}
            output.append(failed)
            continue
        if (
            candidate.get("instrument") != baseline.get("instrument")
            or candidate.get("contract") != baseline.get("contract")
            or candidate.get("previous_close") != baseline.get("previous_close")
            or candidate.get("actual") != baseline.get("actual")
        ):
            raise ValueError("candidate and baseline truth/reference differ")
        cutoff = at(day)
        eligible = [
            record
            for record in history
            if record["day"] < day and record["available_at"] <= cutoff
        ][-window:]
        if len(eligible) < minimum:
            alpha = 0.0
            status = "WARMUP_BASELINE"
        else:
            losses = []
            for value in grid:
                errors = [
                    abs(
                        record["actual"]
                        - (
                            record["baseline"]
                            + value * (record["candidate"] - record["baseline"])
                        )
                    )
                    for record in eligible
                ]
                losses.append(float(np.mean(errors)))
            alpha = float(grid[int(np.argmin(losses))])
            status = "CAUSAL_GRID_SELECTED"
        result = deepcopy(candidate)
        for field in OHLC:
            result["predicted"][field] = float(
                baseline["predicted"][field]
                + alpha * (candidate["predicted"][field] - baseline["predicted"][field])
            )
            result["interval"][field] = [
                float(
                    baseline["interval"][field][index]
                    + alpha
                    * (
                        candidate["interval"][field][index]
                        - baseline["interval"][field][index]
                    )
                )
                for index in (0, 1)
            ]
        result["interval_method"] = (
            "causal convex blend of frozen head and locked baseline intervals"
        )
        result["blend"] = {
            "status": status,
            "alpha": alpha,
            "samples": len(eligible),
            "window": window,
            "objective": config["objective"],
        }
        output.append(result)
        available = label_available_at[candidate["date"]]
        if available.utcoffset() is None or available <= cutoff:
            raise ValueError("daytime actual must become available after forecast cutoff")
        history.append(
            {
                "day": day,
                "available_at": available,
                "candidate": float(candidate["predicted"]["close"]),
                "baseline": float(baseline["predicted"]["close"]),
                "actual": float(candidate["actual"]["close"]),
            }
        )
    return output


def evaluate_saved_head_blend(
    dataset_dir: Path,
    experiment_dir: Path,
    training_config: dict,
    blend_config: dict,
    calibration_config: dict,
) -> tuple[dict, list[dict]]:
    """Verify frozen folds, blend causally, calibrate, and rerun the full gate."""
    experiment_dir = Path(experiment_dir)
    result_path = experiment_dir / "result.json"
    source = json.loads(result_path.read_text())
    identity = source["identity"]
    data, dataset_id = load_dataset(Path(dataset_dir))
    _validate_blend_config(blend_config)
    if (
        identity["dataset_id"] != dataset_id
        or identity["config_hash"] != digest(training_config)
        or identity["spec"]["context"] != blend_config["candidate_context"]
        or calibration_config["candidate_context"] != blend_config["candidate_context"]
    ):
        raise ValueError("dataset/config/context differs from frozen experiment")
    for field in ("timesfm_code_commit", "timesfm_weight_revision"):
        if identity[field] != training_config[field]:
            raise ValueError("pinned TimesFM revision mismatch")
    market = identity["instrument"]
    if market not in ("TAIEX", "TX"):
        raise ValueError("independent market experiment required")

    candidate: list[dict] = []
    baseline: list[dict] = []
    fold_ranges: list[tuple[int, int]] = []
    fold_hashes: dict[str, str] = {}
    baseline_models = set()
    for index, fold in enumerate(source["fold_results"], 1):
        path = experiment_dir / f"fold-{index}.json"
        saved = json.loads(path.read_text())
        if saved["identity_hash"] != digest(identity) or saved["fold"] != index:
            raise ValueError("fold identity mismatch")
        if saved["summary"] != fold:
            raise ValueError("fold summary mismatch")
        checkpoint = Path(fold["checkpoint"]["path"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != fold["checkpoint"]["sha256"]:
            raise ValueError("checkpoint hash mismatch")
        expected = [
            str(day)
            for day in data.calendar.days
            if fold["validation_split"][0] <= str(day) <= fold["validation_split"][1]
        ]
        for key in ("candidate_rows", "baseline_rows"):
            if [row["date"] for row in saved[key]] != expected:
                raise ValueError("fold dates missing, duplicated or unordered")
        start = len(candidate)
        candidate.extend(saved["candidate_rows"])
        baseline.extend(saved["baseline_rows"])
        fold_ranges.append((start, len(candidate)))
        fold_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        baseline_models.add(fold["selected_baseline"]["model"])
    if len(baseline_models) != 1:
        raise ValueError("causal blend requires one locked baseline model across folds")

    times: dict[str, datetime] = {}
    for row in candidate + baseline:
        if row["status"] != "OK":
            continue
        day = date.fromisoformat(row["date"])
        actual = data.actual(market, day)
        if (
            row["instrument"] != market
            or row["contract"] != data.contract(market, day)
            or row["previous_close"] != data.reference(market, day).close
            or row["actual"] != {field: getattr(actual, field) for field in OHLC}
        ):
            raise ValueError("saved prediction label/reference differs from certified data")
        times[row["date"]] = actual.available_at

    raw_source_gate = development_acceptance(
        candidate,
        baseline,
        training_config,
        fold_improvements=[fold["close_relative_improvement"] for fold in source["fold_results"]],
    )
    if raw_source_gate != source["development_gate"]:
        raise ValueError("saved development gate not reproducible")
    blended = causal_blend_rows(candidate, baseline, times, blend_config)
    fold_improvements = [
        paired_validation_metrics(
            blended[start:end], baseline[start:end], training_config["neutral_bps"]
        )["close_relative_improvement"]
        for start, end in fold_ranges
    ]
    blended_raw_gate = development_acceptance(
        blended, baseline, training_config, fold_improvements=fold_improvements
    )
    calibrated = calibrate_head_intervals(
        blended,
        times,
        window=int(calibration_config["window"]),
        minimum=int(calibration_config["minimum"]),
    )
    gate = development_acceptance(
        calibrated, baseline, training_config, fold_improvements=fold_improvements
    )
    alphas = [row["blend"]["alpha"] for row in blended if row["status"] == "OK"]
    source_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    report = {
        "schema_version": "timesfm3-causal-head-baseline-blend-evaluation-v1",
        "status": "DEVELOPMENT_GATE_PASS" if gate["status"] == "PASS" else "DEVELOPMENT_GATE_FAIL",
        "instrument": market,
        "research_scope": "development_only_not_independent_holdout",
        "production_switch": False,
        "source_result": str(result_path),
        "source_result_sha256": source_sha,
        "source_fold_sha256": fold_hashes,
        "source_identity": identity,
        "baseline_model": next(iter(baseline_models)),
        "blend_config": deepcopy(blend_config),
        "calibration_config": deepcopy(calibration_config),
        "counts": {
            "rows": len(calibrated),
            "blend_status": dict(Counter(row["blend"]["status"] for row in blended)),
            "calibration_status": dict(
                Counter(row["calibration"]["status"] for row in calibrated)
            ),
        },
        "alpha": {
            "minimum": min(alphas),
            "maximum": max(alphas),
            "mean": float(np.mean(alphas)),
            "last": alphas[-1],
        },
        "fold_improvements": fold_improvements,
        "raw_source_gate": raw_source_gate,
        "blended_raw_gate": blended_raw_gate,
        "development_gate": gate,
        "limitations": [
            "Blend and calibration hyperparameters were selected on development data.",
            "A development PASS requires a newly frozen future holdout before any final claim.",
            "Marginal interval coverage does not establish joint OHLC path coverage.",
        ],
    }
    return report, calibrated
