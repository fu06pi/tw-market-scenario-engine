"""Audit GBUD range-loss geometry from allowed development evidence only.

The audit reconstructs each saved fold's optimization-training and early-stop
samples from the immutable dataset and verifies them against the persisted
sample audit.  Saved validation rows are read only to measure the already-known
range-underprediction symptom; they never choose the proposed candidate weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from statistics import mean, median

import numpy as np

from tw_scenario.quant.backtest import load_dataset
from tw_scenario.quant.contracts import EPS, Prices, digest, encode
from tw_scenario.quant.timesfm_head_training import (
    HeadTrainingSpec,
    audited_head_samples,
    output_head_loss,
    training_loss_scales,
)

TARGET_NAMES = ("g", "b", "log_u", "log_d")
QUANTILES = np.arange(0.1, 1.0, 0.1, dtype=np.float64)
SUMMARY_QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
CANDIDATE_WICK_WEIGHT = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("summary requires a finite non-empty vector")
    quantiles = np.quantile(values, SUMMARY_QUANTILES)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(SUMMARY_QUANTILES, quantiles, strict=True)
        },
    }


def _extensions(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64)
    u = np.maximum(0.0, np.exp(targets[:, 2]) - EPS)
    d = np.maximum(0.0, np.exp(targets[:, 3]) - EPS)
    return u, d


def _log_ranges(targets: np.ndarray) -> np.ndarray:
    u, d = _extensions(targets)
    return np.abs(targets[:, 1]) + u + d


def _sample_distribution(samples) -> dict:
    targets = np.stack([sample.target for sample in samples]).astype(np.float64)
    if targets.shape[1] != 4:
        raise ValueError("range audit currently requires one four-target market")
    u, d = _extensions(targets)
    previous = np.asarray([sample.previous_close for sample in samples], dtype=np.float64)
    point_ranges = np.asarray(
        [sample.actual["high"] - sample.actual["low"] for sample in samples],
        dtype=np.float64,
    )
    encoded_ranges = _log_ranges(targets)
    direct_ranges = np.asarray(
        [np.log(sample.actual["high"] / sample.actual["low"]) for sample in samples],
        dtype=np.float64,
    )
    if not np.allclose(encoded_ranges, direct_ranges, atol=2e-7, rtol=2e-7):
        raise ValueError("GBUD-derived and direct log ranges differ")
    target_summary = {
        name: _summary(targets[:, index]) for index, name in enumerate(TARGET_NAMES)
    }
    positive = {}
    for name, values in (("u", u), ("d", d)):
        is_zero = values <= EPS * 1e-3
        positive_values = values[~is_zero]
        positive[name] = {
            "exact_zero_count": int(is_zero.sum()),
            "exact_zero_fraction": float(is_zero.mean()),
            "positive_extension_bps": (
                _summary(positive_values * 10_000) if len(positive_values) else None
            ),
            "all_extension_bps": _summary(values * 10_000),
        }
    return {
        "samples": len(samples),
        "start": str(samples[0].day),
        "end": str(samples[-1].day),
        "targets": target_summary,
        "extensions": positive,
        "ohlc_range": {
            "log_range_bps": _summary(encoded_ranges * 10_000),
            "points": _summary(point_ranges),
            "bps_of_previous_close": _summary(point_ranges / previous * 10_000),
            "body_share_of_log_range": _summary(
                np.divide(
                    np.abs(targets[:, 1]),
                    encoded_ranges,
                    out=np.zeros_like(encoded_ranges),
                    where=encoded_ranges > 0,
                )
            ),
        },
    }


def _constant_quantile_logits(targets: np.ndarray, rows: int) -> np.ndarray:
    values = np.quantile(targets, QUANTILES, axis=0).T
    return np.repeat(values[None, :, :], rows, axis=0).astype(np.float32)


def _loss_breakdown(logits, target, scales) -> dict:
    import torch

    units = torch.as_tensor(scales["target"], dtype=logits.dtype)
    close_units = torch.as_tensor(scales["close"], dtype=logits.dtype)
    q = torch.as_tensor(QUANTILES, dtype=logits.dtype)
    median_index = int(torch.argmin(torch.abs(q - 0.5)))
    point = logits[:, :, median_index]
    point_terms = torch.nn.functional.smooth_l1_loss(
        point / units, target / units, reduction="none"
    ).mean(dim=0)
    errors = target[:, :, None] / units[None, :, None] - logits / units[None, :, None]
    pinball_terms = torch.maximum(q * errors, (q - 1) * errors).mean(dim=(0, 2))
    predicted_close = point[:, 0] + point[:, 1]
    actual_close = target[:, 0] + target[:, 1]
    close_loss = torch.nn.functional.smooth_l1_loss(
        predicted_close / close_units[0], actual_close / close_units[0]
    )
    predicted_u = torch.clamp(torch.exp(point[:, 2]) - EPS, min=0.0)
    predicted_d = torch.clamp(torch.exp(point[:, 3]) - EPS, min=0.0)
    actual_u = torch.clamp(torch.exp(target[:, 2]) - EPS, min=0.0)
    actual_d = torch.clamp(torch.exp(target[:, 3]) - EPS, min=0.0)
    predicted_wick = torch.log(torch.clamp(predicted_u + predicted_d, min=EPS))
    actual_wick = torch.log(torch.clamp(actual_u + actual_d, min=EPS))
    wick_unit = torch.as_tensor(scales["wick"][0], dtype=logits.dtype)
    wick_loss = torch.nn.functional.smooth_l1_loss(
        predicted_wick / wick_unit, actual_wick / wick_unit
    )
    return {
        "point_smooth_l1_by_target": {
            name: float(value) for name, value in zip(TARGET_NAMES, point_terms, strict=True)
        },
        "pinball_by_target": {
            name: float(value) for name, value in zip(TARGET_NAMES, pinball_terms, strict=True)
        },
        "close_smooth_l1": float(close_loss),
        "candidate_wick_smooth_l1": float(wick_loss),
        "constant_estimator_log_wick_span": {
            "predicted_mean": float(predicted_wick.mean()),
            "actual_mean": float(actual_wick.mean()),
            "mean_bias": float((predicted_wick - actual_wick).mean()),
            "mae": float(torch.abs(predicted_wick - actual_wick).mean()),
            "underprediction_fraction": float((predicted_wick < actual_wick).float().mean()),
        },
    }


def _gradient_summary(gradient) -> dict:
    import torch

    absolute = torch.abs(gradient)
    per_target = absolute.sum(dim=(0, 2))
    total = per_target.sum()
    median_index = 4
    return {
        "absolute_output_gradient_l1_by_target": {
            name: float(value) for name, value in zip(TARGET_NAMES, per_target, strict=True)
        },
        "absolute_output_gradient_share_by_target": {
            name: float(value / total)
            for name, value in zip(TARGET_NAMES, per_target, strict=True)
        },
        "median_channel_mean_absolute_gradient_by_target": {
            name: float(value)
            for name, value in zip(
                TARGET_NAMES, absolute[:, :, median_index].mean(dim=0), strict=True
            )
        },
    }


def _loss_and_gradient_audit(train_samples, evaluation_samples, saved_scales) -> dict:
    import torch

    train_targets = np.stack([sample.target for sample in train_samples]).astype(np.float32)
    evaluation_targets = np.stack(
        [sample.target for sample in evaluation_samples]
    ).astype(np.float32)
    recomputed_scales = training_loss_scales(train_samples)
    for key in ("target", "close"):
        if not np.allclose(recomputed_scales[key], saved_scales[key], atol=2e-7, rtol=2e-7):
            raise ValueError(f"saved {key} loss scale differs from reconstructed training data")
    train_u, train_d = _extensions(train_targets)
    recomputed_scales["wick"] = [
        max(
            float(np.log(np.maximum(train_u + train_d, EPS)).std()),
            recomputed_scales["minimum_scale"],
        )
    ]
    logits_np = _constant_quantile_logits(train_targets, len(evaluation_targets))
    target = torch.as_tensor(evaluation_targets)
    base_spec = HeadTrainingSpec(
        context=32,
        learning_rate=1e-4,
        weight_decay=0.0,
        loss_normalization="train_std",
    )
    gradients = {}
    losses = {}
    for label, wick_weight in (
        ("current_wick_weight_0", 0.0),
        ("candidate_wick_weight_0_25", CANDIDATE_WICK_WEIGHT),
    ):
        logits = torch.as_tensor(logits_np).requires_grad_(True)
        spec = replace(base_spec, wick_weight=wick_weight)
        loss = output_head_loss(logits, target, QUANTILES, spec, recomputed_scales)
        loss.backward()
        losses[label] = float(loss.detach())
        gradients[label] = _gradient_summary(logits.grad)
    return {
        "estimator": (
            "training-only marginal quantiles held constant; diagnostics are output-space "
            "loss geometry, not saved-model parameter gradients"
        ),
        "loss_scales": recomputed_scales,
        "loss": losses,
        "components": _loss_breakdown(
            torch.as_tensor(logits_np), target, recomputed_scales
        ),
        "output_gradients": gradients,
    }


def _artifact_range_diagnostic(rows: list[dict]) -> dict:
    target_errors = {name: [] for name in TARGET_NAMES}
    range_errors, range_abs_errors, ratios = [], [], []
    extension_errors = {"u": [], "d": []}
    actual_zero = {"u": 0, "d": 0}
    predicted_zero = {"u": 0, "d": 0}
    for row in rows:
        if row["status"] != "OK":
            raise ValueError("saved fold contains a failed candidate row")
        actual = Prices(**row["actual"])
        predicted = Prices(**row["predicted"])
        previous = float(row["previous_close"])
        actual_target = encode(actual, previous, "gbud")
        predicted_target = encode(predicted, previous, "gbud")
        for index, name in enumerate(TARGET_NAMES):
            target_errors[name].append(float(predicted_target[index] - actual_target[index]))
        actual_u, actual_d = _extensions(actual_target[None, :])
        predicted_u, predicted_d = _extensions(predicted_target[None, :])
        for name, observed, forecast in (
            ("u", actual_u[0], predicted_u[0]),
            ("d", actual_d[0], predicted_d[0]),
        ):
            extension_errors[name].append(float((forecast - observed) * 10_000))
            actual_zero[name] += int(observed <= EPS * 1e-3)
            predicted_zero[name] += int(forecast <= EPS * 1e-3)
        actual_range = actual.high - actual.low
        predicted_range = predicted.high - predicted.low
        error = predicted_range - actual_range
        range_errors.append(error)
        range_abs_errors.append(abs(error))
        ratios.append(predicted_range / actual_range if actual_range > 0 else np.nan)
    valid_ratios = np.asarray([value for value in ratios if np.isfinite(value)])
    return {
        "days": len(rows),
        "target_error": {
            name: {
                "bias": mean(values),
                "mae": mean(map(abs, values)),
            }
            for name, values in target_errors.items()
        },
        "extension_error_bps": {
            name: {
                "bias": mean(values),
                "mae": mean(map(abs, values)),
                "actual_zero_fraction": actual_zero[name] / len(rows),
                "predicted_zero_fraction": predicted_zero[name] / len(rows),
            }
            for name, values in extension_errors.items()
        },
        "ohlc_range_points": {
            "bias": mean(range_errors),
            "mae": mean(range_abs_errors),
            "median_error": median(range_errors),
            "underprediction_fraction": mean(value < 0 for value in range_errors),
            "predicted_to_actual_ratio": _summary(valid_ratios),
        },
    }


def _resolve_repo_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def audit(dataset_dir: Path, config_path: Path, experiment_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_path = experiment_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    identity = result["identity"]
    if identity["instrument"] != "TAIEX" or identity["config_hash"] != digest(config):
        raise ValueError("audit inputs do not match the saved TAIEX experiment identity")
    if identity["spec"].get("loss_normalization") != "train_std":
        raise ValueError("range audit requires the saved train_std experiment")
    data, dataset_id = load_dataset(dataset_dir)
    if dataset_id != identity["dataset_id"]:
        raise ValueError("dataset identity mismatch")
    repo_root = config_path.resolve().parents[2]
    output = {
        "schema_version": "timesfm3-output-head-range-loss-audit-v1",
        "scope": {
            "labels": "optimization-training and early-stop samples only",
            "saved_validation_artifacts": (
                "read-only symptom measurement; not used to choose candidate or claim acceptance"
            ),
            "final_test_opened": False,
            "current_day_actual_used": False,
            "new_training_run": False,
            "candidate_selection_rule": (
                "wick_weight=0.25 equals one of four equally averaged point-target channels; "
                "chosen from loss structure before reading saved validation errors"
            ),
        },
        "identity": {
            "dataset_id": dataset_id,
            "experiment": str(experiment_dir),
            "config": str(config_path),
            "config_hash": digest(config),
            "instrument": identity["instrument"],
            "context": identity["spec"]["context"],
            "loss_normalization": identity["spec"]["loss_normalization"],
        },
        "source_hashes": {
            str(config_path): _sha256(config_path),
            str(result_path): _sha256(result_path),
            str(dataset_dir / "manifest.json"): _sha256(dataset_dir / "manifest.json"),
        },
        "folds": [],
        "candidate": {
            "single_model_variable": "wick_weight",
            "current": 0.0,
            "proposed": CANDIDATE_WICK_WEIGHT,
            "term": (
                "SmoothL1(log(predicted u+d) / training_wick_std, "
                "log(actual u+d) / training_wick_std)"
            ),
            "training_scale_source": "each fold's optimization-training labels only",
            "all_other_training_and_gate_settings": "unchanged",
            "status": "IMPLEMENTED_AND_UNIT_TESTED_NOT_TRAINED_NOT_SELECTED",
        },
    }
    context = int(identity["spec"]["context"])
    stride = int(config["training_stride"])
    early_stop_sessions = int(config["early_stop_sessions"])
    all_days = data.calendar.days
    for fold_index, (fold_identity, fold_summary) in enumerate(
        zip(identity["folds"], result["fold_results"], strict=True), start=1
    ):
        train_start, train_end = map(date.fromisoformat, fold_identity["train"])
        train_days = [day for day in all_days if train_start <= day <= train_end]
        usable_train_days = train_days[context:]
        training_days = usable_train_days[:-early_stop_sessions:stride]
        early_stop_days = usable_train_days[-early_stop_sessions:]
        training_samples, training_audit = audited_head_samples(
            data, "TAIEX", training_days, context
        )
        early_stop_samples, early_stop_audit = audited_head_samples(
            data, "TAIEX", early_stop_days, context
        )
        persisted_audit_path = _resolve_repo_path(
            fold_summary["sample_audit"]["path"], repo_root
        )
        persisted_audit = json.loads(persisted_audit_path.read_text(encoding="utf-8"))
        for split, reconstructed in (
            ("training", training_audit),
            ("early_stop", early_stop_audit),
        ):
            if persisted_audit[split] != reconstructed:
                raise ValueError(f"fold {fold_index} {split} sample audit mismatch")
        fold_path = experiment_dir / f"fold-{fold_index}.json"
        fold_payload = json.loads(fold_path.read_text(encoding="utf-8"))
        saved_scales = fold_summary["training"]["loss_scales"]
        train_distribution = _sample_distribution(training_samples)
        early_distribution = _sample_distribution(early_stop_samples)
        fold_output = {
            "fold": fold_index,
            "training": train_distribution,
            "early_stop": early_distribution,
            "early_stop_vs_training": {
                "mean_log_range_ratio": (
                    early_distribution["ohlc_range"]["log_range_bps"]["mean"]
                    / train_distribution["ohlc_range"]["log_range_bps"]["mean"]
                ),
                "mean_point_range_ratio": (
                    early_distribution["ohlc_range"]["points"]["mean"]
                    / train_distribution["ohlc_range"]["points"]["mean"]
                ),
                "p90_log_range_ratio": (
                    early_distribution["ohlc_range"]["log_range_bps"]["quantiles"]["q90"]
                    / train_distribution["ohlc_range"]["log_range_bps"]["quantiles"]["q90"]
                ),
            },
            "training_loss_geometry": _loss_and_gradient_audit(
                training_samples, training_samples, saved_scales
            ),
            "early_stop_loss_geometry": _loss_and_gradient_audit(
                training_samples, early_stop_samples, saved_scales
            ),
            "saved_validation_symptom": _artifact_range_diagnostic(
                fold_payload["candidate_rows"]
            ),
        }
        output["folds"].append(fold_output)
        output["source_hashes"].update(
            {
                str(fold_path): _sha256(fold_path),
                str(persisted_audit_path): _sha256(persisted_audit_path),
            }
        )
    output["findings"] = {
        "transform_zero_point_mass": (
            "log_u/log_d encode exact zero extensions at log(1e-6)=-13.8155, "
            "creating a separated point mass rather than a continuous Gaussian-like target"
        ),
        "current_objective_gap": (
            "the current loss has a dedicated close-return term but no joint wick-span term; "
            "u and d are learned marginally in transformed units and receive negligible "
            "output-space gradient beside g/b"
        ),
        "candidate_interpretation": (
            "the proposed log(u+d) term couples upper/lower extensions without changing b, "
            "and uses only a fold's optimization-training labels for scale"
        ),
        "limitations": [
            "Output gradients use a training-only constant marginal-quantile estimator, not the saved head's parameter gradients.",
            "Saved validation rows only confirm the pre-existing symptom and do not estimate candidate performance.",
            "No full fold training, promotion decision, final-test reuse, or current-day outcome is included.",
        ],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset_dir, args.config_path, args.experiment_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
