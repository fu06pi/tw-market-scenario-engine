from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .backtest import _git_state, _select_baseline, load_dataset, walk_forward
from .contracts import OHLC, decode, digest, encode, write_once
from .evaluation import block_improvement, metrics
from .timesfm3 import (
    CODE_COMMIT,
    WEIGHT_REPO,
    WEIGHT_REVISION,
    load_pinned_forecaster,
    runtime_probe,
)


@dataclass(frozen=True)
class HeadSample:
    day: date
    history: np.ndarray
    past_features: np.ndarray
    forecast_features: np.ndarray
    target: np.ndarray
    previous_close: float
    actual: dict[str, float]
    contract: str


@dataclass(frozen=True)
class HeadTrainingSpec:
    context: int
    learning_rate: float
    weight_decay: float
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 1
    point_weight: float = 1.0
    close_weight: float = 2.0
    quantile_weight: float = 0.5
    crossing_weight: float = 0.1

    def validate(self, input_patch_len: int = 32) -> None:
        if self.context < input_patch_len or self.context % input_patch_len:
            raise ValueError("TimesFM head context must be a positive patch multiple")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("invalid training limits")


def _spec_from_config(config: dict, candidate_index: int) -> HeadTrainingSpec:
    grid = config.get("training_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("training_grid must contain at least one candidate")
    if not 0 <= candidate_index < len(grid):
        raise ValueError(
            f"candidate_index must be between 0 and {len(grid) - 1}"
        )
    candidate = grid[candidate_index]
    if not isinstance(candidate, dict):
        raise TypeError("training_grid candidates must be objects")
    return HeadTrainingSpec(
        context=int(candidate["context"]),
        learning_rate=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
        max_epochs=int(config["max_epochs"]),
        patience=int(config["patience"]),
        batch_size=int(config["batch_size"]),
        point_weight=float(config["point_weight"]),
        close_weight=float(config["close_weight"]),
        quantile_weight=float(config["quantile_weight"]),
        crossing_weight=float(config["crossing_weight"]),
    )


def expanding_folds(
    days: list[date], *, folds: int, validation_sessions: int, minimum_train: int
) -> list[tuple[list[date], list[date]]]:
    """Create deterministic expanding folds ending at the last supplied session."""
    if folds < 1 or validation_sessions < 1 or minimum_train < 1:
        raise ValueError("fold parameters must be positive")
    required = minimum_train + folds * validation_sessions
    if len(days) < required:
        raise ValueError(f"insufficient sessions for expanding folds: {len(days)} < {required}")
    first_validation = len(days) - folds * validation_sessions
    output = []
    for index in range(folds):
        start = first_validation + index * validation_sessions
        stop = start + validation_sessions
        output.append((days[:start], days[start:stop]))
    return output


def _feature_scaler(samples: list[HeadSample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("feature scaler requires training samples")
    values = np.concatenate(
        [sample.past_features for sample in samples]
        + [sample.forecast_features[None, :] for sample in samples],
        axis=0,
    )
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("non-finite train-only feature scaler")
    return mean.astype(np.float32), scale.astype(np.float32)


def build_head_samples(data, instrument: str, days: list[date], context: int) -> list[HeadSample]:
    """Materialize causal samples; failed dates remain explicit at the caller."""
    samples = []
    for day in days:
        history, features, _ = data.research_history(
            instrument, day, "gbud", context, context
        )
        previous_close = data.reference(instrument, day).close
        actual_bar = data.actual(instrument, day)
        target = encode(actual_bar, previous_close, "gbud").astype(np.float32)
        samples.append(HeadSample(
            day=day,
            history=history.astype(np.float32),
            past_features=features.astype(np.float32),
            forecast_features=data.research_features(instrument, day).astype(np.float32),
            target=target,
            previous_close=float(previous_close),
            actual={field: float(getattr(actual_bar, field)) for field in OHLC},
            contract=data.contract(instrument, day),
        ))
    return samples


def _batch_inputs(model, samples: list[HeadSample], mean: np.ndarray, scale: np.ndarray):
    import torch

    if not samples:
        raise ValueError("empty TimesFM training batch")
    context = samples[0].history.shape[0]
    if context % model.input_patch_len:
        raise ValueError("TimesFM training context must align to input patches")
    feature_count = samples[0].past_features.shape[1]
    if any(
        sample.history.shape != (context, 4)
        or sample.past_features.shape != (context, feature_count)
        or sample.forecast_features.shape != (feature_count,)
        for sample in samples
    ):
        raise ValueError("inconsistent TimesFM head sample shapes")

    histories = np.stack([sample.history.T for sample in samples])
    features = np.stack([((sample.past_features - mean) / scale).T for sample in samples])
    future = np.stack([(sample.forecast_features - mean) / scale for sample in samples])
    context_values = torch.tensor(
        np.concatenate([histories, features], axis=1), dtype=torch.float32
    )
    batch, variates, _ = context_values.shape
    horizon = model.output_patch_len
    horizon_values = torch.zeros((batch, variates, horizon), dtype=torch.float32)
    horizon_masks = torch.ones((batch, variates, horizon), dtype=torch.bool)
    horizon_values[:, 4:, 0] = torch.tensor(future, dtype=torch.float32)
    horizon_masks[:, 4:, 0] = False

    all_values = torch.cat([context_values, horizon_values], dim=-1)
    all_masks = torch.cat(
        [torch.zeros_like(context_values, dtype=torch.bool), horizon_masks], dim=-1
    )
    values = all_values.reshape(batch, variates, -1, model.input_patch_len)
    masks = all_masks.reshape(batch, variates, -1, model.input_patch_len)
    target_flags = torch.zeros((batch, variates, values.shape[2]), dtype=torch.bool)
    target_flags[:, :4, :] = True
    context_patches = context // model.input_patch_len
    cpm_mask = torch.zeros((batch, values.shape[2]), dtype=torch.bool)
    cpm_mask[:, context_patches:] = True
    return {
        "values": values,
        "masks": masks,
        "patch_is_target": target_flags,
    }, cpm_mask, context_patches


def _one_step_logits(model, inputs, cpm_mask, context_patches):
    output = model.forward(inputs, patch_cpm_mask=cpm_mask)
    return output["logits"][:, :4, context_patches - 1, 0, :]


def output_head_loss(logits, target, quantiles, spec: HeadTrainingSpec):
    """Robust point, close-return, quantile and crossing losses for GBUD."""
    import torch

    if logits.ndim != 3 or logits.shape[1] != 4:
        raise ValueError("TimesFM head logits must have shape (batch, 4, quantiles)")
    if target.shape != logits.shape[:2]:
        raise ValueError("TimesFM head target shape mismatch")
    q = torch.as_tensor(quantiles, dtype=logits.dtype, device=logits.device)
    if q.shape != (logits.shape[2],):
        raise ValueError("TimesFM quantile schema mismatch")
    median_index = int(torch.argmin(torch.abs(q - 0.5)))
    point = logits[:, :, median_index]
    point_loss = torch.nn.functional.smooth_l1_loss(point, target)
    close_loss = torch.nn.functional.smooth_l1_loss(
        point[:, 0] + point[:, 1], target[:, 0] + target[:, 1]
    )
    errors = target[:, :, None] - logits
    pinball = torch.maximum(q * errors, (q - 1) * errors).mean()
    crossing = torch.relu(logits[:, :, :-1] - logits[:, :, 1:]).mean()
    return (
        spec.point_weight * point_loss
        + spec.close_weight * close_loss
        + spec.quantile_weight * pinball
        + spec.crossing_weight * crossing
    )


def _evaluate_loss(model, samples, mean, scale, spec):
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), spec.batch_size):
            batch = samples[start : start + spec.batch_size]
            inputs, mask, patches = _batch_inputs(model, batch, mean, scale)
            target = torch.tensor(np.stack([item.target for item in batch]))
            logits = _one_step_logits(model, inputs, mask, patches)
            losses.append(float(output_head_loss(logits, target, model.quantiles, spec)))
    return float(np.mean(losses))


def train_output_head(
    model,
    train_samples: list[HeadSample],
    early_stop_samples: list[HeadSample],
    spec: HeadTrainingSpec,
    *,
    seed: int,
    deadline: float,
) -> dict:
    """Train only the pinned model output head and restore its best epoch."""
    import torch

    spec.validate(model.input_patch_len)
    if not train_samples or not early_stop_samples:
        raise ValueError("output-head training requires train and early-stop samples")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.output_head.parameters():
        parameter.requires_grad_(True)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(not name.startswith("output_head.") for name in trainable):
        raise RuntimeError("custom training must update only output_head parameters")

    mean, scale = _feature_scaler(train_samples)
    optimizer = torch.optim.AdamW(
        model.output_head.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, spec.max_epochs + 1):
        if time.monotonic() >= deadline:
            break
        order = rng.permutation(len(train_samples))
        # Keep the frozen backbone in deterministic inference mode; gradients
        # are still recorded for the unfrozen linear output head.
        model.eval()
        epoch_losses = []
        for offset in range(0, len(order), spec.batch_size):
            if time.monotonic() >= deadline:
                break
            batch = [train_samples[index] for index in order[offset : offset + spec.batch_size]]
            inputs, mask, patches = _batch_inputs(model, batch, mean, scale)
            target = torch.tensor(np.stack([item.target for item in batch]))
            optimizer.zero_grad(set_to_none=True)
            logits = _one_step_logits(model, inputs, mask, patches)
            loss = output_head_loss(logits, target, model.quantiles, spec)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite output-head training loss")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation_loss = _evaluate_loss(
            model, early_stop_samples, mean, scale, spec
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "early_stop_loss": validation_loss,
        })
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().contiguous().clone()
                for name, value in model.output_head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    if best_state is None:
        raise TimeoutError("training budget expired before a complete epoch")
    model.output_head.load_state_dict(best_state)
    model.eval()
    return {
        "best_epoch": best_epoch,
        "best_early_stop_loss": best_loss,
        "epochs_completed": len(history),
        "stopped_early": stale >= spec.patience,
        "budget_exhausted": time.monotonic() >= deadline,
        "history": history,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "trainable_parameter_names": trainable,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.output_head.parameters()
        ),
    }


def predict_head(model, sample: HeadSample, mean: np.ndarray, scale: np.ndarray):
    import torch

    model.eval()
    with torch.no_grad():
        inputs, mask, patches = _batch_inputs(model, [sample], mean, scale)
        logits = _one_step_logits(model, inputs, mask, patches)[0].cpu().numpy()
    ordered = np.sort(logits, axis=1)
    median = ordered[:, len(model.quantiles) // 2]
    point = decode(median, sample.previous_close, "gbud")
    coherent_paths = np.stack([
        decode(ordered[:, index], sample.previous_close, "gbud").array()
        for index in range(ordered.shape[1])
    ])
    lower = coherent_paths.min(axis=0)
    upper = coherent_paths.max(axis=0)
    interval = {
        field: [
            float(min(lower[index], getattr(point, field))),
            float(max(upper[index], getattr(point, field))),
        ]
        for index, field in enumerate(OHLC)
    }
    return point, interval, ordered


def development_acceptance(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    config: dict,
    *,
    fold_improvements: list[float],
) -> dict:
    """Stricter pre-forward gate; this never opens or relabels the old final."""
    candidate = {row["date"]: row for row in candidate_rows if row["status"] == "OK"}
    baseline = {row["date"]: row for row in baseline_rows if row["status"] == "OK"}
    paired_dates = sorted(candidate.keys() & baseline.keys())
    candidate_paired = [candidate[day] for day in paired_dates]
    baseline_paired = [baseline[day] for day in paired_dates]
    if not paired_dates:
        return {"status": "FAIL_NO_PAIRED_DAYS", "paired_days": 0}
    candidate_metrics = metrics(candidate_paired, config["neutral_bps"])
    baseline_metrics = metrics(baseline_paired, config["neutral_bps"])
    close_improvement = 1 - (
        candidate_metrics["ohlc"]["close"]["mae_points"]
        / baseline_metrics["ohlc"]["close"]["mae_points"]
    )
    ohl_degradation = {
        field: (
            candidate_metrics["ohlc"][field]["mae_points"]
            / baseline_metrics["ohlc"][field]["mae_points"]
            - 1
        )
        for field in ("open", "high", "low")
    }
    coverage = {
        field: candidate_metrics.get("intervals", {}).get(field, {}).get("coverage")
        for field in OHLC
    }
    interval_score_noninferior = all(
        field in candidate_metrics.get("intervals", {})
        and field in baseline_metrics.get("intervals", {})
        and candidate_metrics["intervals"][field]["interval_score"]
        <= baseline_metrics["intervals"][field]["interval_score"]
        for field in OHLC
    )
    uncertainty = block_improvement(
        candidate_paired,
        baseline_paired,
        config["bootstrap_block"],
        config["bootstrap_draws"],
        config["seed"],
    )
    ci = uncertainty.get("close_relative_improvement_ci95")
    paired_coverage = len(paired_dates) / max(len(candidate_rows), len(baseline_rows))
    requirements = {
        "close_buffer": close_improvement >= float(config["promotion_close_improvement"]),
        "fold_majority": sum(value > 0 for value in fold_improvements)
        > len(fold_improvements) / 2,
        "bootstrap_lower_nonnegative": ci is not None and ci[0] >= 0,
        "ohl_noninferiority": all(value <= 0.05 for value in ohl_degradation.values()),
        "interval_coverage": all(
            value is not None and 0.75 <= value <= 0.85 for value in coverage.values()
        ),
        "interval_score_noninferior": interval_score_noninferior,
        "forecast_coverage": paired_coverage >= float(config["promotion_min_coverage"]),
    }
    return {
        "status": "PASS" if all(requirements.values()) else "FAIL",
        "paired_days": len(paired_dates),
        "paired_coverage": paired_coverage,
        "close_relative_improvement": close_improvement,
        "ohl_degradation": ohl_degradation,
        "interval_coverage": coverage,
        "fold_improvements": fold_improvements,
        "block_improvement": uncertainty,
        "requirements": requirements,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
    }


def select_output_head_candidate(results: list[dict], max_no_gain_rounds: int) -> dict:
    """Apply the predeclared bounded-search stop rule to completed candidates."""
    if not results:
        raise ValueError("candidate selection requires at least one result")
    if max_no_gain_rounds < 1:
        raise ValueError("max_no_gain_rounds must be positive")
    identities = [result.get("identity", {}) for result in results]
    common = {
        (identity.get("dataset_id"), identity.get("instrument"), identity.get("config_hash"))
        for identity in identities
    }
    if len(common) != 1 or None in next(iter(common)):
        raise ValueError("output-head candidate identities do not match")
    indexes = [identity.get("candidate_index") for identity in identities]
    if any(not isinstance(index, int) for index in indexes) or len(set(indexes)) != len(indexes):
        raise ValueError("candidate indexes must be unique integers")

    best = None
    best_improvement = -math.inf
    no_gain = 0
    evaluated = []
    stopped_early = False
    for result in sorted(results, key=lambda item: item["identity"]["candidate_index"]):
        gate = result.get("development_gate")
        if not isinstance(gate, dict) or "close_relative_improvement" not in gate:
            raise ValueError("candidate result lacks a development gate")
        improvement = float(gate["close_relative_improvement"])
        entry = {
            "candidate_index": result["identity"]["candidate_index"],
            "status": result.get("status"),
            "development_gate": gate.get("status"),
            "close_relative_improvement": improvement,
            "result_hash": digest(result),
        }
        evaluated.append(entry)
        if improvement > best_improvement + 1e-12:
            best = entry
            best_improvement = improvement
            no_gain = 0
        else:
            no_gain += 1
            if no_gain >= max_no_gain_rounds:
                stopped_early = True
                break
    if best is None:
        raise RuntimeError("no output-head candidate could be selected")
    passed = best["development_gate"] == "PASS"
    dataset_id, instrument, config_hash = next(iter(common))
    return {
        "schema_version": "timesfm3-output-head-selection-v1",
        "status": (
            "LOCKED_AWAITING_NEW_FORWARD_HOLDOUT"
            if passed
            else "DEVELOPMENT_SEARCH_STOPPED_NO_PASS"
        ),
        "dataset_id": dataset_id,
        "instrument": instrument,
        "config_hash": config_hash,
        "selected": best,
        "evaluated": evaluated,
        "stopped_after_no_gain": stopped_early,
        "max_no_gain_rounds": max_no_gain_rounds,
        "final_test_opened": False,
        "old_final_reused_as_independent_test": False,
        "production_candidate": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_output_head_training(
    dataset_dir: Path,
    config_path: Path,
    cache_dir: Path,
    output_root: Path,
    instrument: str,
    candidate_index: int = 0,
) -> Path:
    """Train one market-specific TimesFM output head without opening a final test."""
    from safetensors.torch import save_file

    if instrument not in {"TAIEX", "TX"}:
        raise ValueError("instrument must be TAIEX or TX")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    probe = runtime_probe(cache_dir, int(config["memory_probe_limit_mb"]))
    if probe.status != "READY_TO_LOAD":
        raise RuntimeError(f"TimesFM 3 load blocked by preflight: {probe.status}")
    data, dataset_id = load_dataset(dataset_dir)
    development_end = date.fromisoformat(config["development_end"])
    early_stop_sessions = int(config["early_stop_sessions"])
    spec = _spec_from_config(config, candidate_index)
    context = spec.context
    eligible = [
        day for day in data.calendar.days
        if date.fromisoformat(config["train_start"]) <= day <= development_end
    ]
    if len(eligible) <= context + early_stop_sessions:
        raise ValueError("insufficient development history for output-head training")
    sample_days = eligible[context:]
    training_days = sample_days[:-early_stop_sessions]
    early_stop_days = sample_days[-early_stop_sessions:]
    stride = int(config["training_stride"])
    if stride < 1:
        raise ValueError("training_stride must be positive")
    train_samples = build_head_samples(
        data, instrument, training_days[::stride], context
    )
    early_stop_samples = build_head_samples(
        data, instrument, early_stop_days, context
    )
    identity = {
        "name": "custom_timesfm3_output_head_training",
        "instrument": instrument,
        "dataset_id": dataset_id,
        "config_hash": digest(config),
        "code_version": _git_state(config_path.parent.parent.parent),
        "timesfm_code_commit": CODE_COMMIT,
        "timesfm_weight_revision": WEIGHT_REVISION,
        "train_end": str(train_samples[-1].day),
        "early_stop_split": [str(early_stop_samples[0].day), str(early_stop_samples[-1].day)],
        "feature_names": data.research_feature_names(),
        "candidate_index": candidate_index,
        "spec": spec.__dict__,
    }
    experiment_dir = output_root / "timesfm3-output-head" / digest(identity)[:20]
    result_path = experiment_dir / "result.json"
    checkpoint_path = experiment_dir / "output_head.safetensors"
    if result_path.exists():
        return result_path
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    forecaster = load_pinned_forecaster(cache_dir, local_files_only=True)
    model = forecaster.model
    original_head = {
        name: value.detach().cpu().clone()
        for name, value in model.output_head.state_dict().items()
    }
    try:
        training = train_output_head(
            model,
            train_samples,
            early_stop_samples,
            spec,
            seed=int(config["seed"]),
            deadline=started + float(config["budget_cpu_seconds"]),
        )
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.output_head.state_dict().items()
            },
            checkpoint_path,
        )
        result = {
            "schema_version": "timesfm3-custom-output-head-training-v1",
            "status": "TRAINED_NOT_PROMOTED",
            "official_timesfm3_finetuning": False,
            "training_scope": "pinned TimesFM 3 backbone frozen; output_head only",
            "final_test_opened": False,
            "old_final_reused_as_independent_test": False,
            "production_candidate": False,
            "identity": identity,
            "weight_repo": WEIGHT_REPO,
            "train_samples": len(train_samples),
            "early_stop_samples": len(early_stop_samples),
            "training": training,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "scope": "output_head_only_local_research_artifact",
            },
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "limitations": [
                "This is a custom research training path, not an official TimesFM fine-tuning API.",
                "Training and early stopping do not establish walk-forward improvement.",
                "Promotion requires the separate multi-fold development gate and a new future holdout.",
            ],
        }
        write_once(result_path, result)
        return result_path
    finally:
        model.output_head.load_state_dict(original_head)
        model.eval()


def _prediction_row(model, sample, mean, scale, instrument):
    point, interval, quantiles = predict_head(model, sample, mean, scale)
    return {
        "date": str(sample.day),
        "instrument": instrument,
        "contract": sample.contract,
        "status": "OK",
        "previous_close": sample.previous_close,
        "predicted": point.model_dump(),
        "actual": sample.actual,
        "interval": interval,
        "interval_method": "rank-aligned TimesFM transformed quantiles",
        "raw_transformed_quantiles": quantiles.tolist(),
    }


def run_output_head_walk_forward(
    dataset_dir: Path,
    config_path: Path,
    cache_dir: Path,
    output_root: Path,
    instrument: str,
    candidate_index: int = 0,
) -> Path:
    """Run the fixed custom head through expanding development folds."""
    from safetensors.torch import save_file

    if instrument not in {"TAIEX", "TX"}:
        raise ValueError("instrument must be TAIEX or TX")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    probe = runtime_probe(cache_dir, int(config["memory_probe_limit_mb"]))
    if probe.status != "READY_TO_LOAD":
        raise RuntimeError(f"TimesFM 3 load blocked by preflight: {probe.status}")
    data, dataset_id = load_dataset(dataset_dir)
    development_end = date.fromisoformat(config["development_end"])
    eligible_days = [
        day for day in data.calendar.days
        if date.fromisoformat(config["train_start"]) <= day <= development_end
    ]
    folds = expanding_folds(
        eligible_days,
        folds=int(config["walk_forward_folds"]),
        validation_sessions=int(config["fold_validation_sessions"]),
        minimum_train=int(config["minimum_train_sessions"]),
    )
    spec = _spec_from_config(config, candidate_index)
    identity = {
        "name": "custom_timesfm3_output_head_expanding_walk_forward",
        "instrument": instrument,
        "dataset_id": dataset_id,
        "config_hash": digest(config),
        "code_version": _git_state(config_path.parent.parent.parent),
        "timesfm_code_commit": CODE_COMMIT,
        "timesfm_weight_revision": WEIGHT_REVISION,
        "folds": [
            {
                "train": [str(train[0]), str(train[-1])],
                "validation": [str(validation[0]), str(validation[-1])],
            }
            for train, validation in folds
        ],
        "feature_names": data.research_feature_names(),
        "candidate_index": candidate_index,
        "spec": spec.__dict__,
    }
    experiment_dir = output_root / "timesfm3-output-head-walk-forward" / digest(identity)[:20]
    result_path = experiment_dir / "result.json"
    if result_path.exists():
        return result_path
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(config["budget_cpu_seconds"])
    forecaster = load_pinned_forecaster(cache_dir, local_files_only=True)
    model = forecaster.model
    original_head = {
        name: value.detach().cpu().clone()
        for name, value in model.output_head.state_dict().items()
    }
    fold_results = []
    candidate_rows = []
    baseline_rows = []
    fold_improvements = []
    try:
        for fold_index, (train_days, validation_days) in enumerate(folds, start=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("output-head walk-forward budget exhausted")
            fold_result_path = experiment_dir / f"fold-{fold_index}.json"
            if fold_result_path.exists():
                saved = json.loads(fold_result_path.read_text(encoding="utf-8"))
                if (
                    saved.get("identity_hash") != digest(identity)
                    or saved.get("fold") != fold_index
                ):
                    raise ValueError(f"output-head fold identity mismatch: {fold_result_path}")
                reused_summary = {**saved["summary"], "reused": True}
                fold_results.append(reused_summary)
                candidate_rows.extend(saved["candidate_rows"])
                baseline_rows.extend(saved["baseline_rows"])
                fold_improvements.append(reused_summary["close_relative_improvement"])
                continue
            model.output_head.load_state_dict(original_head)
            early_stop_sessions = int(config["early_stop_sessions"])
            usable_train_days = train_days[spec.context:]
            if len(usable_train_days) <= early_stop_sessions:
                raise ValueError("fold lacks output-head training samples")
            stride = int(config["training_stride"])
            if stride < 1:
                raise ValueError("training_stride must be positive")
            training_days = usable_train_days[:-early_stop_sessions:stride]
            early_stop_days = usable_train_days[-early_stop_sessions:]
            training_samples = build_head_samples(
                data, instrument, training_days, spec.context
            )
            early_stop_samples = build_head_samples(
                data, instrument, early_stop_days, spec.context
            )
            training = train_output_head(
                model,
                training_samples,
                early_stop_samples,
                spec,
                seed=int(config["seed"]) + fold_index,
                deadline=deadline,
            )
            checkpoint_path = experiment_dir / f"fold-{fold_index}-output_head.safetensors"
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in model.output_head.state_dict().items()
                },
                checkpoint_path,
            )
            mean = np.asarray(training["feature_mean"], dtype=np.float32)
            scale = np.asarray(training["feature_scale"], dtype=np.float32)
            fold_candidate = []
            for validation_day in validation_days:
                try:
                    sample = build_head_samples(
                        data, instrument, [validation_day], spec.context
                    )[0]
                    fold_candidate.append(
                        _prediction_row(model, sample, mean, scale, instrument)
                    )
                except Exception as exc:  # noqa: BLE001 - preserve every expected day
                    fold_candidate.append({
                        "date": str(validation_day),
                        "instrument": instrument,
                        "contract": data.contract(instrument, validation_day),
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    })

            baseline_validation = {}
            expected_dates = [str(day) for day in validation_days]
            for baseline_model in config["stable_baselines"]:
                rows = walk_forward(
                    data,
                    instrument,
                    baseline_model,
                    "gbud",
                    validation_days[0],
                    validation_days[-1],
                    config,
                )
                baseline_validation[f"{instrument}/gbud/{baseline_model}"] = {
                    "rows": rows,
                    "metrics": metrics(rows, config["neutral_bps"]),
                }
            selected_baseline, selection_audit = _select_baseline(
                baseline_validation,
                instrument,
                expected_dates,
                float(config["promotion_min_coverage"]),
                float(config["neutral_bps"]),
                config["baseline_stability"],
            )
            baseline_key = (
                f"{instrument}/{selected_baseline['representation']}/"
                f"{selected_baseline['model']}"
            )
            fold_baseline = baseline_validation[baseline_key]["rows"]
            candidate_metrics = metrics(fold_candidate, config["neutral_bps"])
            baseline_metrics = metrics(fold_baseline, config["neutral_bps"])
            improvement = 1 - (
                candidate_metrics["ohlc"]["close"]["mae_points"]
                / baseline_metrics["ohlc"]["close"]["mae_points"]
            )
            fold_improvements.append(improvement)
            candidate_rows.extend(fold_candidate)
            baseline_rows.extend(fold_baseline)
            fold_summary = {
                "fold": fold_index,
                "train_split": [str(train_days[0]), str(train_days[-1])],
                "validation_split": [str(validation_days[0]), str(validation_days[-1])],
                "training": training,
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                },
                "selected_baseline": selected_baseline,
                "baseline_selection_audit": selection_audit,
                "candidate_metrics": candidate_metrics,
                "baseline_metrics": baseline_metrics,
                "close_relative_improvement": improvement,
                "reused": False,
            }
            write_once(fold_result_path, {
                "identity_hash": digest(identity),
                "fold": fold_index,
                "summary": fold_summary,
                "candidate_rows": fold_candidate,
                "baseline_rows": fold_baseline,
            })
            fold_results.append(fold_summary)
        gate = development_acceptance(
            candidate_rows,
            baseline_rows,
            config,
            fold_improvements=fold_improvements,
        )
        result = {
            "schema_version": "timesfm3-output-head-walk-forward-v1",
            "status": (
                "PROMOTION_GATE_PASS_AWAITING_NEW_FORWARD_HOLDOUT"
                if gate["status"] == "PASS"
                else "DEVELOPMENT_GATE_FAIL"
            ),
            "official_timesfm3_finetuning": False,
            "training_scope": "pinned TimesFM 3 backbone frozen; output_head only",
            "final_test_opened": False,
            "old_final_reused_as_independent_test": False,
            "production_candidate": False,
            "identity": identity,
            "weight_repo": WEIGHT_REPO,
            "fold_results": fold_results,
            "development_gate": gate,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "limitations": [
                "All dates through 2026-10-08 are development evidence after the old final failed.",
                "A passing development gate still requires a newly locked future holdout.",
                "This custom training path is not an official TimesFM fine-tuning API.",
            ],
        }
        write_once(result_path, result)
        return result_path
    finally:
        model.output_head.load_state_dict(original_head)
        model.eval()
