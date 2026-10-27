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
from .contracts import EPS, OHLC, decode, digest, encode, write_once
from .evaluation import block_improvement, metrics
from .head_budget import bounded_head_run, budget_deadline
from .numeric_runtime import capture_torch_cpu_runtime, require_matching_torch_cpu_runtime
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
    previous_close: float | tuple[float, float]
    actual: dict[str, float] | dict[str, dict[str, float]]
    contract: str | dict[str, str]


@dataclass(frozen=True)
class FrozenHeadFeatures:
    """Output-head inputs whose frozen TimesFM backbone need not be rerun."""

    hidden: object
    revin_mean: object
    revin_std: object
    total_variates: int
    total_patches: int
    selected_patch: int
    target_count: int


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
    wick_weight: float = 0.0
    loss_normalization: str = "none"

    def validate(self, input_patch_len: int = 32) -> None:
        if self.loss_normalization not in {"none", "train_std"}:
            raise ValueError("unsupported loss normalization")
        if self.context < input_patch_len or self.context % input_patch_len:
            raise ValueError("TimesFM head context must be a positive patch multiple")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("invalid training limits")
        if min(
            self.point_weight,
            self.close_weight,
            self.quantile_weight,
            self.crossing_weight,
            self.wick_weight,
        ) < 0:
            raise ValueError("loss weights must be nonnegative")


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
        wick_weight=float(config.get("wick_weight", 0.0)),
        loss_normalization=config.get("loss_normalization", "none"),
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


def build_head_inference_sample(data, instrument: str, day: date, context: int):
    """Build pre-open inputs without accessing D's daytime labels."""
    if instrument not in {"TAIEX", "TX", "JOINT"}:
        raise ValueError("instrument must be TAIEX, TX, or JOINT")
    markets = ("TAIEX", "TX") if instrument == "JOINT" else (instrument,)
    histories, features, future, references, contracts, dates = [], [], [], [], {}, []
    for market in markets:
        history, past, used = data.research_history(market, day, "gbud", context, context)
        histories.append(history)
        features.append(past)
        dates.append(used)
        future.append(data.research_features(market, day))
        references.append(float(data.reference(market, day).close))
        contracts[market] = data.contract(market, day)
    if any(used != dates[0] for used in dates):
        raise ValueError("joint output-head histories are not aligned")
    return HeadSample(
        day=day,
        history=np.column_stack(histories).astype(np.float32),
        past_features=np.column_stack(features).astype(np.float32),
        forecast_features=np.concatenate(future).astype(np.float32),
        # Targets are absent at inference; NaNs prevent accidental training reuse.
        target=np.full(4 * len(markets), np.nan, dtype=np.float32),
        previous_close=tuple(references) if instrument == "JOINT" else references[0],
        actual={},
        contract=contracts if instrument == "JOINT" else contracts[instrument],
    )


def restore_output_head(model, checkpoint_path: Path, result: dict, *, forecast_day: date):
    """Validate a standalone trained artifact before mutating the pinned model."""
    import torch
    from safetensors.torch import load_file

    schema = result.get("schema_version")
    if schema not in ("timesfm3-custom-output-head-training-v1",
                      "timesfm3-walk-forward-fold-inference-v1"):
        raise ValueError("requires standalone or provenance-checked walk-forward head result")
    identity = result["identity"]
    if (identity.get("timesfm_code_commit") != CODE_COMMIT
            or identity.get("timesfm_weight_revision") != WEIGHT_REVISION
            or result.get("weight_repo") != WEIGHT_REPO):
        raise ValueError("checkpoint model revision mismatch")
    # Early-stop labels selected this head and must precede the forecast too.
    if schema == "timesfm3-walk-forward-fold-inference-v1":
        if result.get("research_only") is not True or result.get("production_candidate") is not False:
            raise ValueError("walk-forward head is research only")
        declared_end = identity["folds"][result["source_fold"] - 1]["train"][1]
        if result["training_label_end"] != declared_end:
            raise ValueError("fold label boundary mismatch")
        last_label = declared_end
    else:
        last_label = max(identity["train_end"], identity["early_stop_split"][1])
    if date.fromisoformat(last_label) >= forecast_day:
        raise ValueError("checkpoint contains forecast-day or future labels")
    if _sha256(checkpoint_path) != result["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    training = result["training"]
    if "numeric_runtime" in training:
        require_matching_torch_cpu_runtime(training["numeric_runtime"])
    if training.get("epochs_completed", 0) < 1:
        raise ValueError("checkpoint has no completed training epoch")
    mean = np.asarray(training["feature_mean"], dtype=np.float32)
    scale = np.asarray(training["feature_scale"], dtype=np.float32)
    names = identity["feature_names"]
    feature_count = sum(len(values) for values in names.values()) if isinstance(names, dict) else len(names)
    if (mean.shape != (feature_count,) or scale.shape != mean.shape
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)):
        raise ValueError("invalid checkpoint feature scaler")
    state = load_file(str(checkpoint_path), device="cpu")
    expected = model.output_head.state_dict()
    if state.keys() != expected.keys() or any(
        state[name].shape != expected[name].shape
        or state[name].dtype != expected[name].dtype
        or not torch.isfinite(state[name]).all()
        for name in expected
    ):
        raise ValueError("checkpoint head schema or finite-value mismatch")
    model.output_head.load_state_dict(state, strict=True)
    model.eval()
    return mean, scale


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


def build_joint_head_samples(data, days: list[date], context: int) -> list[HeadSample]:
    """Materialize aligned eight-target TAIEX/TX samples and market-specific features."""
    samples = []
    for day in days:
        taiex_history, taiex_features, taiex_dates = data.research_history(
            "TAIEX", day, "gbud", context, context
        )
        tx_history, tx_features, tx_dates = data.research_history(
            "TX", day, "gbud", context, context
        )
        if taiex_dates != tx_dates:
            raise ValueError("joint output-head histories are not aligned")
        references = (
            float(data.reference("TAIEX", day).close),
            float(data.reference("TX", day).close),
        )
        actual_bars = {
            instrument: data.actual(instrument, day)
            for instrument in ("TAIEX", "TX")
        }
        targets = np.concatenate([
            encode(actual_bars["TAIEX"], references[0], "gbud"),
            encode(actual_bars["TX"], references[1], "gbud"),
        ]).astype(np.float32)
        samples.append(HeadSample(
            day=day,
            history=np.column_stack([taiex_history, tx_history]).astype(np.float32),
            past_features=np.column_stack([taiex_features, tx_features]).astype(np.float32),
            forecast_features=np.concatenate([
                data.research_features("TAIEX", day),
                data.research_features("TX", day),
            ]).astype(np.float32),
            target=targets,
            previous_close=references,
            actual={
                instrument: {
                    field: float(getattr(actual_bars[instrument], field))
                    for field in OHLC
                }
                for instrument in ("TAIEX", "TX")
            },
            contract={"TAIEX": "", "TX": data.contract("TX", day)},
        ))
    return samples


def materialize_training_samples(data, instrument, days, context):
    """Audit expected history failures without swallowing unrelated programming errors."""
    samples, records = [], []
    for day in days:
        try:
            batch = (
                build_joint_head_samples(data, [day], context)
                if instrument == "JOINT"
                else build_head_samples(data, instrument, [day], context)
            )
        except ValueError as exc:
            if not str(exc).startswith("insufficient contiguous expanded history:"):
                raise
            records.append({
                "date": str(day), "status": "INELIGIBLE_HISTORY",
                "reason": str(exc),
            })
        else:
            samples.extend(batch)
            records.append({"date": str(day), "status": "OK"})
    return samples, {
        "requested": len(days), "usable": len(samples),
        "ineligible": len(days) - len(samples), "dates": records,
        "policy": "training_only; validation dates never filtered",
    }


def audited_head_samples(data, instrument: str, days: list[date], context: int):
    """Cloud-compatible audit using only classified history exclusions."""
    if instrument not in {"TAIEX", "TX", "JOINT"}:
        raise ValueError("instrument must be TAIEX, TX, or JOINT")
    samples, audit = materialize_training_samples(data, instrument, days, context)
    return samples, {
        "requested": audit["requested"], "usable": audit["usable"],
        "failed": audit["ineligible"], "rows": audit["dates"],
    }


def _persist_sample_audit(path: Path, identity: dict, training: dict, early_stop: dict):
    """Persist data exclusions before any optimizer step, including empty splits."""
    audit = {
        "schema_version": "timesfm3-head-sample-audit-v1",
        "identity_hash": digest(identity),
        "instrument": identity["instrument"],
        "policy": "audited_optimization_exclusions_no_imputation_no_gap_compression",
        "training": training,
        "early_stop": early_stop,
        "validation_dates_filtered": False,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != audit:
            raise ValueError(f"sample audit changed during resume: {path}")
    else:
        write_once(path, audit)
    if training["usable"] < 60 or not early_stop["usable"]:
        raise ValueError(f"fewer than 60 training samples or empty early-stop split; audit: {path}")
    return {
        "path": str(path), "sha256": _sha256(path),
        "training_requested": training["requested"],
        "training_usable": training["usable"],
        "training_failed": training["failed"],
        "early_stop_requested": early_stop["requested"],
        "early_stop_usable": early_stop["usable"],
        "early_stop_failed": early_stop["failed"],
    }


def _batch_inputs(model, samples: list[HeadSample], mean: np.ndarray, scale: np.ndarray):
    import torch

    if not samples:
        raise ValueError("empty TimesFM training batch")
    context = samples[0].history.shape[0]
    if context % model.input_patch_len:
        raise ValueError("TimesFM training context must align to input patches")
    target_count = samples[0].history.shape[1]
    if target_count not in {4, 8}:
        raise ValueError("TimesFM head supports four or eight GBUD targets")
    feature_count = samples[0].past_features.shape[1]
    if any(
        sample.history.shape != (context, target_count)
        or sample.past_features.shape != (context, feature_count)
        or sample.forecast_features.shape != (feature_count,)
        or sample.target.shape != (target_count,)
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
    horizon_values[:, target_count:, 0] = torch.tensor(future, dtype=torch.float32)
    horizon_masks[:, target_count:, 0] = False

    all_values = torch.cat([context_values, horizon_values], dim=-1)
    all_masks = torch.cat(
        [torch.zeros_like(context_values, dtype=torch.bool), horizon_masks], dim=-1
    )
    values = all_values.reshape(batch, variates, -1, model.input_patch_len)
    masks = all_masks.reshape(batch, variates, -1, model.input_patch_len)
    target_flags = torch.zeros((batch, variates, values.shape[2]), dtype=torch.bool)
    target_flags[:, :target_count, :] = True
    context_patches = context // model.input_patch_len
    cpm_mask = torch.zeros((batch, values.shape[2]), dtype=torch.bool)
    cpm_mask[:, context_patches:] = True
    return {
        "values": values,
        "masks": masks,
        "patch_is_target": target_flags,
    }, cpm_mask, context_patches, target_count


def _one_step_logits(model, inputs, cpm_mask, context_patches, target_count):
    output = model.forward(inputs, patch_cpm_mask=cpm_mask)
    return output["logits"][:, :target_count, context_patches - 1, 0, :]


def _supports_frozen_head_cache(model) -> bool:
    """Restrict the optimized path to the verified TimesFM 3 forward structure."""
    return (
        model.__class__.__module__ == "timesfm3.torch.model"
        and hasattr(model, "_preprocess")
        and hasattr(model, "transformer_stack")
        and hasattr(model, "output_head")
    )


def _frozen_head_features(model, inputs, cpm_mask, context_patches, target_count):
    """Run the frozen official backbone once and retain the selected head inputs.

    The custom trainer predicts step zero from the last observed context patch.
    That patch is outside the CPM region, so its reverse-RevIN statistics do not
    depend on the trainable output head. This mirrors TimesFM3Torch.forward up to
    (but excluding) output_head and keeps gradients only through output_head.
    """
    import torch

    if not _supports_frozen_head_cache(model):
        raise TypeError("frozen head cache requires the pinned TimesFM 3 torch model")
    selected_patch = context_patches - 1
    if selected_patch < 0 or cpm_mask[:, selected_patch].any():
        raise ValueError("cached head patch must be the last observed non-CPM patch")
    with torch.no_grad():
        values = torch.nan_to_num(inputs["values"], nan=0.0)
        values = torch.clamp(values, -model.value_clip, model.value_clip)
        masks = inputs["masks"].bool()
        patch_is_target = inputs["patch_is_target"]
        if values.shape[-1] != model.input_patch_len:
            raise ValueError("cached TimesFM input patch length mismatch")
        (
            _, transformer_input, transformer_patch_mask, revin_stats, _,
        ) = model._preprocess(
            values,
            masks,
            patch_is_target,
            freeze_after=None,
            patch_cpm_mask=cpm_mask,
        )
        effective_patch_mask = torch.cumprod(
            transformer_patch_mask.int(), dim=2
        ).bool()
        transformer_output, _, _ = model.transformer_stack(
            transformer_input, effective_patch_mask
        )
        revin_mean, revin_std = revin_stats
        return FrozenHeadFeatures(
            hidden=transformer_output[:, :target_count, selected_patch].detach().clone(),
            revin_mean=revin_mean[:, :target_count, selected_patch].detach().clone(),
            revin_std=revin_std[:, :target_count, selected_patch].detach().clone(),
            total_variates=transformer_output.shape[1],
            total_patches=transformer_output.shape[2],
            selected_patch=selected_patch,
            target_count=target_count,
        )


def _one_step_logits_from_features(model, features: FrozenHeadFeatures):
    """Apply the trainable head and exact reverse-RevIN for cached features."""
    import torch

    # Preserve the same Linear input matrix shape used by TimesFM3Torch.forward.
    # A smaller target-only matmul can differ by a few float32 ulps on CPU and
    # is enough to break strict checkpoint replay after price reconstruction.
    batch, _, hidden_dims = features.hidden.shape
    full_hidden = torch.zeros(
        (batch, features.total_variates, features.total_patches, hidden_dims),
        dtype=features.hidden.dtype,
        device=features.hidden.device,
    )
    full_hidden[:, : features.target_count, features.selected_patch] = features.hidden
    raw = model.output_head(full_hidden)
    full_mean = torch.zeros(
        raw.shape[:3], dtype=features.revin_mean.dtype, device=features.revin_mean.device
    )
    full_std = torch.ones(
        raw.shape[:3], dtype=features.revin_std.dtype, device=features.revin_std.device
    )
    full_mean[:, : features.target_count, features.selected_patch] = features.revin_mean
    full_std[:, : features.target_count, features.selected_patch] = features.revin_std
    logits = raw * full_std.unsqueeze(-1) + full_mean.unsqueeze(-1)
    logits = torch.clamp(logits, -model.value_clip, model.value_clip)
    return logits.view(
        logits.shape[0],
        logits.shape[1],
        logits.shape[2],
        model.output_patch_len,
        len(model.quantiles),
    )[:, : features.target_count, features.selected_patch, 0, :]


def _stack_frozen_head_features(features: list[FrozenHeadFeatures]):
    import torch

    if not features:
        raise ValueError("cannot stack empty frozen head feature cache")
    metadata = {
        (item.total_variates, item.total_patches, item.selected_patch, item.target_count)
        for item in features
    }
    if len(metadata) != 1:
        raise ValueError("frozen head cache shapes differ within a batch")
    total_variates, total_patches, selected_patch, target_count = metadata.pop()
    return FrozenHeadFeatures(
        hidden=torch.cat([item.hidden for item in features], dim=0),
        revin_mean=torch.cat([item.revin_mean for item in features], dim=0),
        revin_std=torch.cat([item.revin_std for item in features], dim=0),
        total_variates=total_variates,
        total_patches=total_patches,
        selected_patch=selected_patch,
        target_count=target_count,
    )


def _cache_frozen_head_features(model, samples, mean, scale, *, deadline):
    """Cache one-sample backbone results and verify the optimized math once."""
    import torch

    cached = []
    parity_max_abs = 0.0
    for index, sample in enumerate(samples):
        if time.monotonic() >= deadline:
            raise TimeoutError("training budget expired while caching frozen backbone")
        inputs, mask, patches, target_count = _batch_inputs(model, [sample], mean, scale)
        features = _frozen_head_features(model, inputs, mask, patches, target_count)
        if index == 0:
            with torch.no_grad():
                expected = _one_step_logits(model, inputs, mask, patches, target_count)
                actual = _one_step_logits_from_features(model, features)
                parity_max_abs = float(torch.max(torch.abs(expected - actual)))
            if parity_max_abs > 1e-6:
                raise RuntimeError(
                    f"cached frozen-backbone logits differ from full forward: {parity_max_abs}"
                )
        cached.append(features)
    return cached, parity_max_abs


def training_loss_scales(samples):
    """Fit loss units from optimization labels only, never early-stop/validation."""
    if not samples:
        raise ValueError("loss scales require training samples")
    targets = np.stack([s.target for s in samples]).astype(np.float64)
    if targets.ndim != 2 or targets.shape[1] not in (4, 8) or not np.isfinite(targets).all():
        raise ValueError("invalid training labels for loss scaling")
    scales = np.maximum(targets.std(axis=0), 1e-6)
    closes = targets[:, 0::4] + targets[:, 1::4]
    close_scales = np.maximum(closes.std(axis=0), 1e-6)
    extensions_up = np.maximum(np.exp(targets[:, 2::4]) - EPS, 0.0)
    extensions_down = np.maximum(np.exp(targets[:, 3::4]) - EPS, 0.0)
    wick_spans = extensions_up + extensions_down
    log_wick_spans = np.log(np.maximum(wick_spans, EPS))
    wick_scales = np.maximum(log_wick_spans.std(axis=0), 1e-6)
    return {"target": scales.tolist(), "close": close_scales.tolist(),
            "wick": wick_scales.tolist(),
            "fit_samples": len(samples), "fit_source": "optimization_training_labels_only",
            "minimum_scale": 1e-6}


def output_head_loss(logits, target, quantiles, spec: HeadTrainingSpec, loss_scales=None):
    """Robust point, close-return, quantile and crossing losses for GBUD."""
    import torch

    if logits.ndim != 3 or logits.shape[1] not in {4, 8}:
        raise ValueError("TimesFM head logits must contain four or eight targets")
    if target.shape != logits.shape[:2]:
        raise ValueError("TimesFM head target shape mismatch")
    q = torch.as_tensor(quantiles, dtype=logits.dtype, device=logits.device)
    if q.shape != (logits.shape[2],):
        raise ValueError("TimesFM quantile schema mismatch")
    median_index = int(torch.argmin(torch.abs(q - 0.5)))
    point = logits[:, :, median_index]
    if spec.loss_normalization == "train_std":
        if loss_scales is None:
            raise ValueError("train_std loss requires frozen training scales")
        units = torch.as_tensor(loss_scales["target"], dtype=logits.dtype, device=logits.device)
        close_units = torch.as_tensor(loss_scales["close"], dtype=logits.dtype, device=logits.device)
        wick_units = torch.as_tensor(
            loss_scales.get("wick", []), dtype=logits.dtype, device=logits.device
        )
        if (units.shape != (target.shape[1],) or close_units.shape != (target.shape[1] // 4,)
                or not torch.isfinite(units).all() or not torch.isfinite(close_units).all()
                or (units <= 0).any() or (close_units <= 0).any()):
            raise ValueError("invalid frozen loss scales")
        if spec.wick_weight and (
            wick_units.shape != (target.shape[1] // 4,)
            or not torch.isfinite(wick_units).all()
            or (wick_units <= 0).any()
        ):
            raise ValueError("invalid frozen wick loss scales")
    elif spec.loss_normalization == "none":
        units, close_units, wick_units = 1.0, 1.0, 1.0
    else:
        raise ValueError("unsupported loss normalization")
    point_loss = torch.nn.functional.smooth_l1_loss(point / units, target / units)
    close_indexes = [(index, index + 1) for index in range(0, point.shape[1], 4)]
    predicted_close = torch.stack([point[:, g] + point[:, b] for g, b in close_indexes], 1)
    actual_close = torch.stack([target[:, g] + target[:, b] for g, b in close_indexes], 1)
    close_loss = torch.nn.functional.smooth_l1_loss(
        predicted_close / close_units, actual_close / close_units)
    normalized_logits = logits / (units[None, :, None] if torch.is_tensor(units) else units)
    errors = (target / units)[:, :, None] - normalized_logits
    pinball = torch.maximum(q * errors, (q - 1) * errors).mean()
    crossing = torch.relu(normalized_logits[:, :, :-1] - normalized_logits[:, :, 1:]).mean()
    wick_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if spec.wick_weight:
        predicted_wicks = []
        actual_wicks = []
        for offset in range(0, point.shape[1], 4):
            predicted_wicks.append(
                torch.clamp(torch.exp(point[:, offset + 2]) - EPS, min=0.0)
                + torch.clamp(torch.exp(point[:, offset + 3]) - EPS, min=0.0)
            )
            actual_wicks.append(
                torch.clamp(torch.exp(target[:, offset + 2]) - EPS, min=0.0)
                + torch.clamp(torch.exp(target[:, offset + 3]) - EPS, min=0.0)
            )
        predicted_wick = torch.log(
            torch.clamp(torch.stack(predicted_wicks, dim=1), min=EPS)
        )
        actual_wick = torch.log(
            torch.clamp(torch.stack(actual_wicks, dim=1), min=EPS)
        )
        wick_loss = torch.nn.functional.smooth_l1_loss(
            predicted_wick / wick_units, actual_wick / wick_units
        )
    return (
        spec.point_weight * point_loss
        + spec.close_weight * close_loss
        + spec.quantile_weight * pinball
        + spec.crossing_weight * crossing
        + spec.wick_weight * wick_loss
    )


def _evaluate_loss(
    model,
    samples,
    mean,
    scale,
    spec,
    deadline=math.inf,
    loss_scales=None,
    feature_cache=None,
):
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), spec.batch_size):
            if time.monotonic() >= deadline:
                raise TimeoutError("early-stop evaluation budget exhausted")
            batch = samples[start : start + spec.batch_size]
            target = torch.tensor(np.stack([item.target for item in batch]))
            if feature_cache is None:
                inputs, mask, patches, target_count = _batch_inputs(
                    model, batch, mean, scale
                )
                logits = _one_step_logits(model, inputs, mask, patches, target_count)
            else:
                logits = _one_step_logits_from_features(
                    model,
                    _stack_frozen_head_features(
                        feature_cache[start : start + spec.batch_size]
                    ),
                )
            losses.append(float(output_head_loss(logits, target, model.quantiles, spec, loss_scales)))
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
    numeric_runtime = capture_torch_cpu_runtime()
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
    loss_scales = training_loss_scales(train_samples) if spec.loss_normalization == "train_std" else None
    train_feature_cache = None
    early_stop_feature_cache = None
    cache_metadata = {
        "enabled": False,
        "method": "full_forward_each_optimizer_step",
    }
    if _supports_frozen_head_cache(model):
        train_feature_cache, train_parity = _cache_frozen_head_features(
            model, train_samples, mean, scale, deadline=deadline
        )
        early_stop_feature_cache, early_stop_parity = _cache_frozen_head_features(
            model, early_stop_samples, mean, scale, deadline=deadline
        )
        cache_metadata = {
            "enabled": True,
            "method": "last_observed_patch_transformer_features_v1",
            "train_samples": len(train_feature_cache),
            "early_stop_samples": len(early_stop_feature_cache),
            "full_forward_parity_max_abs": max(train_parity, early_stop_parity),
            "parity_tolerance": 1e-6,
        }
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
            target = torch.tensor(np.stack([item.target for item in batch]))
            optimizer.zero_grad(set_to_none=True)
            if train_feature_cache is None:
                inputs, mask, patches, target_count = _batch_inputs(
                    model, batch, mean, scale
                )
                logits = _one_step_logits(model, inputs, mask, patches, target_count)
            else:
                logits = _one_step_logits_from_features(
                    model,
                    _stack_frozen_head_features(
                        [train_feature_cache[index] for index in order[offset : offset + spec.batch_size]]
                    ),
                )
            loss = output_head_loss(logits, target, model.quantiles, spec, loss_scales)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite output-head training loss")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        if len(epoch_losses) != math.ceil(len(order) / spec.batch_size):
            break
        try:
            validation_loss = _evaluate_loss(
                model,
                early_stop_samples,
                mean,
                scale,
                spec,
                deadline=deadline,
                loss_scales=loss_scales,
                feature_cache=early_stop_feature_cache,
            )
        except TimeoutError:
            break
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
    require_matching_torch_cpu_runtime(numeric_runtime)
    return {
        "numeric_runtime": numeric_runtime,
        "loss_normalization": spec.loss_normalization,
        "loss_scales": loss_scales,
        "best_epoch": best_epoch,
        "best_early_stop_loss": best_loss,
        "epochs_completed": len(history),
        "stopped_early": stale >= spec.patience,
        "budget_exhausted": time.monotonic() >= deadline,
        "history": history,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "frozen_backbone_cache": cache_metadata,
        "trainable_parameter_names": trainable,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.output_head.parameters()
        ),
    }


def predict_head(model, sample: HeadSample, mean: np.ndarray, scale: np.ndarray):
    import torch

    model.eval()
    with torch.no_grad():
        inputs, mask, patches, target_count = _batch_inputs(
            model, [sample], mean, scale
        )
        if target_count != 4:
            raise ValueError("predict_head requires a market-specific four-target sample")
        logits = _one_step_logits(
            model, inputs, mask, patches, target_count
        )[0].cpu().numpy()
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


def _decode_quantile_block(values: np.ndarray, previous_close: float):
    ordered = np.sort(values, axis=1)
    median = ordered[:, ordered.shape[1] // 2]
    point = decode(median, previous_close, "gbud")
    coherent_paths = np.stack([
        decode(ordered[:, index], previous_close, "gbud").array()
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


def predict_joint_head(
    model, sample: HeadSample, mean: np.ndarray, scale: np.ndarray
) -> dict[str, tuple]:
    """Decode one eight-target head with market-specific reference prices."""
    import torch

    if not isinstance(sample.previous_close, tuple) or len(sample.previous_close) != 2:
        raise ValueError("joint head requires separate TAIEX and TX reference prices")
    model.eval()
    with torch.no_grad():
        inputs, mask, patches, target_count = _batch_inputs(
            model, [sample], mean, scale
        )
        if target_count != 8:
            raise ValueError("predict_joint_head requires an eight-target sample")
        logits = _one_step_logits(
            model, inputs, mask, patches, target_count
        )[0].cpu().numpy()
    return {
        instrument: _decode_quantile_block(
            logits[offset : offset + 4], sample.previous_close[index]
        )
        for index, (instrument, offset) in enumerate((("TAIEX", 0), ("TX", 4)))
    }


def paired_validation_metrics(candidate_rows, baseline_rows, threshold):
    """Compare the same dates, retaining full-population failure accounting."""
    indexes = []
    for rows in (candidate_rows, baseline_rows):
        if len({row["date"] for row in rows}) != len(rows):
            raise ValueError("duplicate validation date")
        indexes.append({row["date"]: row for row in rows if row["status"] == "OK"})
    candidate, baseline = indexes
    dates = sorted(candidate.keys() & baseline.keys())
    for day in dates:
        if (candidate[day]["actual"] != baseline[day]["actual"]
                or candidate[day]["previous_close"] != baseline[day]["previous_close"]):
            raise ValueError(f"validation truth/reference mismatch: {day}")
    cm = metrics([candidate[day] for day in dates], threshold)
    bm = metrics([baseline[day] for day in dates], threshold)
    denominator = bm.get("ohlc", {}).get("close", {}).get("mae_points", 0)
    improvement = (
        1 - cm["ohlc"]["close"]["mae_points"] / denominator
        if dates and denominator > 0 else None
    )
    expected = {row["date"] for row in candidate_rows + baseline_rows}
    return {
        "status": ("OK" if improvement is not None else
                   "FAIL_NO_PAIRED_DAYS" if not dates else "FAIL_ZERO_BASELINE_ERROR"),
        "paired_days": len(dates),
        "paired_dates": dates,
        "paired_coverage": len(dates) / len(expected) if expected else 0.0,
        "close_relative_improvement": improvement,
        "candidate_metrics": cm,
        "baseline_metrics": bm,
        "candidate_all_dates_metrics": metrics(candidate_rows, threshold),
        "baseline_all_dates_metrics": metrics(baseline_rows, threshold),
    }


def development_acceptance(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    config: dict,
    *,
    fold_improvements: list[float | None],
) -> dict:
    """Stricter pre-forward gate; this never opens or relabels the old final."""
    candidate = {row["date"]: row for row in candidate_rows if row["status"] == "OK"}
    baseline = {row["date"]: row for row in baseline_rows if row["status"] == "OK"}
    paired_dates = sorted(candidate.keys() & baseline.keys())
    candidate_paired = [candidate[day] for day in paired_dates]
    baseline_paired = [baseline[day] for day in paired_dates]
    comparison = paired_validation_metrics(
        candidate_rows, baseline_rows, config["neutral_bps"]
    )
    if comparison["status"] != "OK":
        return {**comparison, "fold_improvements": fold_improvements}
    candidate_metrics = comparison["candidate_metrics"]
    baseline_metrics = comparison["baseline_metrics"]
    close_improvement = comparison["close_relative_improvement"]
    ohl_degradation = {
        field: (
            candidate_metrics["ohlc"][field]["mae_points"]
            / baseline_metrics["ohlc"][field]["mae_points"]
            - 1
            if baseline_metrics["ohlc"][field]["mae_points"] > 0
            else (0.0 if candidate_metrics["ohlc"][field]["mae_points"] == 0 else None)
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
    paired_coverage = comparison["paired_coverage"]
    requirements = {
        "close_buffer": close_improvement >= float(config["promotion_close_improvement"]),
        "fold_majority": sum(value is not None and value > 0 for value in fold_improvements)
        > len(fold_improvements) / 2,
        "bootstrap_lower_nonnegative": ci is not None and ci[0] >= 0,
        "ohl_noninferiority": all(
            value is not None and value <= 0.05 for value in ohl_degradation.values()
        ),
        "interval_coverage": all(
            value is not None and 0.75 <= value <= 0.85 for value in coverage.values()
        ),
        "interval_score_noninferior": interval_score_noninferior,
        "forecast_coverage": paired_coverage >= float(config["promotion_min_coverage"]),
    }
    return {
        **comparison,
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


def joint_development_acceptance(instrument_gates: dict[str, dict]) -> dict:
    """Require both market gates; expose their worst close improvement for selection."""
    if set(instrument_gates) != {"TAIEX", "TX"}:
        raise ValueError("joint development gate requires TAIEX and TX")
    improvements = {
        instrument: gate.get("close_relative_improvement")
        for instrument, gate in instrument_gates.items()
    }
    comparable = all(value is not None for value in improvements.values())
    both_pass = all(gate.get("status") == "PASS" for gate in instrument_gates.values())
    return {
        "status": "PASS" if both_pass else "FAIL",
        "close_relative_improvement": (
            min(float(value) for value in improvements.values()) if comparable else None
        ),
        "instrument_close_relative_improvement": improvements,
        "instrument_gates": instrument_gates,
        "requirements": {"both_markets_pass": both_pass},
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
    best_passing = None
    best_improvement = -math.inf
    no_gain = 0
    evaluated = []
    stopped_early = False
    for result in sorted(results, key=lambda item: item["identity"]["candidate_index"]):
        gate = result.get("development_gate")
        if not isinstance(gate, dict) or "close_relative_improvement" not in gate:
            raise ValueError("candidate result lacks a development gate")
        raw_improvement = gate["close_relative_improvement"]
        improvement = float(raw_improvement) if raw_improvement is not None else -math.inf
        if gate.get("status") == "PASS" and not math.isfinite(improvement):
            raise ValueError("passing gate requires a finite improvement")
        entry = {
            "candidate_index": result["identity"]["candidate_index"],
            "status": result.get("status"),
            "development_gate": gate.get("status"),
            "close_relative_improvement": raw_improvement,
            "result_hash": digest(result),
        }
        evaluated.append(entry)
        # Promotion requires every gate, even when a rejected candidate has
        # a larger close-only improvement. Keep the search stop rule separate.
        if gate.get("status") == "PASS" and (
            best_passing is None
            or improvement > best_passing["close_relative_improvement"]
        ):
            best_passing = entry
        if best is None or improvement > best_improvement + 1e-12:
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
    best_research = best
    best = best_passing if best_passing is not None else best_research
    passed = best_passing is not None
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
        "best_close_only_research_candidate": best_research,
        "evaluated": evaluated,
        "stopped_after_no_gain": stopped_early,
        "max_no_gain_rounds": max_no_gain_rounds,
        "final_test_opened": False,
        "old_final_reused_as_independent_test": False,
        "production_candidate": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@bounded_head_run
def run_output_head_training(
    dataset_dir: Path,
    config_path: Path,
    cache_dir: Path,
    output_root: Path,
    instrument: str,
    candidate_index: int = 0,
) -> Path:
    """Train one market-specific or joint TimesFM head without opening a final test."""
    from safetensors.torch import save_file

    if instrument not in {"TAIEX", "TX", "JOINT"}:
        raise ValueError("instrument must be TAIEX, TX, or JOINT")
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
    if instrument == "JOINT":
        feature_names = {
            market: [f"{market.lower()}__{name}" for name in data.research_feature_names()]
            for market in ("TAIEX", "TX")
        }
    else:
        feature_names = data.research_feature_names()
    identity = {
        "name": "custom_timesfm3_output_head_training",
        "instrument": instrument,
        "dataset_id": dataset_id,
        "config_hash": digest(config),
        "code_version": _git_state(config_path.parent.parent.parent),
        "timesfm_code_commit": CODE_COMMIT,
        "timesfm_weight_revision": WEIGHT_REVISION,
        "train_end": str(training_days[::stride][-1]),
        "early_stop_split": [str(early_stop_days[0]), str(early_stop_days[-1])],
        "sample_policy": "classified_history_exclusions_v2",
        "feature_names": feature_names,
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
    train_samples, training_audit = audited_head_samples(
        data, instrument, training_days[::stride], context
    )
    early_stop_samples, early_stop_audit = audited_head_samples(
        data, instrument, early_stop_days, context
    )
    sample_audit = _persist_sample_audit(
        experiment_dir / "sample-audit.json", identity, training_audit, early_stop_audit
    )
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
            deadline=budget_deadline(started + float(config["budget_cpu_seconds"])),
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
            "sample_audit": sample_audit,
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


def _joint_prediction_rows(model, sample, mean, scale) -> list[dict]:
    predictions = predict_joint_head(model, sample, mean, scale)
    if not isinstance(sample.contract, dict) or not isinstance(sample.actual, dict):
        raise TypeError("joint head sample lacks market metadata")
    rows = []
    for instrument in ("TAIEX", "TX"):
        point, interval, quantiles = predictions[instrument]
        rows.append({
            "date": str(sample.day),
            "instrument": instrument,
            "contract": sample.contract[instrument],
            "status": "OK",
            "previous_close": sample.previous_close[0 if instrument == "TAIEX" else 1],
            "predicted": point.model_dump(),
            "actual": sample.actual[instrument],
            "interval": interval,
            "interval_method": "market-block rank-aligned TimesFM transformed quantiles",
            "raw_transformed_quantiles": quantiles.tolist(),
        })
    return rows


@bounded_head_run
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

    if instrument not in {"TAIEX", "TX", "JOINT"}:
        raise ValueError("instrument must be TAIEX, TX, or JOINT")
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
        "sample_policy": "classified_history_exclusions_v2",
        "evaluation_policy": "paired_dates_failures_preserved_v2",
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
        "feature_names": (
            {
                market: [f"{market.lower()}__{name}" for name in data.research_feature_names()]
                for market in ("TAIEX", "TX")
            }
            if instrument == "JOINT"
            else data.research_feature_names()
        ),
        "candidate_index": candidate_index,
        "spec": spec.__dict__,
    }
    experiment_dir = output_root / "timesfm3-output-head-walk-forward" / digest(identity)[:20]
    result_path = experiment_dir / "result.json"
    if result_path.exists():
        return result_path
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = budget_deadline(started + float(config["budget_cpu_seconds"]))
    forecaster = load_pinned_forecaster(cache_dir, local_files_only=True)
    model = forecaster.model
    original_head = {
        name: value.detach().cpu().clone()
        for name, value in model.output_head.state_dict().items()
    }
    fold_results = []
    instruments = ("TAIEX", "TX") if instrument == "JOINT" else (instrument,)
    candidate_rows = {market: [] for market in instruments}
    baseline_rows = {market: [] for market in instruments}
    fold_improvements = {market: [] for market in instruments}
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
                saved_candidates = saved["candidate_rows"]
                saved_baselines = saved["baseline_rows"]
                if instrument == "JOINT":
                    for market in instruments:
                        candidate_rows[market].extend(saved_candidates[market])
                        baseline_rows[market].extend(saved_baselines[market])
                        fold_improvements[market].append(
                            reused_summary["instrument_results"][market][
                                "close_relative_improvement"
                            ]
                        )
                else:
                    candidate_rows[instrument].extend(saved_candidates)
                    baseline_rows[instrument].extend(saved_baselines)
                    fold_improvements[instrument].append(
                        reused_summary["close_relative_improvement"]
                    )
                continue
            model.output_head.load_state_dict(original_head)
            early_stop_sessions = int(config["early_stop_sessions"])
            usable_train_days = train_days[spec.context:]
            training_window = config.get("training_window_sessions")
            if training_window is not None:
                training_window = int(training_window)
                if training_window <= early_stop_sessions:
                    raise ValueError(
                        "training_window_sessions must exceed early_stop_sessions"
                    )
                usable_train_days = usable_train_days[-training_window:]
            if len(usable_train_days) <= early_stop_sessions:
                raise ValueError("fold lacks output-head training samples")
            stride = int(config["training_stride"])
            if stride < 1:
                raise ValueError("training_stride must be positive")
            training_days = usable_train_days[:-early_stop_sessions:stride]
            early_stop_days = usable_train_days[-early_stop_sessions:]
            training_samples, training_audit = audited_head_samples(
                data, instrument, training_days, spec.context
            )
            early_stop_samples, early_stop_audit = audited_head_samples(
                data, instrument, early_stop_days, spec.context
            )
            sample_audit = _persist_sample_audit(
                experiment_dir / f"fold-{fold_index}-sample-audit.json",
                {**identity, "fold": fold_index}, training_audit, early_stop_audit,
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
            fold_candidate = {market: [] for market in instruments}
            for validation_day in validation_days:
                try:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("validation budget exhausted")
                    if instrument == "JOINT":
                        sample = build_joint_head_samples(
                            data, [validation_day], spec.context
                        )[0]
                        for row in _joint_prediction_rows(model, sample, mean, scale):
                            fold_candidate[row["instrument"]].append(row)
                    else:
                        sample = build_head_samples(
                            data, instrument, [validation_day], spec.context
                        )[0]
                        fold_candidate[instrument].append(
                            _prediction_row(model, sample, mean, scale, instrument)
                        )
                except Exception as exc:  # noqa: BLE001 - preserve every expected day
                    for market in instruments:
                        try:
                            contract = data.contract(market, validation_day)
                        except Exception:  # noqa: BLE001 - failure row must survive metadata errors
                            contract = None
                        fold_candidate[market].append({
                            "date": str(validation_day),
                            "instrument": market,
                            "contract": contract,
                            "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                        })

            expected_dates = [str(day) for day in validation_days]
            validation_audit_path = experiment_dir / (
                f"fold-{fold_index}-validation-{digest(fold_candidate)[:20]}.json"
            )
            if not validation_audit_path.exists():
                write_once(validation_audit_path, fold_candidate)
            instrument_results = {}
            fold_baseline = {}
            for market in instruments:
                baseline_validation = {}
                for baseline_model in config["stable_baselines"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("baseline evaluation budget exhausted")
                    rows = walk_forward(
                        data,
                        market,
                        baseline_model,
                        "gbud",
                        validation_days[0],
                        validation_days[-1],
                        config,
                    )
                    baseline_validation[f"{market}/gbud/{baseline_model}"] = {
                        "rows": rows,
                        "metrics": metrics(rows, config["neutral_bps"]),
                    }
                selected_baseline, selection_audit = _select_baseline(
                    baseline_validation,
                    market,
                    expected_dates,
                    float(config["promotion_min_coverage"]),
                    float(config["neutral_bps"]),
                    config["baseline_stability"],
                )
                baseline_key = (
                    f"{market}/{selected_baseline['representation']}/"
                    f"{selected_baseline['model']}"
                )
                fold_baseline[market] = baseline_validation[baseline_key]["rows"]
                comparison = paired_validation_metrics(
                    fold_candidate[market], fold_baseline[market], config["neutral_bps"]
                )
                improvement = comparison["close_relative_improvement"]
                fold_improvements[market].append(improvement)
                candidate_rows[market].extend(fold_candidate[market])
                baseline_rows[market].extend(fold_baseline[market])
                instrument_results[market] = {
                    "selected_baseline": selected_baseline,
                    "baseline_selection_audit": selection_audit,
                    **comparison,
                }
            fold_summary = {
                "fold": fold_index,
                "train_split": [str(train_days[0]), str(train_days[-1])],
                "validation_split": [str(validation_days[0]), str(validation_days[-1])],
                "sample_audit": sample_audit,
                "training": training,
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                },
                "close_relative_improvement": (
                    min(result["close_relative_improvement"]
                        for result in instrument_results.values())
                    if all(result["close_relative_improvement"] is not None
                           for result in instrument_results.values()) else None
                ),
                "reused": False,
            }
            if instrument == "JOINT":
                fold_summary["instrument_results"] = instrument_results
            else:
                fold_summary.update(instrument_results[instrument])
            write_once(fold_result_path, {
                "identity_hash": digest(identity),
                "fold": fold_index,
                "summary": fold_summary,
                "candidate_rows": (
                    fold_candidate if instrument == "JOINT" else fold_candidate[instrument]
                ),
                "baseline_rows": (
                    fold_baseline if instrument == "JOINT" else fold_baseline[instrument]
                ),
            })
            fold_results.append(fold_summary)
        instrument_gates = {
            market: development_acceptance(
                candidate_rows[market],
                baseline_rows[market],
                config,
                fold_improvements=fold_improvements[market],
            )
            for market in instruments
        }
        gate = (
            joint_development_acceptance(instrument_gates)
            if instrument == "JOINT"
            else instrument_gates[instrument]
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
