from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from datetime import date
from pathlib import Path

import numpy as np

from .backtest import load_dataset
from .contracts import digest, encode, write_once
from .timesfm3 import (
    CODE_COMMIT,
    WEIGHT_REPO,
    WEIGHT_REVISION,
    load_pinned_forecaster,
)


def _training_inputs(model, history: np.ndarray):
    """Build the pinned TimesFM 3 forward inputs for a one-step target."""
    import torch

    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 4:
        raise ValueError("gradient smoke history must have shape (time, 4)")
    if history.shape[0] % model.input_patch_len:
        raise ValueError("gradient smoke history must align to input patches")
    if not np.all(np.isfinite(history)):
        raise ValueError("gradient smoke history must be finite")

    context = torch.tensor(history.T[None], dtype=torch.float32)
    batch, variates, context_length = context.shape
    num_context_patches = context_length // model.input_patch_len
    # With the pinned stitching configuration, a one-step horizon needs one
    # output patch (two input-sized masked patches).
    padded_horizon = model.output_patch_len
    horizon_values = torch.zeros((batch, variates, padded_horizon))
    horizon_masks = torch.ones(
        (batch, variates, padded_horizon), dtype=torch.bool
    )
    all_values = torch.cat([context, horizon_values], dim=-1)
    all_masks = torch.cat(
        [torch.zeros_like(context, dtype=torch.bool), horizon_masks], dim=-1
    )
    values = all_values.reshape(batch, variates, -1, model.input_patch_len)
    masks = all_masks.reshape(batch, variates, -1, model.input_patch_len)
    patch_is_target = torch.ones(
        (batch, variates, values.shape[2]), dtype=torch.bool
    )
    horizon_cpm_mask = torch.zeros(
        (batch, values.shape[2]), dtype=torch.bool
    )
    horizon_cpm_mask[:, num_context_patches:] = True
    inputs = {
        "values": values,
        "masks": masks,
        "patch_is_target": patch_is_target,
    }
    return inputs, horizon_cpm_mask, num_context_patches


def _one_step_median(model, inputs, horizon_cpm_mask, num_context_patches):
    outputs = model.forward(inputs, patch_cpm_mask=horizon_cpm_mask)
    median_index = model.num_quantiles // 2
    return outputs["logits"][
        :, :, num_context_patches - 1, 0, median_index
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_output_head_gradient_smoke(
    dataset_dir: Path,
    config_path: Path,
    cache_dir: Path,
    output_root: Path,
) -> Path:
    """Run one output-head-only SGD step on one pre-validation TAIEX sample."""
    import torch
    from safetensors.torch import load_file, save_file

    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data, dataset_id = load_dataset(dataset_dir)
    validation_start = date.fromisoformat(config["validation_start"])
    sample_date = data.calendar.previous(validation_start)
    if sample_date >= validation_start:
        raise ValueError("gradient smoke sample must precede validation")

    history_window = 64
    learning_rate = 1e-5
    history, _, history_dates = data.history(
        "TAIEX",
        sample_date,
        "gbud",
        history_window,
        history_window,
        False,
    )
    reference = data.reference("TAIEX", sample_date).close
    target = encode(data.actual("TAIEX", sample_date), reference, "gbud").astype(
        np.float32
    )
    identity = {
        "name": "custom_output_head_gradient_smoke",
        "dataset_id": dataset_id,
        "config_hash": digest(config),
        "code_commit": CODE_COMMIT,
        "weight_revision": WEIGHT_REVISION,
        "sample_date": sample_date.isoformat(),
        "history_dates_hash": digest([str(day) for day in history_dates]),
        "history_window": history_window,
        "representation": "gbud",
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "steps": 1,
        "seed": 0,
    }
    experiment_id = digest(identity)[:20]
    experiment_dir = output_root / "timesfm3-output-head-smoke" / experiment_id
    result_path = experiment_dir / "result.json"
    checkpoint_path = experiment_dir / "output_head.safetensors"
    if result_path.exists():
        return result_path

    torch.manual_seed(0)
    forecaster = load_pinned_forecaster(cache_dir, local_files_only=True)
    model = forecaster.model
    model.eval()
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    original_head = {
        name: value.detach().cpu().clone()
        for name, value in model.output_head.state_dict().items()
    }
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.output_head.parameters():
            parameter.requires_grad_(True)
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_names or any(
            not name.startswith("output_head.") for name in trainable_names
        ):
            raise RuntimeError("gradient smoke must train only output_head parameters")

        inputs, horizon_cpm_mask, num_context_patches = _training_inputs(
            model, history
        )
        target_tensor = torch.tensor(target, dtype=torch.float32)[None]
        with torch.no_grad():
            forecast_before = _one_step_median(
                model, inputs, horizon_cpm_mask, num_context_patches
            ).clone()

        optimizer = torch.optim.SGD(
            model.output_head.parameters(), lr=learning_rate
        )
        optimizer.zero_grad(set_to_none=True)
        forecast = _one_step_median(
            model, inputs, horizon_cpm_mask, num_context_patches
        )
        loss_before = torch.nn.functional.smooth_l1_loss(
            forecast, target_tensor
        )
        loss_before.backward()
        gradient_norm = torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in model.output_head.parameters()
                if parameter.grad is not None
            )
        )
        optimizer.step()

        with torch.no_grad():
            forecast_after = _one_step_median(
                model, inputs, horizon_cpm_mask, num_context_patches
            ).clone()
            loss_after = torch.nn.functional.smooth_l1_loss(
                forecast_after, target_tensor
            )
        updated_head = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.output_head.state_dict().items()
        }
        parameter_delta = torch.sqrt(
            sum(
                ((updated_head[name] - original_head[name]) ** 2).sum()
                for name in original_head
            )
        )
        forecast_delta = torch.linalg.vector_norm(
            forecast_after - forecast_before
        )

        experiment_dir.mkdir(parents=True, exist_ok=True)
        save_file(updated_head, checkpoint_path)
        model.output_head.load_state_dict(original_head)
        model.output_head.load_state_dict(load_file(checkpoint_path))
        with torch.no_grad():
            forecast_reloaded = _one_step_median(
                model, inputs, horizon_cpm_mask, num_context_patches
            )
        reload_max_abs = torch.max(
            torch.abs(forecast_reloaded - forecast_after)
        )

        evidence = {
            "finite_loss": math.isfinite(float(loss_before.detach())),
            "nonzero_output_head_gradient": float(gradient_norm) > 0,
            "output_head_parameter_delta": float(parameter_delta) > 0,
            "forecast_delta": float(forecast_delta) > 0,
            "save_reload_equivalence": float(reload_max_abs) <= 1e-7,
        }
        result = {
            "schema_version": "timesfm3-custom-output-head-gradient-smoke-v1",
            "status": "PASS" if all(evidence.values()) else "FAIL",
            "name": "custom_output_head_gradient_smoke",
            "official_timesfm3_finetuning": False,
            "training_scope": "one pre-validation sample; output head only; one SGD step",
            "production_candidate": False,
            "changes_model_selection": False,
            "final_test_opened": False,
            "identity": identity,
            "weight_repo": WEIGHT_REPO,
            "sample": {
                "instrument": "TAIEX",
                "sample_date": sample_date.isoformat(),
                "history_start": str(history_dates[0]),
                "history_end": str(history_dates[-1]),
                "history_rows": len(history_dates),
                "split": "train",
            },
            "trainable_parameter_names": trainable_names,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.output_head.parameters()
            ),
            "loss_before": float(loss_before.detach()),
            "loss_after": float(loss_after),
            "gradient_norm_l2": float(gradient_norm),
            "parameter_delta_l2": float(parameter_delta),
            "forecast_delta_l2": float(forecast_delta),
            "save_reload_max_abs": float(reload_max_abs),
            "evidence": evidence,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "scope": "output_head_only_local_research_artifact",
            },
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "limitations": [
                "This is a custom PyTorch gradient smoke, not an official TimesFM 3 fine-tuning API.",
                "One train-period sample and one optimizer step do not establish forecast improvement.",
                "The checkpoint is not eligible for daily output or final-test reevaluation.",
            ],
        }
        write_once(result_path, result)
        return result_path
    finally:
        model.output_head.load_state_dict(original_head)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
        model.eval()
