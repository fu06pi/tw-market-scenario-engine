"""Bounded validation-only experiments. Final-test dates never enter model inputs/metrics."""

from __future__ import annotations

import copy
import importlib.metadata as md
import json
import math
import resource
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from . import baselines, night
from .data import (
    FIELDS,
    canonical_hash,
    indices,
    load_dataset,
    metrics,
    reconstruct,
    sample_arrays,
    write_new,
)
from .model import CODE_COMMIT, MODEL, REVISION, Forecaster


def acceptable(scores: dict, reference: dict, close_ratio: float = 1.0) -> bool:
    return scores["close"]["mae_points"] <= close_ratio * reference["close"]["mae_points"] and all(
        scores[f]["mae_points"] <= 1.05 * reference[f]["mae_points"] for f in FIELDS[:3]
    )


def extra_metrics(prediction, actual, previous):
    predicted_bps = (prediction[:, 3] / previous - 1) * 10000
    actual_bps = (actual[:, 3] / previous - 1) * 10000

    def classes(x):
        return np.where(x > 5, 2, np.where(x < -5, 0, 1))

    pc, ac = classes(predicted_bps), classes(actual_bps)
    confusion = np.zeros((3, 3), dtype=int)
    for a, p in zip(ac, pc, strict=True):
        confusion[a, p] += 1
    return {
        "range_mae_points": float(
            np.abs((prediction[:, 1] - prediction[:, 2]) - (actual[:, 1] - actual[:, 2])).mean()
        ),
        "close_direction_confusion_actual_rows_predicted_columns": confusion.tolist(),
        "class_order": ["down", "flat", "up"],
        "neutral_threshold_bps": 5,
        "actual_class_counts": np.bincount(ac, minlength=3).tolist(),
        "predicted_class_counts": np.bincount(pc, minlength=3).tolist(),
        "direction_accuracy": float(np.mean(pc == ac)),
    }


def run_market(root: Path, folder: Path, market: str, dataset: dict, cfg: dict, model: Forecaster):
    started = time.monotonic()
    raw, bars = load_dataset(root / dataset["file"])
    # Remove sealed final-test bars before even computing transforms/windows/normalizers.
    bars = [bar for bar in bars if str(bar.trading_date) <= cfg["validation_end"]]
    context = cfg["context"]
    training = indices(bars, context, "1900-01-01", cfg["train_end"])
    validation = indices(bars, context, cfg["validation_start"], cfg["validation_end"])
    if len(training) < 16 or len(validation) < 20:
        raise ValueError("insufficient train/validation coverage")
    target_dates = [str(bars[i + 1].trading_date) for i in validation]
    actual = np.array([[getattr(bars[i + 1], f) for f in FIELDS] for i in validation])
    previous = np.array([bars[i].close for i in validation])
    summary = {
        "market": market,
        "contract": bars[0].contract,
        "dataset_checksum": raw["checksum"],
        "train_count": len(training),
        "validation_count": len(validation),
        "validation_dates": target_dates,
        "final_test_evaluated": False,
        "baseline_results": {},
        "baseline_failures": {},
        "candidates": [],
        "scope": "retrospective; publication availability and independent calendar NOT verified",
    }
    night_series = None
    night_source_hash = None
    if cfg.get("night_covariates"):
        night_data = json.loads((root / cfg["night_covariates"]).read_text())
        night_source_hash = night_data["checksum"]
        if (
            canonical_hash({k: v for k, v in night_data.items() if k != "checksum"})
            != night_source_hash
        ):
            raise ValueError("night source checksum mismatch")
        night_series = night_data["series"]
        if any(day > cfg["validation_end"] for day in night_series):
            raise ValueError("night source crosses sealed final-test boundary")
        summary["night_source_checksum"] = night_source_hash
        summary["known_night_validation_rows"] = sum(
            str(bars[i + 1].trading_date) in night_series for i in validation
        )
    baseline_arrays = {}
    for name in baselines.BASELINES:
        forecasts = []
        failures = []
        for i in validation:
            try:
                forecasts.append((i, baselines.forecast(bars, [i], name)[0]))
            except ValueError as error:
                failures.append({"date": str(bars[i + 1].trading_date), "reason": str(error)})
        if failures:
            summary["baseline_failures"][name] = {
                "failures": failures,
                "coverage": len(forecasts) / len(validation),
            }
        else:
            values = np.stack([value for _, value in forecasts])
            baseline_arrays[name] = values
            summary["baseline_results"][name] = metrics(values, actual, previous)
    if night_series is not None:
        values = night.regression(bars, training, validation, night_series)
        baseline_arrays["night_regression_train_only"] = values
        summary["baseline_results"]["night_regression_train_only"] = metrics(
            values, actual, previous
        )
    if not baseline_arrays:
        raise ValueError("no complete baseline")
    strongest = min(
        baseline_arrays, key=lambda name: summary["baseline_results"][name]["close"]["mae_points"]
    )
    reference = summary["baseline_results"][strongest]
    summary["locked_baseline"] = strongest
    summary["locked_baseline_selection"] = (
        "validation close MAE; only complete forecasts; incomplete baselines reported"
    )
    write_new(folder / f"{market}-baseline-lock.json", summary)
    print(
        json.dumps(
            {
                "market": market,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "strongest_baseline": strongest,
                "baseline_close_mae": reference["close"]["mae_points"],
                "baseline_failures": list(summary["baseline_failures"]),
            }
        ),
        flush=True,
    )
    best_score = math.inf
    best_name = None
    best_predictions = None
    # Fixed raw model + output-head trials. No candidate selection uses the sealed final test.
    no_improvement = 0
    for trial_number, trial in enumerate(cfg["trials"]):
        if time.monotonic() - started >= cfg["max_seconds"]:
            summary["stopped"] = "time_budget"
            break
        model.reset()
        representation = trial["representation"]
        x, y, _, _ = sample_arrays(bars, representation, context, training)
        vx, _, vprevious, vactual = sample_arrays(bars, representation, context, validation)
        train_cov = (
            night.covariates(bars, training, context, night_series)
            if night_series is not None
            else None
        )
        valid_cov = (
            night.covariates(bars, validation, context, night_series)
            if night_series is not None
            else None
        )
        parity = model.verify_paths(vx, covariates=valid_cov)
        initial = model.predict(vx, cfg["batch_size"], covariates=valid_cov)
        initial_prices = reconstruct(initial[:, :, 4], vprevious, representation)
        initial_scores = metrics(initial_prices, vactual, vprevious)
        scores_history = [{"epoch": 0, "metrics": initial_scores}]
        best_trial_head = copy.deepcopy(model.model.output_head.state_dict())
        best_trial_predictions = initial_prices.copy()
        best_trial_scores = initial_scores
        best_epoch = 0

        def selection(scores):
            return (
                scores["close"]["mae_points"]
                if acceptable(scores, reference, close_ratio=math.inf)
                else math.inf
            )

        epoch_score = selection(initial_scores)
        for p in model.model.output_head.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.model.output_head.parameters(), lr=trial["learning_rate"], weight_decay=0.0
        )
        channel_scale = torch.tensor(np.maximum(np.std(y, axis=0), 1e-5))
        parameter_start = [p.detach().clone() for p in model.model.output_head.parameters()]
        steps = 0
        stale = 0
        max_gradient = 0.0
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.model.train()
            loss_values = []
            for start in range(0, len(training), cfg["batch_size"]):
                if time.monotonic() - started >= cfg["max_seconds"]:
                    break
                # chronological batches; no temporal shuffle or target overlap into context
                b = slice(start, min(start + cfg["batch_size"], len(training)))
                logits = model.logits(
                    x[b], grad=True, covariates=None if train_cov is None else train_cov[b]
                )
                error = torch.tensor(y[b])[:, :, None] - logits
                loss = (
                    torch.maximum(model.quantiles * error, (model.quantiles - 1) * error)
                    / channel_scale[None, :, None]
                ).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite train loss")
                optimizer.zero_grad()
                loss.backward()
                parameters = list(model.model.output_head.parameters())
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise ValueError("missing/nonfinite gradients")
                gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
                steps += 1
                loss_values.append(float(loss.detach()))
            values = model.predict(vx, cfg["batch_size"], covariates=valid_cov)
            predictions = reconstruct(values[:, :, 4], vprevious, representation)
            scores = metrics(predictions, vactual, vprevious)
            scores_history.append(
                {
                    "epoch": epoch,
                    "train_normalized_pinball": float(np.mean(loss_values))
                    if loss_values
                    else None,
                    "metrics": scores,
                }
            )
            score = selection(scores)
            if score < epoch_score:
                epoch_score = score
                best_epoch = epoch
                best_trial_scores = scores
                best_trial_head = copy.deepcopy(model.model.output_head.state_dict())
                best_trial_predictions = predictions.copy()
                stale = 0
            else:
                stale += 1
            if (
                stale >= cfg["early_stop_epochs"]
                or time.monotonic() - started >= cfg["max_seconds"]
            ):
                break
        delta = math.sqrt(
            sum(
                float((p.detach() - old).square().sum())
                for p, old in zip(
                    model.model.output_head.parameters(), parameter_start, strict=True
                )
            )
        )
        if max_gradient <= 0 or delta <= 0:
            raise ValueError("no real parameter update")
        name = f"{market}-trial{trial_number}-{representation}"
        checkpoint = folder / f"{name}.pt"
        torch.save(
            {
                "weight_revision": REVISION,
                "model": MODEL,
                "head": best_trial_head,
                "last_head": model.model.output_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "representation": representation,
                "context": context,
                "dataset_checksum": raw["checksum"],
                "best_epoch": best_epoch,
                "scope": "head_only",
                "config_hash": canonical_hash(cfg),
                "night_source_checksum": night_source_hash,
            },
            checkpoint,
        )
        model.load_head(checkpoint)
        restored = reconstruct(
            model.predict(vx, cfg["batch_size"], covariates=valid_cov)[:, :, 4],
            vprevious,
            representation,
        )
        if not np.array_equal(restored, best_trial_predictions):
            raise ValueError("checkpoint inference readback mismatch")
        result = {
            "name": name,
            "representation": representation,
            "learning_rate": trial["learning_rate"],
            "inference_parity": parity,
            "best_epoch": best_epoch,
            "epochs": scores_history,
            "steps": steps,
            "max_gradient": max_gradient,
            "last_parameter_delta_l2": delta,
            "checkpoint": str(checkpoint.relative_to(root)),
            "checkpoint_readback": "exact_prediction_match",
            "best_validation_metrics": best_trial_scores,
            "selection_ohl_constraints_pass": acceptable(
                best_trial_scores, reference, close_ratio=math.inf
            ),
            "validation_point_gate_pass": acceptable(
                best_trial_scores, reference, close_ratio=0.95
            ),
            "extra_validation_metrics": extra_metrics(best_trial_predictions, vactual, vprevious),
            "raw_transformed_quantiles_not_calibrated": True,
        }
        write_new(folder / f"{name}.json", result)
        summary["candidates"].append(result)
        score = selection(best_trial_scores)
        if score < best_score:
            best_score = score
            best_name = name
            best_predictions = best_trial_predictions
            no_improvement = 0
        else:
            no_improvement += 1
        print(
            json.dumps(
                {
                    "market": market,
                    "trial": trial_number,
                    "repr": representation,
                    "best_epoch": best_epoch,
                    "close_mae": best_trial_scores["close"]["mae_points"],
                    "ohl_constraints": result["selection_ohl_constraints_pass"],
                    "point_gate": result["validation_point_gate_pass"],
                }
            ),
            flush=True,
        )
        if no_improvement >= cfg["max_non_improving_rounds"]:
            summary["stopped"] = "non_improving_rounds"
            break
    summary["best_eligible_timesfm"] = best_name
    summary["default_candidate"] = strongest
    if best_predictions is not None:
        best_metrics = metrics(best_predictions, actual, previous)
        summary["best_eligible_metrics"] = best_metrics
        summary["validation_point_gate_pass"] = acceptable(best_metrics, reference, 0.95)
        if best_score < reference["close"]["mae_points"]:
            summary["default_candidate"] = best_name
    else:
        summary["validation_point_gate_pass"] = False
    summary["acceptance_passed"] = False
    summary["acceptance_blockers"] = [
        "final test not evaluated",
        "historical availability not verified",
        "independent trading calendar not verified",
        "80% OHLC coverage not calibrated",
        "night-regression and ARIMAX not evaluated",
        "live feed stale",
        "V0 integration not verified",
    ]
    if summary["baseline_failures"]:
        summary["acceptance_blockers"].append("incomplete baseline failures")
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_new(folder / f"{market}-summary.json", summary)
    return summary


def run(root: Path, config: Path, audit: Path, folder: Path):
    cfg = json.loads(config.read_text())
    data_audit = json.loads(audit.read_text())
    if (
        not cfg["train_end"]
        < cfg["validation_start"]
        <= cfg["validation_end"]
        < cfg["final_test_start"]
    ):
        raise ValueError("invalid chronological split")
    if cfg["context"] % 32 or cfg["context"] < 32:
        raise ValueError("invalid patch context")
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "config.json", cfg)
    write_new(folder / "sources.json", data_audit)
    model = Forecaster()
    report = {
        "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "model": MODEL,
        "weight_revision": REVISION,
        "official_code_commit": CODE_COMMIT,
        "packages": {
            p: md.version(p) for p in ["timesfm", "torch", "numpy", "pandas", "statsmodels"]
        },
        "config_hash": canonical_hash(cfg),
        "night_source_checksum": json.loads((root / cfg["night_covariates"]).read_text())[
            "checksum"
        ]
        if cfg.get("night_covariates")
        else None,
        "markets": {},
        "acceptance_passed": False,
        "checkpoint_license": "TimesFM non-commercial v1.0; no production or distribution",
    }
    for market, data in data_audit["markets"].items():
        try:
            report["markets"][market] = run_market(root, folder, market, data, cfg, model)
        except Exception as error:
            write_new(
                folder / f"{market}-failure.json",
                {"stage": "experiment", "type": type(error).__name__},
            )
            raise
    write_new(folder / "summary.json", report)
    return report
