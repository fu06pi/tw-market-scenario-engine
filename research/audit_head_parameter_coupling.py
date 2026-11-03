"""Audit shared-head coupling using synthetic tensors, never pretrained weights.

Run from the repository root with:
  .venv/bin/python research/audit_head_parameter_coupling.py

This is a structural counterexample to inferring parameter independence from
zero partial derivatives with respect to g/b logits. It is not a new model
candidate, a real-data training experiment, or evidence of accuracy improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from tw_scenario.quant import timesfm_head_training as training
from tw_scenario.quant.timesfm3 import CODE_COMMIT, installed_provenance

TARGETS = ("g", "b", "log_u", "log_d")
QUANTILES = [index / 10 for index in range(1, 10)]
MEDIAN = 4
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record(path: Path, tokens: list[str]) -> dict:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    for token in tokens:
        if token not in source:
            raise AssertionError(f"required audited source structure missing: {token}")
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": _sha256(path),
        "evidence": [
            {"line": index + 1, "text": line.strip()}
            for index, line in enumerate(lines)
            if any(token in line for token in tokens)
        ],
    }


def _synthetic_counterexample(optimizer_name: str) -> dict:
    # Same shared Linear and reverse-RevIN layout as the pinned head. Small
    # hidden dimension only; no official model instance or data is loaded.
    head = torch.nn.Linear(3, 64 * len(QUANTILES), dtype=torch.float64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    model = SimpleNamespace(
        output_head=head, output_patch_len=64,
        quantiles=QUANTILES, value_clip=1e20,
    )
    hidden = torch.tensor(
        [[[1.0, 0.2, 0.3], [0.5, 1.0, 0.2],
          [1.0, 0.3, 0.7], [0.8, 0.6, 1.0]]],
        dtype=torch.float64,
    )
    means = torch.tensor([[0.001, -0.002, -6.0, -6.5]], dtype=torch.float64)
    scales = torch.tensor([[0.01, 0.02, 0.8, 1.1]], dtype=torch.float64)
    features = training.FrozenHeadFeatures(
        hidden=hidden, revin_mean=means, revin_std=scales,
        total_variates=6, total_patches=4, selected_patch=1, target_count=4,
    )
    target = torch.tensor([[0.001, -0.002, -5.5, -6.0]], dtype=torch.float64)
    spec = training.HeadTrainingSpec(
        context=64, learning_rate=0.01, weight_decay=0.0,
        point_weight=0.0, close_weight=0.0,
        quantile_weight=0.0, crossing_weight=0.0,
        wick_weight=0.25, loss_normalization="none",
    )
    optimizer_class = {"SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_class(head.parameters(), lr=0.01, weight_decay=0.0)
    before = training._one_step_logits_from_features(model, features)
    before.retain_grad()
    loss = training.output_head_loss(before, target, QUANTILES, spec)
    loss.backward()
    output_grad = before.grad.detach().clone()
    weight_grad = head.weight.grad.detach().clone()
    bias_grad = head.bias.grad.detach().clone()
    assert torch.count_nonzero(output_grad[:, :2]) == 0
    assert torch.count_nonzero(output_grad[:, 2:, MEDIAN]) == 2
    active_rows = torch.nonzero(
        weight_grad.abs().sum(dim=1) + bias_grad.abs(), as_tuple=True
    )[0].tolist()
    assert active_rows == [MEDIAN]
    optimizer.step()
    after = training._one_step_logits_from_features(model, features).detach()
    delta = (after - before.detach())[0, :, MEDIAN]
    assert all(float(abs(delta[index])) > 1e-10 for index in (0, 1))
    close_before = torch.exp(before.detach()[0, 0, MEDIAN] + before.detach()[0, 1, MEDIAN])
    close_after = torch.exp(after[0, 0, MEDIAN] + after[0, 1, MEDIAN])
    sgd_formula_error = None
    if optimizer_name == "SGD":
        # Change for variate i = -lr * sigma_i * (h_i dot grad_w + grad_bias).
        expected = -0.01 * scales[0] * (
            hidden[0] @ weight_grad[MEDIAN] + bias_grad[MEDIAN]
        )
        sgd_formula_error = float(torch.max(torch.abs(delta - expected)))
        assert sgd_formula_error < 1e-12
    return {
        "fixture": "synthetic_fixed_hidden_no_pretrained_weights_no_market_data",
        "optimizer": optimizer_name,
        "optimizer_steps": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "spec": asdict(spec),
        "shared_head_weight_shape": list(head.weight.shape),
        "hidden_by_target": hidden[0].tolist(),
        "revin_mean_by_target": means[0].tolist(),
        "revin_std_by_target": scales[0].tolist(),
        "loss": float(loss.detach()),
        "absolute_logit_gradient_l1_by_target": dict(zip(
            TARGETS, output_grad.abs().sum(dim=(0, 2)).tolist(), strict=True
        )),
        "nonzero_parameter_gradient_rows": active_rows,
        "shared_median_weight_gradient": weight_grad[MEDIAN].tolist(),
        "shared_median_bias_gradient": float(bias_grad[MEDIAN]),
        "median_logit_change_by_target": dict(zip(TARGETS, delta.tolist(), strict=True)),
        "close_return_change": float(delta[0] + delta[1]),
        "close_points_change_at_synthetic_previous_close_22000": float(
            22000 * (close_after - close_before)
        ),
        "sgd_analytic_max_abs_error": sgd_formula_error,
        "counterexample_confirmed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "reports/quant/head-wick-parameter-coupling-2026-10-25.json"
    ))
    args = parser.parse_args()
    provenance = installed_provenance()
    if not provenance["commit_matches"] or not provenance["version_matches"]:
        raise RuntimeError("installed TimesFM source is not the pinned package")
    package = importlib.util.find_spec("timesfm3")
    official_source = Path(package.origin).parent / "torch" / "model.py"
    trainer_source = Path(training.__file__)
    trainer = inspect.getsource(training.train_output_head)
    assert "clip_grad" not in trainer
    report = {
        "schema_version": "head-shared-parameter-coupling-audit-v1",
        "status": "SHARED_PARAMETER_COUPLING_CONFIRMED",
        "claim_rejected": "Zero wick-loss partial derivatives with respect to g/b logits imply g/b point forecasts can never change.",
        "provenance": provenance,
        "pinned_code_commit": CODE_COMMIT,
        "sources": [
            _source_record(official_source, [
                "self.output_head = nn.Linear(",
                "output_patch_len * self.num_quantiles,",
                "raw_logits = self.output_head(transformer_output)",
                "b, v, n_patches, self.output_patch_len, self.num_quantiles",
            ]),
            _source_record(trainer_source, [
                "def _batch_inputs(", "raw = model.output_head(full_hidden)",
                "optimizer = torch.optim.AdamW(", "loss.backward()", "optimizer.step()",
                "spec.wick_weight * wick_loss", "early_stop_loss",
            ]),
        ],
        "axis_interpretation": {
            "inputs": "batch x variate x patch x input_patch_length",
            "targets": "g,b,log_u,log_d occupy the first four variates, not independent output-head parameter rows",
            "head_weight": "(horizon x quantile) x hidden; shared across batches, variates and patches",
            "median_head_row": "horizon 0 x 9 + quantile index 4 = row 4 for every GBUD variate",
        },
        "counterexamples": [
            _synthetic_counterexample("SGD"),
            _synthetic_counterexample("AdamW"),
        ],
        "training_implications": [
            "The direct wick output gradient is confined to u/d medians, but those medians update the same weight row and bias used by g/b medians.",
            "For the selected non-CPM context patch, cached reverse-RevIN statistics are fixed; positive per-variate scales rescale the shared-parameter coupling rather than remove it.",
            "The current trainer has no gradient clipping. Do not attribute behavior to an absent global clipping operation.",
            "AdamW's elementwise adaptive updates can mask some gradient changes on a particular step; exact cancellation, zero derivatives or orthogonality can occur. Coupling is possible, not a guarantee every actual g/b forecast changes.",
            "The existing crossing term couples adjacent quantile outputs through their loss gradients. A changed median can affect other quantile-row updates on later steps.",
            "Early stopping uses the full loss including wick_weight, so adding wick loss can also change the restored epoch even where some optimizer updates coincide.",
        ],
        "may_conclude": [
            "The asserted structural impossibility of changing g/b or Close is false.",
            "A wick candidate may change Open and Close and must be assessed on all OHLC gates and folds.",
        ],
        "may_not_conclude": [
            "This synthetic audit cannot predict actual validation error, improvement, degradation, effect size or selected epoch.",
            "No new real-data candidate was trained; no pretrained weights were loaded and no acceptance gate was run or passed.",
            "The audit does not establish that wick loss is the right intervention or that it repairs the TAIEX first-fold weakness.",
        ],
        "research_budget_ledger_changed": False,
        "reproduce": ".venv/bin/python research/audit_head_parameter_coupling.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
