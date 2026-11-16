"""Fixed TimesFM3 zero-shot comparison on previously seen historical dates.

Local, non-commercial research only. No training or distribution of weights.
Two predeclared variants: past relative OHLC; night-adjusted residual OHLC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from scripts.run_historical_validation import score_day, summarize
from scripts.run_night_ablation import build_paired_samples
from tw_scenario.shadow import seal
from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import OHLC, TAIPEI, encode_components
from tw_scenario.taifex import parse_tx_csv

FIELDS = ("open", "high", "low", "close")
METHODS = ("timesfm3_price_only", "timesfm3_night_residual")
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_for(samples, index, method, length=64):
    if method not in METHODS or index < length:
        raise ValueError("unknown method or insufficient context")
    current, past = samples[index], samples[index - length:index]
    if [r["date"] for r in past] != sorted({r["date"] for r in past}):
        raise ValueError("duplicate or unordered context")
    if any(r["date"] >= current["date"] for r in past):
        raise ValueError("context labels must precede forecast day")
    result = []
    for row in past:
        base = math.log1p(row["night_return_bps"] / 10000) if method == METHODS[1] else 0.0
        result.append([math.log(getattr(row["actual"], f) / row["previous_close"]) - base
                       for f in FIELDS])
    array = np.asarray(result, dtype=np.float32).T
    if array.shape != (4, length) or not np.isfinite(array).all():
        raise ValueError("invalid context")
    offset = math.log1p(current["night_return_bps"] / 10000) if method == METHODS[1] else 0.0
    return array, offset


def reconstruct(predicted_log_relative, reference, offset):
    values = np.asarray(predicted_log_relative, dtype=float).reshape(-1)
    if len(values) != 4 or not np.isfinite(values).all() or reference <= 0:
        raise ValueError("invalid model output/reference")
    raw = {f: reference * math.exp(float(v) + offset) for f, v in zip(FIELDS, values)}
    # Predeclared geometric projection, not a fitted or tuned postprocessor.
    projected = {**raw, "high": max(raw["high"], raw["open"], raw["close"]),
                 "low": min(raw["low"], raw["open"], raw["close"])}
    return OHLC(**projected), {f: projected[f] != raw[f] for f in ("high", "low")}


def main(root, output, weights):
    import torch
    from timesfm3 import TimesFM3Forecaster

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    original = read(root / "reports/research/night-ablation-v1/plan.json")
    summary = read(root / "reports/research/night-ablation-v1/summary.json")
    input_hashes = original["input_files"] | original["price_inputs"]
    for name, digest in input_hashes.items():
        if sha(root / name) != digest:
            raise ValueError("frozen data changed")
    if weights.name != REVISION:
        raise ValueError("weights must be the pinned snapshot directory")
    license_text = (weights / "LICENSE").read_text()
    if "Non-Commercial" not in license_text or "Outputs" not in license_text:
        raise ValueError("expected research license not found; inspect before model use")
    plan = {
        "experiment": "timesfm3-fixed-comparison-v1", "frozen_at": datetime.now(TAIPEI).isoformat(),
        "model": "google/timesfm-3.0-pytorch", "revision": REVISION,
        "weights_sha256": sha(weights / "model.safetensors"),
        "license_sha256": sha(weights / "LICENSE"), "license": "non-commercial, non-production research",
        "implementation_sha256": {p: sha(root / p) for p in (
            "scripts/run_timesfm_comparison.py", "scripts/run_night_ablation.py",
            "scripts/run_historical_validation.py", "src/tw_scenario/split_forecast.py",
            "src/tw_scenario/taifex.py")},
        "input_hashes": input_hashes, "context_length": 64, "horizon": 1,
        "target": "four log(price / same-contract previous daytime close) channels",
        "night_variant": "subtract each historical same-contract night log-return, add current completed night log-return",
        "znorm": "official use_znorm=True; per-query past context only",
        "projection": "H=max(rawH,O,C); L=min(rawL,O,C); positive exp reconstruction",
        "methods": list(METHODS), "threads": 4, "batch_size": 1, "budget_seconds": 600,
        "evaluation_start": "2026-01-01", "evaluation_end": "2026-09-30",
        "hyperparameter_search": False, "parameter_updates": 0,
        "evaluation_previously_seen": True, "pretraining_overlap": "unknown",
        "historical_publication_verified": False, "new_independent_test": False,
        "packages": {n: version(n) for n in ("timesfm", "torch", "numpy", "huggingface-hub")},
        "python": platform.python_version(), "weights_distributed": False,
    }
    write_immutable_json(plan, output / "plan.json")
    futures = [r for n in original["input_files"]
               for r in parse_tx_csv((root / n).read_bytes(), n)]
    cash = [r for n in original["price_inputs"] for r in read(root / n)]
    samples, exclusions, opportunities = build_paired_samples(cash, futures)
    torch.set_num_threads(4)
    model = TimesFM3Forecaster.from_pretrained(str(weights), device="cpu", per_core_batch_size=1,
                                             local_files_only=True)
    smoke = np.array([np.sin(np.arange(64) / 8 + j) for j in range(4)], dtype=np.float32)
    with torch.inference_mode():
        smoke_out = model.predict(smoke, horizon=1, use_znorm=True).forecast
    if np.shape(smoke_out) != (4, 1) or not np.isfinite(smoke_out).all():
        raise ValueError("official multivariate zero-shot smoke failed")
    private, public = {}, {}
    for market, rows in samples.items():
        indices = [i for i, r in enumerate(rows) if "2026-01-01" <= r["date"] <= "2026-09-30"]
        expected = [d for d in opportunities[market] if "2026-01-01" <= d <= "2026-09-30"]
        if [rows[i]["date"] for i in indices] != expected:
            raise ValueError("source calendar incomplete; do not drop difficult dates")
        private[market], public[market] = {}, {}
        for method in METHODS:
            results, failures, projection = [], [], {"high": 0, "low": 0}
            for position, index in enumerate(indices):
                if time.monotonic() - started > 600:
                    raise TimeoutError("predeclared runtime budget exhausted")
                sample = rows[index]
                try:
                    context, offset = context_for(rows, index, method)
                    with torch.inference_mode():
                        prediction = model.predict(context, horizon=1, use_znorm=True).forecast
                    prices, projected = reconstruct(prediction[:, 0], sample["previous_close"], offset)
                    scored = score_day(sample, encode_components(sample["previous_close"], prices), prices, 5)
                    scored.update(contract=sample["contract"], previous_close=sample["previous_close"],
                                  context_first=rows[index - 64]["date"], context_last=rows[index - 1]["date"],
                                  context_sha256=hashlib.sha256(context.tobytes()).hexdigest(), projection=projected)
                    results.append(scored)
                    for f, changed in projected.items(): projection[f] += int(changed)
                    write_immutable_json(scored, output / "daily" / market / method / f"{sample['date']}.json")
                except (ValueError, OverflowError, RuntimeError) as exc:
                    failures.append({"date": sample["date"], "error": str(exc)})
                if position % 60 == 0:
                    print(market, method, position + 1, "/", len(indices), flush=True)
            metrics = summarize(results, len(expected))
            metrics["rmse_points"] = {f: float(np.sqrt(np.mean([
                (r["predicted"][f] - r["actual"][f]) ** 2 for r in results]))) for f in FIELDS}
            private[market][method] = results
            base = summary["results"][market]["results"]["evaluation_seen"]
            complete = not failures and len(results) == len(expected)
            public[market][method] = {
                "metrics": metrics, "failures": failures, "projection_count": projection,
                "complete": complete,
                "relative_close_MAE_vs_night_baseline":
                    metrics["close_mae_bps"] / base["night_carry_with_mean_body"]["metrics"]["close_mae_bps"] - 1
                    if complete else None,
                "relative_close_MAE_vs_frozen_night_ridge":
                    metrics["close_mae_bps"] / base["with_night_gap_only"]["metrics"]["close_mae_bps"] - 1
                    if complete else None,
                "baseline_metrics": {m: b["metrics"] for m, b in base.items()},
            }
    for p, h in plan["implementation_sha256"].items():
        if sha(root / p) != h: raise ValueError("implementation changed during run")
    report = seal({"experiment": plan["experiment"], "markets": public,
                   "runtime_seconds": time.monotonic() - started,
                   "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "plan_sha256": sha(output / "plan.json"), "parameter_updates": 0,
                   "formal_acceptance": False, "new_independent_test": False,
                   "historical_publication_verified": False, "pretraining_overlap": "unknown",
                   "information_set_note": "past-only and night-residual variants are separate; no controlled algorithm-only claim",
                   "data_exclusions": exclusions})
    write_immutable_json(private, output / "daily-private.json")
    write_immutable_json(report, output / "summary.json")
    print(json.dumps({"runtime_seconds": report["runtime_seconds"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve(), args.weights.resolve())
