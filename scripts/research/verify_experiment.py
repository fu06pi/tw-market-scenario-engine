"""Reload a completed run and bootstrap validation uncertainty; never touch final test."""

import argparse
import json
from pathlib import Path

import numpy as np

from tw_scenario.research import night
from tw_scenario.research.baselines import forecast
from tw_scenario.research.data import (
    indices,
    load_dataset,
    metrics,
    reconstruct,
    sample_arrays,
    write_new,
)
from tw_scenario.research.model import Forecaster

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--output", default="replay-and-uncertainty.json")
args = parser.parse_args()
folder = root / args.run_dir
cfg = json.loads((folder / "config.json").read_text())
sources = json.loads((folder / "sources.json").read_text())
summary = json.loads((folder / "summary.json").read_text())
model = Forecaster()
report = {"scope": "validation replay only; not final-test acceptance", "markets": {}}
for market, data in sources["markets"].items():
    raw, bars = load_dataset(root / data["file"])
    if raw["checksum"] != summary["markets"][market]["dataset_checksum"]:
        raise ValueError("dataset/run mismatch")
    bars = [b for b in bars if str(b.trading_date) <= cfg["validation_end"]]
    selected = indices(bars, cfg["context"], cfg["validation_start"], cfg["validation_end"])
    best = summary["markets"][market]["best_eligible_timesfm"]
    if best is None:
        report["markets"][market] = {"status": "no_eligible_model"}
        continue
    candidate = next(c for c in summary["markets"][market]["candidates"] if c["name"] == best)
    checkpoint = root / candidate["checkpoint"]
    meta = model.load_head(checkpoint)
    x, _, previous, actual = sample_arrays(bars, meta["representation"], cfg["context"], selected)
    night_series = None
    if cfg.get("night_covariates"):
        night_data = json.loads((root / cfg["night_covariates"]).read_text())
        if night_data["checksum"] != meta["night_source_checksum"]:
            raise ValueError("checkpoint/night source mismatch")
        night_series = night_data["series"]
    cov = (
        night.covariates(bars, selected, cfg["context"], night_series)
        if night_series is not None
        else None
    )
    prediction = reconstruct(
        model.predict(x, cfg["batch_size"], covariates=cov)[:, :, 4],
        previous,
        meta["representation"],
    )
    replay = metrics(prediction, actual, previous)
    if not np.isclose(
        replay["close"]["mae_points"],
        candidate["best_validation_metrics"]["close"]["mae_points"],
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError("replayed validation result differs")
    baseline_name = summary["markets"][market]["locked_baseline"]
    if baseline_name == "night_regression_train_only":
        training = indices(bars, cfg["context"], "1900-01-01", cfg["train_end"])
        baseline = night.regression(bars, training, selected, night_series)
    else:
        baseline = forecast(bars, selected, baseline_name)
    e = np.abs(prediction[:, 3] - actual[:, 3])
    b = np.abs(baseline[:, 3] - actual[:, 3])
    rng = np.random.default_rng(1729)
    ratios = []
    n = len(e)
    block = 5
    for _ in range(2000):
        starts = rng.integers(0, n - block + 1, size=(n + block - 1) // block)
        sample = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
        ratios.append(float(e[sample].mean() / b[sample].mean()))
    report["markets"][market] = {
        "checkpoint_replay": "matched",
        "metrics": replay,
        "best_epoch": candidate["best_epoch"],
        "selected_model": best,
        "baseline": summary["markets"][market]["locked_baseline"],
        "close_mae_ratio": float(e.mean() / b.mean()),
        "moving_block_bootstrap": {
            "block_observations": 5,
            "replicates": 2000,
            "seed": 1729,
            "ratio_95ci": np.quantile(ratios, [0.025, 0.975]).tolist(),
        },
        "rows": len(e),
        "final_test_evaluated": False,
    }
import hashlib

report["implementation_source_hashes"] = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((root / "src/tw_scenario/research").glob("*.py"))
}
write_new(folder / args.output, report)
print(json.dumps(report, indent=2))
