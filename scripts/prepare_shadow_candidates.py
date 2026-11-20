"""Freeze the prospective plan, compare validation only, and export JSON weights.

No new final test is run. 2026 labels are used only for the final, now-frozen
prospective checkpoint refit; they cannot affect selection or interval radii.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

from scripts.run_historical_validation import score_day, summarize
from scripts.run_night_ablation import build_paired_samples, paired_interval
from tw_scenario.intervals import calibrate
from tw_scenario.policy import load_scoring_policy
from tw_scenario.refinements import BASELINES, CANDIDATES, METHODS, fit, predict
from tw_scenario.shadow import now, seal
from tw_scenario.snapshot import write_immutable_json
from tw_scenario.taifex import parse_tx_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    cash_files = [root / "data/processed" / name for name in ("taiex-tej.json", "taiex-twse.json")]
    tx_files = sorted((root / "data/raw/taifex").glob("TX-*.csv"))
    source_files = [*cash_files, *tx_files]
    policy = load_scoring_policy(root / "config/targets.yaml")
    code_files = [root / p for p in (
        "src/tw_scenario/refinements.py", "src/tw_scenario/intervals.py",
        "src/tw_scenario/shadow.py", "scripts/prepare_shadow_candidates.py",
        "scripts/run_night_ablation.py", "scripts/run_historical_validation.py",
        "src/tw_scenario/split_forecast.py", "src/tw_scenario/taifex.py",
        "scripts/run_shadow.py", "scripts/monitor_shadow.py", "scripts/verify_shadow_artifact.py",
        "src/tw_scenario/policy.py", "src/tw_scenario/snapshot.py", "config/targets.yaml",
        ".github/workflows/preopen-shadow.yml", "pyproject.toml",
        "requirements-shadow.txt",
    )]
    plan = {
        "version": "prospective-night-v1", "frozen_at": now().isoformat(),
        "steps": ["preopen_shadow", "night_residual", "body_night_ablation",
                  "taiex_gap_calibration", "marginal_intervals_and_monitoring"],
        "alpha": 0.1, "minimum_training": 60, "baselines": list(BASELINES),
        "candidates": list(CANDIDATES), "validation_start": "2025-10-01",
        "validation_end": "2025-12-31", "selection_first_n": 32,
        "interval_calibration_remaining_minimum": 30, "interval_alpha": 0.2,
        "selection": "minimum close MAE baseline; candidate >=5% close and <=5% OHL worsening",
        "retrospective_validation_already_seen": True,
        "independent_final_test_executed": False,
        "final_refit": "all cached paired labels through 2026-09-30, for FUTURE use only",
        "prospective_weights_update_rule": "frozen; no automatic retraining",
        "budget_seconds": 180, "seed": 20261104,
        "scoring_policy": policy.model_dump(),
        "runtime": {"python": platform.python_version(),
                    "packages": {n: version(n) for n in ("numpy", "scikit-learn", "pydantic")}},
        "input_hashes": {str(p.relative_to(root)): digest(p) for p in source_files},
        "code_hashes": {str(p.relative_to(root)): digest(p) for p in code_files},
    }
    write_immutable_json(plan, output / "plan.json")
    futures = [r for p in tx_files for r in parse_tx_csv(p.read_bytes(), str(p.relative_to(root)))]
    cash = [r for p in cash_files for r in json.loads(p.read_text())]
    samples, exclusions, attempted = build_paired_samples(cash, futures)
    summary, markets, private = {}, {}, {}
    for market, rows in samples.items():
        validation = [i for i, r in enumerate(rows) if "2025-10-01" <= r["date"] <= "2025-12-31"]
        if len(validation) < 62:
            raise ValueError("need 32 selection and 30 independent calibration dates")
        selection, calibration = validation[:32], validation[32:]
        results, failures = {m: [] for m in METHODS}, []
        for index in validation:
            for method in METHODS:
                if time.monotonic() - started > 180:
                    raise TimeoutError("frozen CPU budget exhausted")
                try:
                    components, prices = predict(fit(rows[:index], method, market), rows[index])
                    score = score_day(rows[index], components, prices, policy.flat_threshold_bps)
                    results[method].append({**score, "market": market, "method": method})
                except (ValueError, OverflowError) as exc:
                    failures.append({"date": rows[index]["date"], "method": method, "reason": str(exc)})
        private[market] = results
        select_dates = {rows[i]["date"] for i in selection}
        cal_dates = {rows[i]["date"] for i in calibration}
        scores = {m: summarize([r for r in v if r["date"] in select_dates], len(selection))
                  for m, v in results.items()}
        if failures or any(s["scored"] != len(selection) for s in scores.values()):
            write_immutable_json({"failures": failures}, output / f"{market}-failures.json")
            raise ValueError("incomplete common validation; no selection/export")
        baseline = min(BASELINES, key=lambda m: (scores[m]["close_mae_bps"], m))
        if any(scores[baseline][f"{k}_mae_bps"] <= 0 for k in ("open", "high", "low", "close")):
            raise ValueError("zero-error baseline: relative improvement gate is undefined")
        gates = {}
        for method in CANDIDATES:
            ratios = {k: scores[method][f"{k}_mae_bps"] / scores[baseline][f"{k}_mae_bps"]
                      for k in ("open", "high", "low", "close")}
            gates[method] = {"ratios_vs_locked_baseline": ratios,
                             "validation_gate": ratios["close"] <= 0.95 and
                             all(ratios[k] <= 1.05 for k in ("open", "high", "low"))}
        interval_models = {m: calibrate([r for r in results[m] if r["date"] in cal_dates],
                                        "2026-11-05") for m in METHODS}
        # Do not tune any choices using the previously evaluated 2026 period.
        refit_rows = [r for r in rows if r["date"] <= "2026-09-30"]
        markets[market] = {"default_baseline": baseline, "models": {
            m: fit(refit_rows, m, market) for m in METHODS}, "intervals": interval_models}
        stats = {"seed": 20261104, "draws": 2000, "block_length": 5}
        selected_results = {m: [r for r in results[m] if r["date"] in select_dates] for m in METHODS}
        summary[market] = {
            "selection_n": len(selection), "selection_first": rows[selection[0]]["date"],
            "selection_last": rows[selection[-1]]["date"], "calibration_n": len(calibration),
            "calibration_first": rows[calibration[0]]["date"],
            "calibration_last": rows[calibration[-1]]["date"],
            "locked_baseline": baseline, "default_remains_baseline": True,
            "selection_scores": scores, "candidate_gates": gates,
            "close_improvement_uncertainty_vs_baseline": {m: paired_interval(
                selected_results[baseline], selected_results[m], "close_ae_bps", stats
            ) for m in CANDIDATES},
            "body_night_vs_control": {k: 1 - scores["body_night"][f"{k}_mae_bps"] /
                                     scores["body_control"][f"{k}_mae_bps"]
                                     for k in ("open", "high", "low", "close")},
            "calibration_vs_body_night": {k: 1 - scores["gap_calibrated"][f"{k}_mae_bps"] /
                                         scores["body_night"][f"{k}_mae_bps"]
                                         for k in ("open", "high", "low", "close")},
            "failures": failures, "matched_rows": len(rows), "attempted_rows": len(attempted[market]),
            "exclusions": exclusions[market], "refit_through": refit_rows[-1]["date"],
        }
    # Verify that neither inputs nor implementation changed during the experiment.
    for mapping in (plan["input_hashes"], plan["code_hashes"]):
        if any(digest(root / p) != h for p, h in mapping.items()):
            raise ValueError("frozen inputs/code changed during run")
    checkpoint = seal({
        "schema_version": "shadow-checkpoint-v1", "frozen_at": now().isoformat(),
        "plan_sha256": digest(output / "plan.json"), "markets": markets,
        "scoring_policy": policy.model_dump(),
        "historical_availability_verified": False,
        "interval_warning": "Old retrospective calibration; 80% coverage is unverified prospectively",
    })
    write_immutable_json(checkpoint, output / "checkpoint.json")
    write_immutable_json(private, output / "daily-private.json")
    report = {"version": "prospective-night-v1", "markets": summary,
              "runtime_seconds": time.monotonic() - started,
              "checkpoint_sha256": digest(output / "checkpoint.json"),
              "prospective_forecasts": 0, "formal_acceptance": False,
              "historical_availability_verified": False, "new_independent_test": False}
    write_immutable_json(report, output / "summary.json")
    print(json.dumps({"checkpoint": str(output / "checkpoint.json"),
                      "runtime_seconds": report["runtime_seconds"],
                      "baselines": {m: v["default_baseline"] for m, v in markets.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
