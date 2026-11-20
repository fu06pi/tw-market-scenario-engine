"""Comparable simple baselines on explicitly already-seen historical periods.

The original study and its frozen selections remain unchanged. Per-day rows and
licensed actuals stay in ignored reports/generated; only aggregate reports ship.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from tw_scenario.baselines import (
    DIRECTION_BASELINES,
    OVERNIGHT_FEATURES,
    PRICE_BASELINES,
    VERSION,
    direction_baseline,
    price_baseline,
)
from tw_scenario.policy import load_scoring_policy
from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import OHLC, encode_components

if __package__:
    from .run_historical_validation import build_samples, file_hash, read_json, score_day, summarize
else:
    from run_historical_validation import build_samples, file_hash, read_json, score_day, summarize


def select_complete(rows, metric, maximize=False):
    """Selection never uses evaluation-period scores or incomplete model coverage."""
    eligible = [
        (name, row)
        for name, row in rows.items()
        if row["metrics"].get("scored") == row["metrics"].get("expected")
        and row["metrics"].get("scored", 0) > 0
        and not row["failures"]
        and row["metrics"].get(metric) is not None
    ]
    if not eligible:
        raise ValueError("no complete baseline is eligible for validation selection")
    # Stable insertion order is the predeclared tie-breaker.
    chooser = max if maximize else min
    return chooser(eligible, key=lambda item: item[1]["metrics"][metric])[0]


def complete_comparison(candidate, baseline, price=True):
    """Do not compare aggregates over different denominators or failed subsets."""
    c, b = candidate["metrics"], baseline["metrics"]
    if (
        candidate["failures"]
        or baseline["failures"]
        or not c.get("expected")
        or c["expected"] != b["expected"]
        or c["scored"] != c["expected"]
        or b["scored"] != b["expected"]
    ):
        return None
    if price:
        return 1 - c["close_mae_bps"] / b["close_mae_bps"] if b["close_mae_bps"] > 0 else None
    return c["close_accuracy"] - b["close_accuracy"]


def align_reference(rows, samples, indices):
    expected = {samples[i]["date"]: samples[i] for i in indices}
    dates = [r["date"] for r in rows]
    if len(set(dates)) != len(dates) or set(dates) != set(expected):
        raise ValueError("reference forecasts must cover exactly the common evaluation dates")
    aligned = {}
    for row in rows:
        sample = expected[row["date"]]
        if OHLC.model_validate(row["actual"]) != sample["actual"]:
            raise ValueError("reference actuals disagree with the common source")
        aligned[row["date"]] = OHLC.model_validate(row["predicted"])
    return aligned


def direction_summary(rows, expected):
    metrics = {
        "expected": expected,
        "scored": len(rows),
        "coverage_of_available_rows": len(rows) / expected if expected else None,
        "price_metrics": None,
    }
    if not rows:
        return metrics
    for target in ("gap", "body", "close"):
        metrics[f"{target}_hits"] = sum(r[f"{target}_hit"] for r in rows)
        metrics[f"{target}_accuracy"] = metrics[f"{target}_hits"] / len(rows)
        matrix = {p: {a: 0 for a in ("up", "flat", "down")} for p in ("up", "flat", "down")}
        for row in rows:
            matrix[row[f"{target}_direction_predicted"]][row[f"{target}_direction_actual"]] += 1
        metrics[f"{target}_confusion_predicted_by_actual"] = matrix
    return metrics


def run_comparison(root: Path, reference: Path, output: Path):
    root, reference, output = root.resolve(), reference.resolve(), output.resolve()
    if any(
        output.is_relative_to(root / name)
        for name in (
            "predictions",
            "realizations",
            "data",
            "reports/evaluations",
            "reports/research",
        )
    ):
        raise ValueError("comparison output must use a new research scratch directory")
    original = read_json(reference / "plan.json")
    policy = load_scoring_policy(root / "config/targets.yaml")
    if policy.flat_threshold_bps != original["flat_threshold_bps"]:
        raise ValueError("scoring threshold differs from the frozen reference study")
    for name, digest in original["input_files"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"source changed since reference study: {name}")
    # Freeze all methods and known limitations before computing the new scores.
    output.mkdir(parents=True, exist_ok=False)
    reference_files = [
        reference / name
        for name in (
            "plan.json",
            "summary.json",
            "daily-private.json",
            "TAIEX-locked-selection.json",
            "TX202603-locked-selection.json",
        )
    ]
    plan = {
        "benchmark_version": VERSION,
        "scope": "comparison_on_previously_seen_history",
        "new_independent_test_evidence": False,
        "formal_acceptance_pass": False,
        "historical_asof_verified": False,
        "scoring_policy": policy.model_dump(),
        "input_sha256": original["input_files"],
        "reference_sha256": {p.name: file_hash(p) for p in reference_files},
        "implementation_sha256": {
            str(p.relative_to(root)): file_hash(p)
            for p in (
                root / "src/tw_scenario/baselines.py",
                root / "scripts/compare_simple_baselines.py",
                root / "scripts/run_historical_validation.py",
                root / "src/tw_scenario/policy.py",
            )
        },
        "markets": original["markets"],
        "information_set": "prior completed daytime OHLC",
        "minimum_common_history": original["minimum_training_samples"],
        "price_baselines": list(PRICE_BASELINES),
        "direction_only_baselines": list(DIRECTION_BASELINES),
        "fixed_window": 20,
        "window_search": False,
        "wick_rule": original["wick_rule"],
        "availability_assumption": original["availability_assumption"],
        "references": ["split_ridge", "direct_close_ridge"],
        "selection": {
            "price": "lowest validation close MAE in bps among complete price baselines",
            "direction": "highest validation close accuracy among complete direction-only baselines",
            "ties": "fixed method declaration order",
        },
        "blocked_baselines": {
            "overnight_gap": {
                "status": "unavailable",
                "information_set": "night_and_overseas",
                "targets": ["gap"],
                "reason": "No frozen night/SOX/TSMC ADR feature histories in this price-only dataset",
                "required_features": list(OVERNIGHT_FEATURES),
                "included_in_rankings": False,
            }
        },
    }
    write_immutable_json(plan, output / "plan.json")
    histories = {
        "TAIEX": read_json(root / "data/processed/taiex-tej.json")
        + read_json(root / "data/processed/taiex-twse.json"),
        "TX202603": read_json(root / "data/processed/tx-tej.json"),
    }
    reference_daily = read_json(reference / "daily-private.json")
    reference_summary = read_json(reference / "summary.json")
    summary, private = {}, {}
    for market, history in histories.items():
        samples = build_samples(history)
        split = original["markets"][market]
        phases = {
            "validation": [
                i
                for i, r in enumerate(samples)
                if split["validation_start"] <= r["date"] <= split["validation_end"]
            ],
            "evaluation_seen": [
                i
                for i, r in enumerate(samples)
                if split["test_start"] <= r["date"] <= split["test_end"]
            ],
        }
        if (
            not all(phases.values())
            or min(min(v) for v in phases.values()) < plan["minimum_common_history"]
        ):
            raise ValueError("common dates lack the predeclared warmup")
        locked_reference = read_json(reference / f"{market}-locked-selection.json")
        validation_rows, evaluation_rows = {}, {}
        private[market] = {}
        for phase, indices in phases.items():
            result_rows = validation_rows if phase == "validation" else evaluation_rows
            private[market][phase] = {}
            for name in (*PRICE_BASELINES, *DIRECTION_BASELINES):
                rows, failures = [], []
                for index in indices:
                    sample = samples[index]
                    past = [r["target"] for r in samples[:index]]
                    try:
                        if name in PRICE_BASELINES:
                            prediction = price_baseline(
                                name, past, sample["previous_close"], policy
                            )
                            row = score_day(
                                sample,
                                prediction.components,
                                prediction.prices,
                                policy.flat_threshold_bps,
                            )
                        else:
                            prediction = direction_baseline(name, past, policy)
                            from tw_scenario.baselines import component_directions

                            actual = component_directions(sample["target"], policy)
                            row = {"date": sample["date"]}
                            for target in ("gap", "body", "close"):
                                row[f"{target}_direction_predicted"] = prediction.directions[
                                    target
                                ].value
                                row[f"{target}_direction_actual"] = actual[target].value
                                row[f"{target}_hit"] = int(
                                    prediction.directions[target] == actual[target]
                                )
                        rows.append(row)
                    except (ValueError, OverflowError) as exc:
                        failures.append({"date": sample["date"], "error": str(exc)})
                metrics = (
                    summarize(rows, len(indices))
                    if name in PRICE_BASELINES
                    else (direction_summary(rows, len(indices)))
                )
                metrics["coverage_of_available_rows"] = metrics["scored"] / metrics["expected"]
                result_rows[name] = {
                    "version": VERSION,
                    "kind": "price" if name in PRICE_BASELINES else "direction_only",
                    "information_set": "past_day_ohlc",
                    "metrics": metrics,
                    "failures": failures,
                }
                private[market][phase][name] = rows
            if phase == "validation":
                direct = locked_reference["direct_baseline"]
                result_rows["direct_close_ridge"] = {
                    "kind": "price_reference",
                    "information_set": "past_day_ohlc",
                    "metrics": direct["metrics"],
                    "failures": direct["failures"],
                }
                candidate = locked_reference["candidate"]
                result_rows["split_ridge"] = {
                    "kind": "frozen_candidate",
                    "information_set": "past_day_ohlc",
                    "metrics": candidate["metrics"],
                    "failures": candidate["failures"],
                }
                price_candidates = {
                    k: v
                    for k, v in result_rows.items()
                    if v["kind"] in ("price", "price_reference")
                }
                direction_candidates = {
                    k: v for k, v in result_rows.items() if v["kind"] == "direction_only"
                }
                selected_price = select_complete(price_candidates, "close_mae_bps")
                selected_direction = select_complete(
                    direction_candidates, "close_accuracy", maximize=True
                )
                write_immutable_json(
                    {
                        "selected_price": selected_price,
                        "selected_direction": selected_direction,
                        "validation": validation_rows,
                        "evaluation_already_seen": True,
                    },
                    output / f"{market}-selection.json",
                )
            else:
                for model in ("split_ridge", "direct_close_ridge"):
                    aligned = align_reference(reference_daily[market][model], samples, indices)
                    rows = []
                    for index in indices:
                        sample = samples[index]
                        prices = aligned[sample["date"]]
                        rows.append(
                            score_day(
                                sample,
                                encode_components(sample["previous_close"], prices),
                                prices,
                                policy.flat_threshold_bps,
                            )
                        )
                    metrics = summarize(rows, len(indices))
                    previous = reference_summary["results"][market]["test"][model]["metrics"]
                    for key in ("open_mae_bps", "close_mae_bps", "close_accuracy"):
                        if not math.isclose(
                            metrics[key], previous[key], rel_tol=1e-9, abs_tol=1e-9
                        ):
                            raise ValueError(
                                "frozen reference scores changed; do not silently replace"
                            )
                    result_rows[model] = {
                        "kind": "frozen_candidate" if model == "split_ridge" else "price_reference",
                        "information_set": "past_day_ohlc",
                        "metrics": metrics,
                        "failures": [],
                    }
                    private[market][phase][model] = rows
        summary[market] = {
            "common_dates": {
                phase: [samples[i]["date"] for i in indices] for phase, indices in phases.items()
            },
            "selected_price_baseline": selected_price,
            "selected_direction_baseline": selected_direction,
            "validation": validation_rows,
            "evaluation_seen": evaluation_rows,
            "candidate_vs_locked_price_improvement": complete_comparison(
                evaluation_rows["split_ridge"], evaluation_rows[selected_price]
            ),
            "candidate_minus_locked_direction_accuracy": complete_comparison(
                evaluation_rows["split_ridge"], evaluation_rows[selected_direction], price=False
            ),
        }
    for name, digest in plan["input_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError("source mutated during baseline comparison")
    for name, digest in plan["reference_sha256"].items():
        if file_hash(reference / name) != digest:
            raise ValueError("reference mutated during baseline comparison")
    report = {
        "benchmark_version": VERSION,
        "scope": plan["scope"],
        "new_independent_test_evidence": False,
        "formal_acceptance_pass": False,
        "historical_asof_verified": False,
        "plan_sha256": file_hash(output / "plan.json"),
        "markets": summary,
        "blocked_baselines": plan["blocked_baselines"],
    }
    write_immutable_json(report, output / "summary.json")
    write_immutable_json(private, output / "daily-private.json")
    for market, result in summary.items():
        print(
            market,
            "price baseline:",
            result["selected_price_baseline"],
            "direction baseline:",
            result["selected_direction_baseline"],
        )
        for name, row in result["evaluation_seen"].items():
            metrics = row["metrics"]
            print(
                name,
                "n=",
                metrics["scored"],
                "close hit=",
                metrics.get("close_accuracy"),
                "close MAE(bp)=",
                metrics.get("close_mae_bps"),
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_comparison(args.root, args.reference, args.output)
