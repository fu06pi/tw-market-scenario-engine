"""Historical daily-price research, with explicit unverified availability assumptions.

This never creates formal V2 pre-open snapshots or alters source available_at.
Detailed rows stay in the ignored reports/generated directory, not public Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tw_scenario.metrics import realized_direction
from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import OHLC, Components, encode_components, fit_ridge_head


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_samples(history):
    ordered = sorted(history, key=lambda row: row["trading_date"])
    dates = [r["trading_date"] for r in ordered]
    if len(set(dates)) != len(dates):
        raise ValueError("duplicate history dates")
    identities = {(r["market"], r.get("contract"), r["session"]) for r in ordered}
    if len(identities) != 1:
        raise ValueError("each experiment requires one market, daytime session and contract")
    if next(iter(identities))[2] != "day":
        raise ValueError("night and day observations cannot be mixed")
    bars = [OHLC.model_validate(r["prices"]) for r in ordered]
    samples = []
    for i in range(2, len(ordered)):
        previous, reference = bars[i - 1], bars[i - 2].close
        prior = encode_components(reference, previous)
        samples.append(
            {
                "date": dates[i],
                "feature_latest_date": dates[i - 1],
                "market": ordered[i]["market"],
                "contract": ordered[i].get("contract"),
                "previous_close": previous.close,
                "actual": bars[i],
                "features": [
                    math.expm1(prior.g) * 10000,
                    math.expm1(prior.g + prior.b) * 10000,
                    math.expm1(prior.b) * 10000,
                    (previous.high - previous.low) / reference * 10000,
                ],
                "target": encode_components(previous.close, bars[i]),
            }
        )
    return samples


def predict_day(samples, index, model, alpha, min_training=60):
    if index < min_training:
        raise ValueError("insufficient past training samples")
    current = samples[index]
    past = samples[:index]
    if any(r["date"] >= current["date"] for r in past):
        raise ValueError("training data must strictly precede prediction date")
    if current["feature_latest_date"] >= current["date"]:
        raise ValueError("features must come from previous sessions")
    targets = [r["target"] for r in past]
    u, d = (float(np.mean([getattr(t, k) for t in targets])) for k in ("u", "d"))
    if model == "zero_return":
        g = b = 0.0
    elif model == "expanding_mean":
        g, b = (float(np.mean([getattr(t, k) for t in targets])) for k in ("g", "b"))
    elif model in ("split_ridge", "direct_close_ridge"):
        x = np.asarray([r["features"] for r in past])
        query = np.asarray(current["features"])
        if model == "split_ridge":
            g, _ = fit_ridge_head(x[:, :2], [t.g for t in targets], query[:2], alpha)
            b, _ = fit_ridge_head(x[:, 2:], [t.b for t in targets], query[2:], alpha)
        else:
            g, _ = fit_ridge_head(x, [t.g for t in targets], query, alpha)
            total, _ = fit_ridge_head(x, [t.g + t.b for t in targets], query, alpha)
            b = total - g
    else:
        raise ValueError(f"unknown model {model}")
    predicted = Components(g=g, b=b, u=u, d=d)
    return predicted, predicted.reconstruct(current["previous_close"])


def score_day(sample, components, prices, flat_threshold):
    previous = sample["previous_close"]
    actual = sample["actual"]
    target = sample["target"]
    row = {
        "date": sample["date"],
        "prior_range_bps": sample["features"][3],
        "actual": actual.model_dump(),
        "predicted": prices.model_dump(),
    }
    for key in OHLC.model_fields:
        error = getattr(prices, key) - getattr(actual, key)
        row[f"{key}_ae_points"] = abs(error)
        row[f"{key}_ae_bps"] = abs(error) / previous * 10000
        row[f"{key}_squared_error_bps"] = (error / previous * 10000) ** 2
    for name, pred, realized in (
        ("gap", components.g, target.g),
        ("body", components.b, target.b),
        ("close", components.g + components.b, target.g + target.b),
    ):
        pred_bps, real_bps = math.expm1(pred) * 10000, math.expm1(realized) * 10000
        p = realized_direction(pred_bps, flat_threshold).value
        r = realized_direction(real_bps, flat_threshold).value
        row[f"{name}_direction_predicted"] = p
        row[f"{name}_direction_actual"] = r
        row[f"{name}_hit"] = int(p == r)
        row[f"{name}_ae_bps"] = abs(pred_bps - real_bps)
    row["range_ae_bps"] = (
        abs((prices.high - prices.low) - (actual.high - actual.low)) / previous * 10000
    )
    return row


def summarize(rows, expected):
    result = {
        "expected": expected,
        "scored": len(rows),
        "coverage_of_available_test_rows": len(rows) / expected if expected else None,
    }
    if not rows:
        return result
    result["first_date"], result["last_date"] = rows[0]["date"], rows[-1]["date"]
    for key in OHLC.model_fields:
        for unit in ("points", "bps"):
            result[f"{key}_mae_{unit}"] = float(np.mean([r[f"{key}_ae_{unit}"] for r in rows]))
        result[f"{key}_rmse_bps"] = float(
            np.sqrt(np.mean([r[f"{key}_squared_error_bps"] for r in rows]))
        )
    for target in ("gap", "body", "close"):
        result[f"{target}_hits"] = sum(r[f"{target}_hit"] for r in rows)
        result[f"{target}_accuracy"] = result[f"{target}_hits"] / len(rows)
        result[f"{target}_mae_bps"] = float(np.mean([r[f"{target}_ae_bps"] for r in rows]))
        matrix = {p: {a: 0 for a in ("up", "flat", "down")} for p in ("up", "flat", "down")}
        for r in rows:
            matrix[r[f"{target}_direction_predicted"]][r[f"{target}_direction_actual"]] += 1
        result[f"{target}_confusion_predicted_by_actual"] = matrix
    classes = Counter(r["close_direction_actual"] for r in rows)
    result["actual_close_class_counts"] = dict(classes)
    result["always_up_close_accuracy"] = classes["up"] / len(rows)
    result["always_down_close_accuracy"] = classes["down"] / len(rows)
    result["range_mae_bps"] = float(np.mean([r["range_ae_bps"] for r in rows]))
    return result


def bootstrap_improvement(candidate, baseline, settings):
    c = {r["date"]: r["close_ae_bps"] for r in candidate}
    b = {r["date"]: r["close_ae_bps"] for r in baseline}
    dates = sorted(set(c) & set(b))
    if not dates:
        return {"paired_n": 0}
    a = np.asarray([c[d] for d in dates])
    z = np.asarray([b[d] for d in dates])
    block = min(settings["block_length"], len(dates))
    rng = np.random.default_rng(settings["seed"])
    draws = []
    for _ in range(settings["draws"]):
        starts = rng.integers(0, len(dates) - block + 1, size=math.ceil(len(dates) / block))
        indices = np.concatenate([np.arange(s, s + block) for s in starts])[: len(dates)]
        draws.append(1 - float(np.mean(a[indices])) / float(np.mean(z[indices])))
    return {
        "paired_n": len(dates),
        "improvement": 1 - float(np.mean(a)) / float(np.mean(z)),
        "time_block_95pct_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        **settings,
    }


def main(root: Path, output: Path):
    started = time.monotonic()
    plan = read_json(output / "plan.json")
    for name, digest in plan["input_files"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"input changed after plan freeze: {name}")
    taiex_2025 = read_json(root / "data/processed/taiex-tej.json")
    taiex_2026 = read_json(root / "data/processed/taiex-twse.json")
    tx = read_json(root / "data/processed/tx-tej.json")
    overlap = read_json(root / "data/raw/twse/twse-taiex-202512.json")
    tej_by_date = {r["trading_date"]: r["prices"] for r in taiex_2025}
    matched = 0
    for row in overlap["data"]:
        y, m, d = map(int, row[0].split("/"))
        day = f"{y + 1911:04d}-{m:02d}-{d:02d}"
        if day not in tej_by_date:
            continue
        for k, value in zip(("open", "high", "low", "close"), row[1:5]):
            if abs(float(value.replace(",", "")) - tej_by_date[day][k]) > 0.011:
                raise ValueError("TAIEX source overlap mismatch; do not splice sources")
        matched += 1
    if matched != len(overlap["data"]):
        raise ValueError("TAIEX source overlap is incomplete")
    histories = {"TAIEX": taiex_2025 + taiex_2026, "TX202603": tx}
    selections, details, summaries, quality = {}, {}, {}, {}
    for market, history in histories.items():
        samples = build_samples(history)
        split = plan["markets"][market]
        validation = [
            i
            for i, r in enumerate(samples)
            if split["validation_start"] <= r["date"] <= split["validation_end"]
        ]
        if not validation or validation[0] < plan["minimum_training_samples"]:
            raise ValueError(f"{market} lacks enough training/validation rows for frozen dates")
        quality[market] = {
            "records": len(history),
            "usable_samples": len(samples),
            "first_date": history[0]["trading_date"],
            "last_date": history[-1]["trading_date"],
            "initial_training_samples": validation[0],
            "source_counts": dict(Counter(r["source"] for r in history)),
            "historical_availability_verified": sum(
                r.get("historical_availability_verified") is True for r in history
            ),
            "median_volume": float(
                np.median([r["volume"] for r in history if r.get("volume") is not None])
            ),
            "volume_below_100_count": sum(
                r.get("volume") is not None and r["volume"] < 100 for r in history
            ),
        }

        def evaluate(indices, model, alpha, samples=samples):
            scored, failures = [], []
            for index in indices:
                if time.monotonic() - started > plan["budget"]["runtime_seconds_limit"]:
                    raise TimeoutError("frozen runtime budget exhausted")
                try:
                    components, prices = predict_day(
                        samples, index, model, alpha, plan["minimum_training_samples"]
                    )
                    scored.append(
                        score_day(samples[index], components, prices, plan["flat_threshold_bps"])
                    )
                except (ValueError, OverflowError) as exc:
                    failures.append({"date": samples[index]["date"], "error": str(exc)})
            return scored, failures

        trials = []
        for model in ("split_ridge", "zero_return", "expanding_mean", "direct_close_ridge"):
            alphas = plan["alphas"] if "ridge" in model else [None]
            for alpha in alphas:
                rows, failures = evaluate(validation, model, alpha)
                metrics = summarize(rows, len(validation))
                trials.append(
                    {"model": model, "alpha": alpha, "metrics": metrics, "failures": failures}
                )
        eligible = [r for r in trials if not r["failures"]]
        candidate = min(
            (r for r in eligible if r["model"] == "split_ridge"),
            key=lambda r: r["metrics"]["close_mae_bps"],
        )
        baseline = min(
            (r for r in eligible if r["model"] != "split_ridge"),
            key=lambda r: r["metrics"]["close_mae_bps"],
        )
        direct = min(
            (r for r in eligible if r["model"] == "direct_close_ridge"),
            key=lambda r: r["metrics"]["close_mae_bps"],
        )
        selections[market] = {
            "candidate": candidate,
            "strongest_baseline": baseline,
            "direct_baseline": direct,
            "validation_trials": trials,
        }
        # Written before this market's test is scored. Never tune using its results.
        write_immutable_json(selections[market], output / f"{market}-locked-selection.json")
        test = [
            i
            for i, r in enumerate(samples)
            if split["test_start"] <= r["date"] <= split["test_end"]
        ]
        models = [
            ("split_ridge", candidate["alpha"]),
            ("zero_return", None),
            ("expanding_mean", None),
            ("direct_close_ridge", direct["alpha"]),
        ]
        results, daily = {}, {}
        for model, alpha in models:
            rows, failures = evaluate(test, model, alpha)
            daily[model] = rows
            results[model] = {
                "alpha": alpha,
                "metrics": summarize(rows, len(test)),
                "failures": failures,
                "by_month": {
                    m: summarize(
                        [r for r in rows if r["date"].startswith(m)],
                        sum(samples[i]["date"].startswith(m) for i in test),
                    )
                    for m in sorted({r["date"][:7] for r in rows})
                },
            }
        base_name = baseline["model"]
        candidate_metrics = results["split_ridge"]["metrics"]
        base_metrics = results[base_name]["metrics"]
        comparison = bootstrap_improvement(
            daily["split_ridge"], daily[base_name], plan["uncertainty"]
        )
        deterioration = {
            k: candidate_metrics[f"{k}_mae_bps"] / base_metrics[f"{k}_mae_bps"] - 1
            for k in ("open", "high", "low")
        }
        point_gate = (
            comparison["improvement"] >= plan["gates"]["close_mae_improvement_at_least"]
            and all(v <= plan["gates"]["max_OHL_mae_deterioration"] for v in deterioration.values())
            and not results["split_ridge"]["failures"]
            and candidate_metrics["scored"] == len(test)
        )
        threshold = float(np.median([samples[i]["features"][3] for i in validation]))
        regime = {
            label: summarize(
                [r for r in daily["split_ridge"] if (r["prior_range_bps"] > threshold) == high],
                sum((samples[i]["features"][3] > threshold) == high for i in test),
            )
            for label, high in (("higher_prior_range", True), ("lower_prior_range", False))
        }
        summaries[market] = {
            "split": split,
            "selected_alpha": candidate["alpha"],
            "locked_strongest_baseline": base_name,
            "test": results,
            "comparison": comparison,
            "OHL_deterioration": deterioration,
            "historical_point_gate_pass": point_gate,
            "formal_acceptance_pass": False,
            "prior_range_threshold_bps": threshold,
            "candidate_by_prior_range": regime,
        }
        details[market] = daily
        print(
            json.dumps(
                {
                    "market": market,
                    "alpha": candidate["alpha"],
                    "baseline": base_name,
                    "candidate": candidate_metrics,
                    "baseline_metrics": base_metrics,
                    "comparison": comparison,
                    "point_gate": point_gate,
                },
                ensure_ascii=False,
            )
        )
    summary = {
        "experiment": plan["experiment"],
        "scope": plan["scope"],
        "historical_asof_verified": False,
        "formal_acceptance_pass": False,
        "plan_sha256": file_hash(output / "plan.json"),
        "input_sha256": plan["input_files"],
        "quality": quality,
        "TAIEX_December_overlap_matches": matched,
        "results": summaries,
        "limitations": plan["limitations"],
        "availability_assumption": plan["availability_assumption"],
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "python": platform.python_version(),
            "packages": {k: version(k) for k in ("numpy", "scikit-learn", "pydantic")},
        },
    }
    for name, digest in plan["input_files"].items():
        if file_hash(root / name) != digest:
            raise ValueError("source file mutated during study")
    write_immutable_json(summary, output / "summary.json")
    write_immutable_json(details, output / "daily-private.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
