"""Paired retrospective TX-night feature ablation. Never writes pre-open forecasts."""

from __future__ import annotations

import argparse
import math
import platform
import time
from collections import Counter
from datetime import date, datetime
from datetime import time as clock
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import OHLC, TAIPEI, Components, encode_components, fit_ridge_head
from tw_scenario.taifex import index_rows, night_feature, parse_tx_csv

if __package__:
    from .run_historical_validation import file_hash, read_json, score_day, summarize
else:
    from run_historical_validation import file_hash, read_json, score_day, summarize


def build_paired_samples(taiex_history, futures):
    indexed = index_rows(futures)
    regular_dates = sorted({r.trading_date for r in futures if r.session == "day"})
    cash = {
        date.fromisoformat(r["trading_date"]): OHLC.model_validate(r["prices"])
        for r in taiex_history
    }
    if len(cash) != len(taiex_history):
        raise ValueError("duplicate TAIEX dates")
    cash_dates = sorted(cash)
    target_dates = {
        "TAIEX": cash_dates,
        "TX_selected_monthly": [
            d for d in regular_dates if date(2025, 1, 1) <= d <= date(2026, 9, 30)
        ],
    }
    samples = {name: [] for name in target_dates}
    exclusions = {name: [] for name in target_dates}
    for market, dates in target_dates.items():
        previous_selected = None
        for day in dates:
            try:
                feature = night_feature(
                    day, futures, datetime.combine(day, clock(8, 30), tzinfo=TAIPEI)
                )
                prior_date = date.fromisoformat(feature["reference_date"])
                if market == "TAIEX":
                    pos = cash_dates.index(day)
                    if pos < 2:
                        raise ValueError("insufficient prior cash sessions for lag features")
                    if cash_dates[pos - 1] != prior_date:
                        raise ValueError("cash/futures previous trading dates disagree")
                    actual, previous = cash[day], cash[cash_dates[pos - 1]]
                    earlier_close = cash[cash_dates[pos - 2]].close
                    reference = previous.close
                else:
                    pos = regular_dates.index(prior_date)
                    if pos < 1:
                        raise ValueError("missing earlier futures day for lag features")
                    contract = feature["contract"]
                    current = indexed.get((day, contract, "day"))
                    prior = indexed.get((prior_date, contract, "day"))
                    earlier = indexed.get((regular_dates[pos - 1], contract, "day"))
                    if any(
                        row is None or row.prices is None or not row.volume
                        for row in (current, prior, earlier)
                    ):
                        raise ValueError(
                            "missing selected-contract actual or same-contract lag prices"
                        )
                    actual, previous = current.prices, prior.prices
                    reference, earlier_close = previous.close, earlier.prices.close
                    if reference != feature["previous_day_close"]:
                        raise ValueError("night and target reference prices disagree")
                lag = encode_components(earlier_close, previous)
                samples[market].append(
                    {
                        "date": day.isoformat(),
                        "market": market,
                        "contract": feature["contract"],
                        "feature_latest_date": prior_date.isoformat(),
                        "previous_close": reference,
                        "actual": actual,
                        "target": encode_components(reference, actual),
                        "features": [
                            math.expm1(lag.g) * 10000,
                            math.expm1(lag.g + lag.b) * 10000,
                            math.expm1(lag.b) * 10000,
                            (previous.high - previous.low) / earlier_close * 10000,
                        ],
                        "night_return_bps": feature["night_return_bps"],
                        "night_age_hours": feature["age_hours"],
                        "night_feature": feature,
                        "rollover": previous_selected is not None
                        and previous_selected != feature["contract"],
                    }
                )
                previous_selected = feature["contract"]
            except (ValueError, KeyError) as exc:
                exclusions[market].append({"date": day.isoformat(), "reason": str(exc)})
    return samples, exclusions, {k: [d.isoformat() for d in v] for k, v in target_dates.items()}


def predict(samples, index, model, alpha=0.1, minimum=60):
    if index < minimum:
        raise ValueError("insufficient shared past samples")
    current, past = samples[index], samples[:index]
    if any(row["date"] >= current["date"] for row in past):
        raise ValueError("training date leakage")
    if current["feature_latest_date"] >= current["date"]:
        raise ValueError("lag feature date leakage")
    targets = [r["target"] for r in past]
    u, d = (float(np.mean([getattr(t, k) for t in targets])) for k in ("u", "d"))
    if model == "night_carry_with_mean_body":
        g = math.log1p(current["night_return_bps"] / 10000)
        b = float(np.mean([t.b for t in targets]))
    elif model in ("without_night", "with_night_gap_only"):
        x = [
            r["features"][:2] + ([r["night_return_bps"]] if model == "with_night_gap_only" else [])
            for r in past
        ]
        query = current["features"][:2] + (
            [current["night_return_bps"]] if model == "with_night_gap_only" else []
        )
        g, _ = fit_ridge_head(x, [t.g for t in targets], query, alpha)
        # Identical fit and inputs in both arms, so body cannot change.
        b, _ = fit_ridge_head(
            [r["features"][2:] for r in past],
            [t.b for t in targets],
            current["features"][2:],
            alpha,
        )
    else:
        raise ValueError("unknown ablation arm")
    components = Components(g=g, b=b, u=u, d=d)
    return components, components.reconstruct(current["previous_close"])


def paired_interval(before, after, metric, settings):
    a = {r["date"]: r[metric] for r in before}
    b = {r["date"]: r[metric] for r in after}
    if set(a) != set(b) or len(a) != len(before) or len(b) != len(after):
        raise ValueError("ablation arms must have exactly identical scored dates")
    dates = sorted(a)
    if not dates:
        raise ValueError("no paired observations")
    first, second = np.array([a[d] for d in dates]), np.array([b[d] for d in dates])
    if np.mean(first) <= 0:
        return {"paired_n": len(dates), "improvement": None, "interval_95": None}
    rng = np.random.default_rng(settings["seed"])
    block = min(len(dates), settings["block_length"])
    estimates = []
    for _ in range(settings["draws"]):
        starts = rng.integers(0, len(dates) - block + 1, size=math.ceil(len(dates) / block))
        indices = np.concatenate([np.arange(i, i + block) for i in starts])[: len(dates)]
        denominator = float(np.mean(first[indices]))
        if denominator > 0:
            estimates.append(1 - float(np.mean(second[indices])) / denominator)
    return {
        "metric": metric,
        "paired_n": len(dates),
        "improvement": 1 - float(np.mean(second)) / float(np.mean(first)),
        "interval_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        **settings,
    }


def main(root, output):
    started = time.monotonic()
    plan = read_json(output / "plan.json")
    for name, digest in (plan["input_files"] | plan["price_inputs"]).items():
        if file_hash(root / name) != digest:
            raise ValueError("input bytes changed after plan freeze")
    futures = []
    for name in plan["input_files"]:
        futures.extend(parse_tx_csv((root / name).read_bytes(), name))
    taiex = []
    for name in plan["price_inputs"]:
        taiex.extend(read_json(root / name))
    samples_by_market, exclusions, opportunities = build_paired_samples(taiex, futures)
    split = plan["split"]
    public, private = {}, {}
    for market, samples in samples_by_market.items():
        phases = {
            "validation": [
                i
                for i, r in enumerate(samples)
                if split["validation_start"] <= r["date"] <= split["validation_end"]
            ],
            "evaluation_seen": [
                i
                for i, r in enumerate(samples)
                if split["evaluation_start"] <= r["date"] <= split["evaluation_end"]
            ],
        }
        if (
            not all(phases.values())
            or min(min(x) for x in phases.values()) < plan["minimum_training_samples"]
        ):
            raise ValueError("not enough common training/validation/evaluation observations")
        results = {}
        private[market] = {}
        max_body_difference = 0.0
        for phase, indices in phases.items():
            results[phase] = {}
            private[market][phase] = {}
            bodies = {}
            for model in plan["models"]:
                rows, failures = [], []
                for index in indices:
                    if time.monotonic() - started > plan["budget_seconds"]:
                        raise TimeoutError("night ablation runtime budget exhausted")
                    sample = samples[index]
                    try:
                        components, prices = predict(
                            samples, index, model, plan["alpha"], plan["minimum_training_samples"]
                        )
                        row = score_day(sample, components, prices, plan["flat_threshold_bps"])
                        row.update(
                            night_age_hours=sample["night_age_hours"],
                            rollover=sample["rollover"],
                            contract=sample["contract"],
                            body_prediction_log=components.b,
                        )
                        rows.append(row)
                        if model == "without_night":
                            bodies[sample["date"]] = components.b
                        if model == "with_night_gap_only":
                            max_body_difference = max(
                                max_body_difference, abs(components.b - bodies[sample["date"]])
                            )
                    except (ValueError, OverflowError) as exc:
                        failures.append({"date": sample["date"], "error": str(exc)})
                results[phase][model] = {
                    "metrics": summarize(rows, len(indices)),
                    "failures": failures,
                }
                private[market][phase][model] = rows
        before = private[market]["evaluation_seen"]["without_night"]
        after = private[market]["evaluation_seen"]["with_night_gap_only"]
        opening = paired_interval(before, after, "open_ae_bps", plan["bootstrap"])
        closing = paired_interval(before, after, "close_ae_bps", plan["bootstrap"])
        before_metrics = results["evaluation_seen"]["without_night"]["metrics"]
        after_metrics = results["evaluation_seen"]["with_night_gap_only"]["metrics"]
        deterioration = {
            k: after_metrics[f"{k}_mae_bps"] / before_metrics[f"{k}_mae_bps"] - 1
            for k in ("open", "high", "low")
        }
        expected = [
            d
            for d in opportunities[market]
            if split["evaluation_start"] <= d <= split["evaluation_end"]
        ]
        excluded = [
            r
            for r in exclusions[market]
            if split["evaluation_start"] <= r["date"] <= split["evaluation_end"]
        ]
        complete = len(phases["evaluation_seen"]) == len(expected) and all(
            not result["failures"] and result["metrics"]["scored"] == len(expected)
            for result in results["evaluation_seen"].values()
        )
        if max_body_difference > 1e-12:
            raise ValueError("body head changed between ablation arms")
        grouped = {
            group: {} for group in ("fresh_night", "weekend_or_holiday", "rollover", "non_rollover")
        }
        for group, group_metrics in grouped.items():

            def in_group(r, group=group):
                if group == "fresh_night":
                    return r["night_age_hours"] < 24
                if group == "weekend_or_holiday":
                    return r["night_age_hours"] >= 24
                return r["rollover"] if group == "rollover" else not r["rollover"]

            for model in ("without_night", "with_night_gap_only"):
                selected = [r for r in private[market]["evaluation_seen"][model] if in_group(r)]
                group_metrics[model] = summarize(selected, len(selected))
        public[market] = {
            "raw_target_dates": len(opportunities[market]),
            "matched_rows": len(samples),
            "initial_training_samples": phases["validation"][0],
            "all_source_exclusions": exclusions[market],
            "evaluation_opportunities": len(expected),
            "evaluation_exclusions": excluded,
            "evaluation_common_rows": len(phases["evaluation_seen"]),
            "paired_coverage_of_target_dates": len(phases["evaluation_seen"]) / len(expected),
            "all_arms_complete": complete,
            "selected_month_counts": dict(Counter(r["contract"] for r in samples)),
            "results": results,
            "opening_comparison": opening,
            "closing_comparison": closing,
            "max_body_prediction_difference": max_body_difference,
            "OHL_deterioration": deterioration,
            "primary_opening_usefulness": complete
            and opening["improvement"] is not None
            and opening["improvement"] >= 0.05
            and opening["interval_95"][0] > 0,
            "conditional_price_gate_vs_without_night": complete
            and closing["improvement"] is not None
            and closing["improvement"] >= 0.05
            and all(v <= 0.05 for v in deterioration.values()),
            "relative_close_MAE_change_vs_prelisted_night_baseline": (
                after_metrics["close_mae_bps"]
                / results["evaluation_seen"]["night_carry_with_mean_body"]["metrics"][
                    "close_mae_bps"
                ]
                - 1
                if complete
                else None
            ),
            "formal_acceptance_pass": False,
            "by_session_context": grouped,
        }
        print(
            market,
            "common",
            public[market]["evaluation_common_rows"],
            "/",
            len(expected),
            "open improvement",
            opening,
            "close improvement",
            closing,
            flush=True,
        )
        for model, r in results["evaluation_seen"].items():
            print(
                model,
                {
                    k: r["metrics"][k]
                    for k in ["open_mae_bps", "gap_accuracy", "close_mae_bps", "close_accuracy"]
                },
                flush=True,
            )
    summary = {
        "experiment": plan["experiment"],
        "scope": plan["scope"],
        "plan_sha256": file_hash(output / "plan.json"),
        "formal_acceptance_pass": False,
        "historical_publication_verified": False,
        "new_independent_test_evidence": False,
        "results": public,
        "limitations": plan["limitations"],
        "source_rule_url": plan["source_rule_url"],
        "implementation_sha256": {
            name: file_hash(root / name)
            for name in [
                "src/tw_scenario/taifex.py",
                "scripts/run_night_ablation.py",
                "src/tw_scenario/split_forecast.py",
                "scripts/run_historical_validation.py",
            ]
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ["numpy", "scikit-learn", "pydantic"]},
        },
    }
    for name, digest in (plan["input_files"] | plan["price_inputs"]).items():
        if file_hash(root / name) != digest:
            raise ValueError("input mutated during ablation")
    write_immutable_json(summary, output / "summary.json")
    write_immutable_json(private, output / "daily-private.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
