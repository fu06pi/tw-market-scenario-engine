"""Validate frozen daily outputs and export presentation evidence; never fit models.

Full daily prices stay in ignored output. Public tables contain aggregates only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.run_historical_validation import score_day, summarize
from scripts.run_night_ablation import build_paired_samples
from tw_scenario.shadow import seal
from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import OHLC, TAIPEI, encode_components
from tw_scenario.taifex import parse_tx_csv

FIELDS = ("open", "high", "low", "close")
METHODS = ("without_night", "with_night_gap_only", "night_carry_with_mean_body")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_market(arms, samples, expected_dates, published):
    if set(arms) != set(METHODS):
        raise ValueError("frozen method set differs")
    if expected_dates != sorted(set(expected_dates)) or not expected_dates:
        raise ValueError("expected calendar must be ordered, nonempty and unique")
    expected = set(expected_dates)
    if not expected <= set(samples):
        raise ValueError("source sample coverage incomplete")
    rows_out, metrics = {}, {}
    for method in METHODS:
        rows = arms[method]
        dates = [r["date"] for r in rows]
        if dates != expected_dates:
            raise ValueError("method has missing, duplicate, reordered or extra evaluation dates")
        verified = []
        for row in rows:
            sample = samples[row["date"]]
            truth = OHLC.model_validate(row["actual"])
            prediction = OHLC.model_validate(row["predicted"])
            if truth != sample["actual"]:
                raise ValueError("daily outcome differs from source-verified same-contract actual")
            # Recompute every error/direction from raw prices and the true reference.
            scored = score_day(sample, encode_components(sample["previous_close"], prediction),
                               prediction, 5.0)
            for key, value in scored.items():
                if key not in row:
                    raise ValueError(f"saved daily field missing: {key}")
                if isinstance(value, (float, int)):
                    if not math.isclose(value, row[key], rel_tol=1e-9, abs_tol=1e-8):
                        raise ValueError(f"saved daily metric disagrees: {key}")
                elif row[key] != value:
                    raise ValueError(f"saved daily identity/direction disagrees: {key}")
            verified.append({**scored, "previous_close": sample["previous_close"],
                             "contract": sample["contract"], "rollover": sample["rollover"]})
        aggregate = summarize(verified, len(expected))
        for key, value in published[method].items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isclose(aggregate[key], value, rel_tol=1e-9, abs_tol=1e-8):
                    raise ValueError(f"aggregate differs from frozen public summary: {key}")
            elif aggregate[key] != value:
                raise ValueError(f"aggregate differs from frozen public summary: {key}")
        aggregate["rmse_points"] = {
            f: float(np.sqrt(np.mean([(r["predicted"][f] - r["actual"][f]) ** 2
                                     for r in verified]))) for f in FIELDS}
        aggregate["missing_or_failed"] = len(expected) - len(verified)
        metrics[method], rows_out[method] = aggregate, verified
    return rows_out, metrics


def inventory(root):
    result = {
        "frozen_comparison": {"methods": list(METHODS), "fitted_type": "standardized Ridge / simple baseline",
                              "alpha": 0.1, "update_rule": "predeclared daily expanding past-only fit"},
        "arima": {"status": "not_evaluated", "role": "optional extension after core evidence"},
        "news": {"status": "explanation_only", "price_change_allowed": False,
                 "independent_direction_ablation": "not_evaluated"},
    }
    path = root / "reports/generated/cloud/taiex-training-pilot.json"
    if path.exists():
        pilot = read(path)
        result["timesfm_pilot"] = {
            k: pilot[k] for k in ("status", "model", "weight_revision", "scope", "context_length",
                                  "horizon", "optimizer_steps", "max_gradient_l2",
                                  "last_parameter_update_l2", "before_validation", "best_validation",
                                  "limitations")}
        result["timesfm_pilot"].update({"training_windows": len(pilot["train_dates"]),
                                         "validation_windows": len(pilot["validation_dates"]),
                                         "artifact_sha256": sha(path),
                                         "same_179_day_comparison": False,
                                         "weights_exported": False})
    else:
        result["timesfm_pilot"] = {"status": "local_artifact_unavailable", "same_179_day_comparison": False}
    return result


def main(root, output, public):
    if output.exists() or public.exists():
        raise FileExistsError("use new versioned output/public directories; never replace research")
    folder = root / "reports/research/night-ablation-v1"
    plan, frozen, manifest = (read(folder / n) for n in ("plan.json", "summary.json", "manifest.json"))
    for name, digest in manifest["public_files"].items():
        if sha(folder / name) != digest:
            raise ValueError("frozen public experiment document changed")
    for name, digest in (plan["input_files"] | plan["price_inputs"]).items():
        if sha(root / name) != digest:
            raise ValueError("original source input changed")
    for name, digest in frozen["implementation_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("frozen joining/scoring implementation changed")
    private_path = root / manifest["private_daily_results"]
    private = read(private_path)
    cash = [r for name in plan["price_inputs"] for r in read(root / name)]
    futures = [r for name in plan["input_files"]
               for r in parse_tx_csv((root / name).read_bytes(), name)]
    samples, exclusions, opportunities = build_paired_samples(cash, futures)
    # Historical chart context is also checked against official TWSE December JSON.
    december_path = root / "data/raw/twse/twse-taiex-202512.json"
    december = {}
    for row in read(december_path)["data"]:
        y, m, d = map(int, row[0].split("/"))
        december[f"{y + 1911:04d}-{m:02d}-{d:02d}"] = OHLC(**dict(zip(
            FIELDS, [float(v.replace(",", "")) for v in row[1:5]])))
    output.mkdir(parents=True)
    public.mkdir(parents=True)
    metrics, presentation = {}, {}
    for market in plan["markets"]:
        expected = [d for d in opportunities[market]
                    if plan["split"]["evaluation_start"] <= d <= plan["split"]["evaluation_end"]]
        sample_map = {r["date"]: r for r in samples[market]}
        published = {m: frozen["results"][market]["results"]["evaluation_seen"][m]["metrics"]
                     for m in METHODS}
        verified, aggregate = validate_market(private[market]["evaluation_seen"], sample_map,
                                              expected, published)
        context = [r for r in samples[market] if "2025-12-01" <= r["date"] <= "2025-12-31"][-20:]
        for row in context:
            if market == "TAIEX" and december[row["date"]] != row["actual"]:
                raise ValueError("historical chart context differs from official TWSE")
        metrics[market] = {
            "expected": len(expected), "first": expected[0], "last": expected[-1],
            "initial_training_samples": frozen["results"][market]["initial_training_samples"],
            "methods": aggregate, "exclusions": exclusions[market],
            "published_close_ablation": frozen["results"][market]["closing_comparison"],
            "published_open_ablation": frozen["results"][market]["opening_comparison"],
            "failure_lists": {m: frozen["results"][market]["results"]["evaluation_seen"][m]["failures"]
                              for m in METHODS},
        }
        selected = verified["with_night_gap_only"]
        error_cases = sorted(selected, key=lambda r: (-r["close_ae_bps"], r["date"]))[:5]
        presentation[market] = {
            "daily": verified, "metrics": aggregate,
            "context": [{"date": r["date"], "actual": r["actual"].model_dump(),
                         "contract": r["contract"]} for r in context],
            "largest_close_errors": error_cases,
        }
    metadata = {
        "schema_version": "academic-evidence-v1", "created_at": datetime.now(TAIPEI).isoformat(),
        "study": "Frozen night-ablation results, independently recomputed; no model fitting",
        "historical_publication_verified": False, "new_independent_test": False,
        "evaluation_previously_seen": True, "formal_acceptance": False,
        "new_prospective_evidence_generated": 0, "model_refit_performed": False,
        "neutral_threshold_bps": 5.0, "split": plan["split"], "inventory": inventory(root),
        "price_horizon": "same-day daytime OHLC, all features before open",
        "plot_context": "last 20 December 2025 validation observations, not newly fitted training",
        "sources_sha256": {str(p.relative_to(root)): sha(p) for p in (
            folder / "plan.json", folder / "summary.json", private_path, december_path)},
    }
    write_immutable_json(seal({**metadata, "markets": metrics}), public / "metrics.json")
    write_immutable_json(seal({**metadata, "markets": presentation}), output / "evidence-private.json")
    with (public / "price-comparison.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["market", "method", "field", "expected", "n",
            "failed_or_missing", "mae_points", "rmse_points", "mae_bps", "rmse_bps"])
        writer.writeheader()
        for market, record in metrics.items():
            for method, m in record["methods"].items():
                for f in FIELDS:
                    writer.writerow({"market": market, "method": method, "field": f,
                        "expected": record["expected"], "n": m["scored"],
                        "failed_or_missing": m["missing_or_failed"], "mae_points": m[f"{f}_mae_points"],
                        "rmse_points": m["rmse_points"][f], "mae_bps": m[f"{f}_mae_bps"],
                        "rmse_bps": m[f"{f}_rmse_bps"]})
    print(json.dumps({"private_evidence": str(output / "evidence-private.json"),
                      "public_aggregate": str(public), "markets": {
                          m: d["expected"] for m, d in metrics.items()}, "model_fits": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve(), args.public.resolve())
