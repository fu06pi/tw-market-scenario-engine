"""Predeclared descriptive threshold sensitivity and reproducible error audits.

Never selects a new threshold or changes the original forecasts/policies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np

from scripts.run_night_ablation import build_paired_samples, predict
from tw_scenario.metrics import realized_direction
from tw_scenario.shadow import seal
from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import OHLC, TAIPEI, encode_components
from tw_scenario.taifex import index_rows, parse_tx_csv


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_metrics(rows, references, threshold):
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("invalid threshold")
    if len({r["date"] for r in rows}) != len(rows) or not rows:
        raise ValueError("empty/duplicate observations")
    result = {}
    for field in ("open", "close"):
        pred, actual = [], []
        for row in rows:
            ref = references[row["date"]]
            if not math.isfinite(ref) or ref <= 0:
                raise ValueError("invalid reference")
            pred.append(realized_direction((row["predicted"][field] / ref - 1) * 10000, threshold).value)
            actual.append(realized_direction((row["actual"][field] / ref - 1) * 10000, threshold).value)
        result[field] = {
            "direction_accuracy": sum(a == b for a, b in zip(pred, actual)) / len(rows),
            "actual_flat_fraction": actual.count("flat") / len(rows),
            "always_up_accuracy": actual.count("up") / len(rows),
            "always_flat_accuracy": actual.count("flat") / len(rows),
        }
    result["n"] = len(rows)
    return result


def decompose_error(reference, actual, predicted):
    truth, pred = encode_components(reference, actual), encode_components(reference, predicted)
    gap, body = (pred.g - truth.g) * 10000, (pred.b - truth.b) * 10000
    total = math.log(predicted.close / actual.close) * 10000
    if not math.isclose(gap + body, total, abs_tol=1e-8):
        raise ValueError("component decomposition failed")
    return {"gap_error_log_bp": gap, "body_error_log_bp": body, "close_error_log_bp": total,
            "dominant_component": "body" if abs(body) >= abs(gap) else "gap",
            "interpretation": "exact algebraic error location, not market/event causation"}


def main(root, public, private):
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    original = read(root / "reports/research/night-ablation-v1/plan.json")
    source = root / "reports/generated/night-ablation-v1-verified/daily-private.json"
    rows_by_market = read(source)
    policy = root / "config/targets.yaml"
    policy_hash = sha(policy)
    plan = {
        "experiment": "threshold-and-error-audit-v1", "frozen_at": datetime.now(TAIPEI).isoformat(),
        "direction_thresholds_bps": [0, 2, 5, 10, 20], "tolerance_points": [25, 50, 100, 200],
        "tolerance_bps": [10, 25, 50, 100], "phases": ["validation", "evaluation_seen"],
        "selection_rule": "none; 5bp remains unchanged; no performance-based threshold selection",
        "error_cases": "top 5 absolute close bp errors per market, fixed with_night_gap_only arm",
        "source_sha256": sha(source), "policy_sha256": policy_hash,
        "code_sha256": sha(Path(__file__)), "market_cause_inference": False,
        "evaluation_previously_seen": True, "new_independent_test": False,
        "input_hashes": original["input_files"] | original["price_inputs"],
    }
    write_immutable_json(plan, public / "diagnostic-plan.json")
    for n, h in plan["input_hashes"].items():
        if sha(root / n) != h: raise ValueError("frozen source changed")
    futures = [r for n in original["input_files"] for r in parse_tx_csv((root / n).read_bytes(), n)]
    cash = [r for n in original["price_inputs"] for r in read(root / n)]
    cash_closes = {r["trading_date"]: r["prices"]["close"] for r in cash}
    samples, _, _ = build_paired_samples(cash, futures)
    indexed = index_rows(futures)
    all_sensitivity, tolerances, audits = [], [], []
    private_cases = []
    for market, rows in samples.items():
        by_date = {r["date"]: r for r in rows}
        references = {d: r["previous_close"] for d, r in by_date.items()}
        for phase in plan["phases"]:
            arms = rows_by_market[market][phase]
            calendars = [[r["date"] for r in arm] for arm in arms.values()]
            if any(d != calendars[0] for d in calendars):
                raise ValueError("threshold arms do not share exact dates")
            for method, arm in arms.items():
                for threshold in plan["direction_thresholds_bps"]:
                    m = threshold_metrics(arm, references, threshold)
                    all_sensitivity.append({"market": market, "phase": phase, "method": method,
                                            "threshold_bps": threshold, **m})
                refs = [references[r["date"]] for r in arm]
                for field in ("open", "high", "low", "close"):
                    errors = [abs(r["predicted"][field] - r["actual"][field]) for r in arm]
                    for unit, thresholds in (("points", plan["tolerance_points"]), ("bps", plan["tolerance_bps"])):
                        values = errors if unit == "points" else [e / ref * 10000 for e, ref in zip(errors, refs)]
                        for threshold in thresholds:
                            tolerances.append({"market": market, "phase": phase, "method": method,
                                               "field": field, "unit": unit, "threshold": threshold,
                                               "n": len(values), "hit_fraction": float(np.mean(np.array(values) <= threshold))})
        arm = rows_by_market[market]["evaluation_seen"]["with_night_gap_only"]
        chosen = sorted(arm, key=lambda r: (-r["close_ae_bps"], r["date"]))[:5]
        for cached in chosen:
            sample = by_date[cached["date"]]
            index = next(i for i, r in enumerate(rows) if r["date"] == cached["date"])
            components, reproduced = predict(rows, index, "with_night_gap_only", alpha=original["alpha"])
            saved = OHLC.model_validate(cached["predicted"])
            replay = all(math.isclose(getattr(reproduced, f), getattr(saved, f), abs_tol=1e-8)
                         for f in OHLC.model_fields)
            actual = sample["actual"]
            day = date.fromisoformat(sample["date"])
            night = sample["night_feature"]
            previous_day = date.fromisoformat(night["reference_date"])
            if market == "TAIEX":
                raw_month = read(root / f"data/raw/twse/twse-taiex-{day:%Y%m}.json")
                raw_day = [r for r in raw_month["data"]
                           if tuple(map(int, r[0].split("/"))) == (day.year - 1911, day.month, day.day)]
                official = OHLC(**dict(zip(("open", "high", "low", "close"),
                                           [float(v.replace(",", "")) for v in raw_day[0][1:5]]))) if len(raw_day) == 1 else None
                # Cash has its own previous close; it need not equal the TX night reference.
                ref_ok = (sample["feature_latest_date"] == previous_day.isoformat()
                          and cash_closes[previous_day.isoformat()] == sample["previous_close"])
            else:
                official = indexed[(day, sample["contract"], "day")].prices
                ref_ok = indexed[(previous_day, sample["contract"], "day")].prices.close == sample["previous_close"]
            checks = {"official_daytime_OHLC_matches": official == actual,
                      "own_market_or_same_contract_reference": ref_ok,
                      "date_and_historical_lags_precede_target": sample["feature_latest_date"] < sample["date"],
                      "fixed_model_replay_matches": replay,
                      "component_roundtrip_coherent": components.reconstruct(sample["previous_close"]) == reproduced}
            if not all(checks.values()): raise ValueError("case audit failed; preserve discrepancy before conclusions")
            decomposition = decompose_error(sample["previous_close"], actual, saved)
            record = {"market": market, "date": sample["date"], "contract": sample["contract"],
                      "close_absolute_error_points": cached["close_ae_points"],
                      "close_absolute_error_bps": cached["close_ae_bps"], "rollover": sample["rollover"],
                      "night_age_hours": sample["night_age_hours"], "checks": checks, **decomposition,
                      "historical_availability_proven": False, "event_cause_proven": False}
            audits.append(record)
            private_cases.append({**record, "actual": actual.model_dump(), "predicted": saved.model_dump(),
                                  "previous_close": sample["previous_close"]})
    if sha(policy) != policy_hash: raise ValueError("original scoring policy changed")
    report = seal({"experiment": plan["experiment"], "direction_sensitivity": all_sensitivity,
                   "tolerance_hit_rates": tolerances, "audited_error_cases": audits,
                   "original_policy_unchanged": True, "threshold_selected": False,
                   "historical_publication_verified": False, "new_independent_test": False,
                   "audit_replays": 10, "audit_ridge_head_fits": 20, "new_hyperparameter_search": False})
    write_immutable_json(report, public / "threshold-and-error-audit.json")
    write_immutable_json(private_cases, private / "error-cases-private.json")
    with (public / "direction-sensitivity.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["market", "phase", "method", "threshold_bp", "n", "close_direction_accuracy", "actual_flat_fraction"])
        for r in all_sensitivity:
            writer.writerow([r["market"], r["phase"], r["method"], r["threshold_bps"], r["n"],
                             r["close"]["direction_accuracy"], r["close"]["actual_flat_fraction"]])
    print(json.dumps({"audited_cases": len(audits), "all_checks_pass": True,
                      "body_dominant_cases": sum(r["dominant_component"] == "body" for r in audits),
                      "policy_changed": False, "public": str(public)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.public.resolve(), args.private.resolve())
