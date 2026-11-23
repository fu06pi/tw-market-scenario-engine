"""Report attempts, missing labels, marginal coverage and degradation, without cutover.

Local output is diagnostic. Authentic remote artifact timing must be checked
separately before any formal forward-accuracy claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.run_historical_validation import score_day, summarize
from scripts.run_night_ablation import paired_interval
from tw_scenario.intervals import FIELDS, interval_score, summarize_intervals
from tw_scenario.policy import ScoringPolicy
from tw_scenario.shadow import now
from tw_scenario.snapshot import verify_document, write_immutable_json
from tw_scenario.split_forecast import OHLC, Components, ObservedBar, encode_components


def monitor(runs, actual_records):
    actuals = {}
    for raw in actual_records:
        actual = ObservedBar.model_validate(raw)
        if actual.available_at > now():
            raise ValueError("actual availability is in the future")
        identity = (actual.trading_date.isoformat(), actual.market, actual.contract)
        if identity in actuals:
            raise ValueError("duplicate actual identity")
        actuals[identity] = actual
    groups = defaultdict(lambda: {"expected": 0, "rows": [], "intervals": defaultdict(list)})
    failures, pending, duplicate_dates = [], [], set()
    for directory in sorted(runs):
        run_groups = defaultdict(lambda: {"expected": 0, "rows": [], "intervals": defaultdict(list)})
        run_pending = []
        try:
            if (directory / "failure.json").exists():
                failures.append({"run": str(directory), "failure": json.loads(
                    (directory / "failure.json").read_text())})
                continue
            forecast = json.loads((directory / "forecast.json").read_text())
            if not verify_document(forecast).valid:
                raise ValueError("forecast hash mismatch")
            policy = ScoringPolicy.model_validate(forecast["scoring_policy"])
            identity = (forecast["trading_date"], forecast["checkpoint_hash"])
            if identity in duplicate_dates:
                raise ValueError("duplicate forecast date/checkpoint; no cherry-picked rerun")
            receipt = json.loads((directory / "local-receipt.json").read_text())
            if receipt["snapshot_hash"] != forecast["snapshot_hash"]:
                raise ValueError("local receipt hash mismatch")
            if datetime.fromisoformat(receipt["local_persisted_at"]) >= datetime.fromisoformat(
                forecast["deadline"]
            ):
                raise ValueError("late persistence")
            for market, data in forecast["markets"].items():
                inputs = data["inputs"]
                actual_market = "TX" if market == "TX_selected_monthly" else market
                actual = actuals.get((forecast["trading_date"], actual_market, inputs["contract"]))
                for method, predicted in data["models"].items():
                    key = (forecast["checkpoint_hash"], market, method, data["default_baseline"])
                    group = run_groups[key]
                    group["expected"] += 1
                    if actual is None:
                        run_pending.append({"date": forecast["trading_date"], "market": market,
                                            "method": method, "reason": "actual unavailable"})
                        continue
                    prices = OHLC.model_validate(predicted["prices"])
                    components = Components.model_validate(predicted["components"])
                    if components.reconstruct(inputs["previous_close"]) != prices:
                        raise ValueError("saved components/prices mismatch")
                    actual_ohlc = OHLC(**{k: getattr(actual, k) for k in FIELDS})
                    sample = {**inputs, "actual": actual_ohlc,
                              "target": encode_components(inputs["previous_close"], actual_ohlc)}
                    group["rows"].append(score_day(sample, components, prices,
                                                   policy.flat_threshold_bps))
                    for field in FIELDS:
                        band = predicted["marginal_80"][field]
                        lower, upper = band["lower"], band["upper"]
                        truth = getattr(actual, field)
                        scale = 10000 / inputs["previous_close"]
                        group["intervals"][field].append({
                            "date": inputs["date"], "coverage": int(lower <= truth <= upper),
                            "width_bps": (upper - lower) * scale,
                            "interval_score_bps": interval_score(lower, upper, truth) * scale,
                        })
            duplicate_dates.add(identity)
            pending.extend(run_pending)
            for key, run_group in run_groups.items():
                groups[key]["expected"] += run_group["expected"]
                groups[key]["rows"].extend(run_group["rows"])
                for field, values in run_group["intervals"].items():
                    groups[key]["intervals"][field].extend(values)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"run": str(directory), "reason": str(exc)})
    report_groups = []
    for (checkpoint, market, method, baseline), group in sorted(groups.items()):
        group["rows"].sort(key=lambda r: r["date"])
        scored = summarize(group["rows"], group["expected"])
        marginal = {f: summarize_intervals(group["intervals"][f]) for f in FIELDS}
        alerts = []
        if len(group["rows"]) < 30:
            alerts.append("insufficient forward observations (<30); no coverage/pass conclusion")
        else:
            if any(not 0.75 <= s["coverage"] <= 0.85 for s in marginal.values()):
                alerts.append("marginal coverage outside 75–85%; review calibration")
        paired = None
        base = groups.get((checkpoint, market, baseline, baseline))
        if (base and group["rows"]
                and len(group["rows"]) == group["expected"] == base["expected"]
                and len(base["rows"]) == base["expected"]):
            paired = paired_interval(base["rows"], group["rows"], "close_ae_bps",
                                     {"draws": 2000, "block_length": 5, "seed": 20261104})
            if scored["close_mae_bps"] > np.mean([r["close_ae_bps"] for r in base["rows"]]):
                alerts.append("close error exceeds locked baseline; keep baseline")
        report_groups.append({"checkpoint": checkpoint, "market": market, "method": method,
                              "baseline": baseline, "point_metrics": scored,
                              "marginal_80": marginal, "close_paired_comparison": paired,
                              "alerts": alerts})
    return {"recorded_at": now().isoformat(), "attempted_runs": len(runs),
            "successful_local_runs": len(duplicate_dates),
            "failed_run_fraction": len(failures) / len(runs) if runs else None,
            "failed_runs": failures, "pending_labels": pending, "groups": report_groups,
            "formal_accuracy_eligible": False, "remote_timing_verification": "pending",
            "production_switch": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="parent of individually immutable shadow run directories")
    parser.add_argument("--actuals", type=Path, required=True,
                        help="JSON list of verified ObservedBar contracts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = monitor([p for p in args.runs.iterdir() if p.is_dir()], json.loads(args.actuals.read_text()))
    write_immutable_json(report, args.output)
