"""Build the public aggregate report for completed output-head experiments.

This script reads ignored local artifacts but writes no daily prices or licensed
raw data. Run from the repository root:

    .venv/bin/python research/summarize_head_delivery.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tw_scenario.quant.daily import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "reports/generated/quant/experiments"
EXPERIMENTS = {
    "TAIEX": {
        "context64": "dd4ef236644360b9e734",
        "context96": "6197e7f4be27ac0c8f6e",
        "context128": "7ed6c8055e40e5d8fb4e",
        "context64_train_std": "88b26bf3d30c9ded6d0d",
    },
    "TX": {
        "context64": "d635e40df84cdfad4cde",
        "context96": "601d01aa4ed209962486",
        "context128": "e48eafb8d283c39acc1d",
        "context64_train_std": "340db2b412224657c0e7",
    },
}
CALIBRATIONS = {
    "TAIEX": "head-calibration-delivery-2026-10-26/TAIEX",
    "TX": "head-calibration-delivery-2026-10-26/TX-best",
}
SNAPSHOT = EXPERIMENT_ROOT / "head-delivery-2026-10-26/snapshot-2026-10-12.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(market: str, label: str, run_id: str) -> dict:
    path = EXPERIMENT_ROOT / "timesfm3-output-head-walk-forward" / run_id / "result.json"
    result = _read(path)
    if result["identity"]["instrument"] != market:
        raise ValueError(f"instrument mismatch in {path}")
    gate = result["development_gate"]
    metrics = gate["candidate_metrics"]
    folds = []
    for fold in result["fold_results"]:
        checkpoint = Path(fold["checkpoint"]["path"])
        if _sha256(checkpoint) != fold["checkpoint"]["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        folds.append({
            "fold": fold["fold"],
            "close_relative_improvement": fold["close_relative_improvement"],
            "checkpoint_sha256": fold["checkpoint"]["sha256"],
        })
    return {
        "label": label,
        "run_id": run_id,
        "result_sha256": _sha256(path),
        "context": result["identity"]["spec"]["context"],
        "loss_normalization": result["identity"]["spec"].get("loss_normalization", "none"),
        "status": result["status"],
        "development_gate": gate["status"],
        "paired_days": gate["paired_days"],
        "forecast_coverage": metrics["forecast_coverage"],
        "failed_days": metrics["failed_days"],
        "close_relative_improvement": gate["close_relative_improvement"],
        "close_improvement_ci95": gate["block_improvement"][
            "close_relative_improvement_ci95"
        ],
        "folds": folds,
        "ohlc": metrics["ohlc"],
        "gap_mae_bps": metrics["gap_mae_bps"],
        "close_return_mae_bps": metrics["close_return_mae_bps"],
        "range_mae_bps": metrics["range_mae_bps"],
        "direction": metrics["direction"],
        "intervals": metrics["intervals"],
        "requirements": gate["requirements"],
    }


def _calibration(market: str, relative: str) -> dict:
    root = EXPERIMENT_ROOT / relative
    result_path = root / "result.json"
    artifact_path = root / "calibration-artifact.json"
    result = _read(result_path)
    artifact = _read(artifact_path)
    if result["instrument"] != market or artifact["instrument"] != market:
        raise ValueError(f"calibration instrument mismatch: {root}")
    if not result["points_unchanged"] or result["all_days"] != 240:
        raise ValueError(f"calibration did not retain all point predictions: {root}")
    gate = result["development_gate"]
    return {
        "result_sha256": _sha256(result_path),
        "artifact_sha256": _sha256(artifact_path),
        "source_result_sha256": result["source_result_sha256"],
        "development_gate": gate["status"],
        "all_days": result["all_days"],
        "counts": result["counts"],
        "points_unchanged": result["points_unchanged"],
        "coverage": gate["interval_coverage"],
        "intervals": gate["candidate_metrics"]["intervals"],
        "requirements": gate["requirements"],
    }


def build() -> dict:
    markets = {}
    for market, runs in EXPERIMENTS.items():
        candidates = [_candidate(market, label, run_id) for label, run_id in runs.items()]
        best = min(candidates, key=lambda value: value["ohlc"]["close"]["mae_points"])
        markets[market] = {
            "candidates": candidates,
            "best_research_not_promoted": best["run_id"],
            "best_selection_rule": "minimum development close MAE after every full gate failed",
            "calibration": _calibration(market, CALIBRATIONS[market]),
        }
    ledger = _read(EXPERIMENT_ROOT / "output-head-search-budget.json")
    charged = sum(attempt["charged_seconds"] for attempt in ledger["attempts"])
    snapshot = load_snapshot(SNAPSHOT)
    return {
        "schema_version": "output-head-delivery-summary-v1",
        "as_of": "2026-10-26",
        "scope": "retrospective development only; old final remains closed",
        "official_timesfm3_finetuning": False,
        "training_method": "custom frozen TimesFM 3 backbone with trainable shared output head",
        "production_switch": False,
        "overall_status": "DEVELOPMENT_GATE_FAIL",
        "production_default": "locked strongest baseline for each market",
        "markets": markets,
        "daily_snapshot": {
            "path": str(SNAPSHOT.relative_to(ROOT)),
            "file_sha256": _sha256(SNAPSHOT),
            "forecast_id": snapshot.forecast_id,
            "snapshot_hash": snapshot.snapshot_hash,
            "trading_date": str(snapshot.trading_date),
            "timeliness": snapshot.timeliness,
            "data_quality": snapshot.data_quality.status,
            "instruments": [forecast.instrument for forecast in snapshot.forecasts],
            "checkpoint_sha256": {
                forecast.instrument: forecast.model.checkpoint_id
                for forecast in snapshot.forecasts
            },
            "causal_calibration": all(
                forecast.intervals.method.startswith(
                    "causal signed price residual marginals v1"
                )
                for forecast in snapshot.forecasts
            ),
            "tamper_check": "PASS",
            "genuine_forward": False,
        },
        "wick_candidate": {
            "run_id": "d77dbc4223777b1f104e",
            "status": "INTERRUPTED_BEFORE_FIRST_FOLD_CHECKPOINT",
            "result_exists": False,
            "sample_audit_exists": (
                EXPERIMENT_ROOT
                / "timesfm3-output-head-walk-forward/d77dbc4223777b1f104e/fold-1-sample-audit.json"
            ).exists(),
            "conclusion": "no accuracy conclusion",
        },
        "budget": {
            "limit_seconds": ledger["total_limit_seconds"],
            "charged_seconds": charged,
            "attempts": len(ledger["attempts"]),
            "last_attempt_status": ledger["attempts"][-1]["status"],
            "note": "The final unclean reservation is charged in full by the frozen policy; it is not measured compute time.",
        },
        "replay": {
            "status": "FAIL",
            "tolerance_points": 0.001,
            "reason": "legacy checkpoints lack numeric runtime metadata and saved interval replay exceeded tolerance",
        },
        "limitations": [
            "No candidate passed the complete development gate in both markets.",
            "The fixed 2025-10-09 through 2026-10-08 final remains a permanent failure and was not reopened.",
            "Historical TimesFM pretraining overlap cannot be excluded; a fixed-version genuine forward holdout is still required.",
            "Prediction error is separate from trading profitability.",
        ],
    }


def markdown(report: dict) -> str:
    lines = [
        "# Output-head model delivery — 2026-10-26",
        "",
        "All results are retrospective development evidence. The complete gate remains **FAIL** and production keeps the locked baselines.",
        "",
        "| Market | Candidate | Context | Loss scale | Close MAE | Improvement | CI95 | Gate |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]
    for market, payload in report["markets"].items():
        for candidate in payload["candidates"]:
            ci = candidate["close_improvement_ci95"]
            lines.append(
                f"| {market} | {candidate['run_id']} | {candidate['context']} | "
                f"{candidate['loss_normalization']} | "
                f"{candidate['ohlc']['close']['mae_points']:.2f} | "
                f"{candidate['close_relative_improvement']:.2%} | "
                f"[{ci[0]:.2%}, {ci[1]:.2%}] | {candidate['development_gate']} |"
            )
    lines.extend(["", "## Causal interval calibration", ""])
    for market, payload in report["markets"].items():
        calibration = payload["calibration"]
        coverage = calibration["coverage"]
        lines.append(
            f"- {market}: O/H/L/C coverage "
            f"{coverage['open']:.2%}/{coverage['high']:.2%}/"
            f"{coverage['low']:.2%}/{coverage['close']:.2%}; "
            f"{calibration['all_days']} days retained; gate {calibration['development_gate']}."
        )
    lines.extend([
        "",
        "## Daily checkpoint delivery",
        "",
        f"The dual-market snapshot for {report['daily_snapshot']['trading_date']} was loaded, calibrated, sealed and read back with hash `{report['daily_snapshot']['snapshot_hash']}`. It is a late retrospective run, not genuine forward evidence.",
        "",
        "The wick candidate was interrupted before a complete fold and has no accuracy result. The earlier claim that its loss could not affect Close was rejected by the shared-parameter audit; this does not establish improvement.",
        "",
        f"Research ledger: {report['budget']['charged_seconds']:.2f}/{report['budget']['limit_seconds']:.0f} charged seconds. The last unclean reservation is not measured compute time.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "reports/quant/output-head-delivery-2026-10-26.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "reports/quant/output-head-delivery-2026-10-26.md",
    )
    args = parser.parse_args()
    report = build()
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "output": str(args.json_output)}))


if __name__ == "__main__":
    main()
