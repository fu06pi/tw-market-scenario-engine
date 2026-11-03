"""Build the public aggregate for the fixed night-regression residual-head run."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from tw_scenario.quant.backtest import load_dataset
from tw_scenario.quant.head_blend import causal_blend_rows
from tw_scenario.quant.timesfm_head_training import (
    development_acceptance,
    paired_validation_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce"
RUN_ID = "73b3f1aadccb646372be"
RUN_ROOT = ROOT / "reports/generated/quant/model-search-v2/timesfm3-output-head-walk-forward" / RUN_ID
FAIR_ROOT = ROOT / "reports/generated/quant/model-search-v2" / f"fair-gate-{RUN_ID}"
CONFIG = ROOT / "config/quant/timesfm_output_head_residual.json"
BLEND_CONFIG = ROOT / "config/quant/head_causal_blend.json"
LEDGER = ROOT / "reports/generated/quant/model-search-v2/output-head-search-budget.json"
JSON_OUTPUT = ROOT / "reports/quant/residual-head-result-2026-10-27.json"
MD_OUTPUT = ROOT / "reports/quant/residual-head-result-2026-10-27.md"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_summary(gate: dict) -> dict:
    candidate = gate["candidate_metrics"]
    baseline = gate["baseline_metrics"]
    return {
        "status": gate["status"],
        "paired_days": gate["paired_days"],
        "paired_coverage": gate["paired_coverage"],
        "close_relative_improvement": gate["close_relative_improvement"],
        "fold_improvements": gate["fold_improvements"],
        "close_improvement_ci95": gate["block_improvement"][
            "close_relative_improvement_ci95"
        ],
        "candidate_ohlc_mae_points": {
            field: candidate["ohlc"][field]["mae_points"]
            for field in ("open", "high", "low", "close")
        },
        "comparator_ohlc_mae_points": {
            field: baseline["ohlc"][field]["mae_points"]
            for field in ("open", "high", "low", "close")
        },
        "candidate_intervals": candidate["intervals"],
        "comparator_intervals": baseline["intervals"],
        "requirements": gate["requirements"],
    }


def build() -> dict:
    source_path = RUN_ROOT / "result.json"
    fair_path = FAIR_ROOT / "result.json"
    predictions_path = FAIR_ROOT / "predictions.json"
    source = _read(source_path)
    fair = _read(fair_path)
    predictions = _read(predictions_path)
    config = _read(CONFIG)
    blend_config = _read(BLEND_CONFIG)
    ledger = _read(LEDGER)
    data, dataset_id = load_dataset(DATASET)

    if source["identity"]["dataset_id"] != dataset_id:
        raise ValueError("residual run dataset mismatch")
    if fair["source_result_sha256"] != _sha256(source_path):
        raise ValueError("fair gate does not bind the residual source result")
    if set(predictions) != {"TAIEX", "TX"}:
        raise ValueError("fair predictions do not contain both markets")

    checkpoints = []
    fold_sizes = []
    for fold in source["fold_results"]:
        checkpoint = Path(fold["checkpoint"]["path"])
        if _sha256(checkpoint) != fold["checkpoint"]["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        checkpoints.append({
            "fold": fold["fold"],
            "sha256": fold["checkpoint"]["sha256"],
            "best_epoch": fold["training"]["best_epoch"],
            "epochs_completed": fold["training"]["epochs_completed"],
        })
        start, end = map(date.fromisoformat, fold["validation_split"])
        fold_sizes.append(sum(start <= day <= end for day in data.calendar.days))
    if fold_sizes != [80, 80, 80]:
        raise ValueError("residual run validation folds changed")

    markets = {}
    for market in ("TAIEX", "TX"):
        fair_payload = fair["instrument_results"][market]
        candidate = predictions[market]["candidate"]
        comparator = predictions[market]["comparator"]
        if len(candidate) != 240 or len(comparator) != 240:
            raise ValueError("fair gate did not retain every expected date")

        # Optimistic, explicit production-style safety diagnostic: a failed
        # residual correction emits its same-date anchor. This is not used to
        # rewrite the frozen candidate or claim a pass.
        safe_candidate = [
            baseline if row["status"] != "OK" and baseline["status"] == "OK" else row
            for row, baseline in zip(candidate, comparator, strict=True)
        ]
        available = {
            row["date"]: data.actual(market, date.fromisoformat(row["date"])).available_at
            for row in comparator
        }
        blended = causal_blend_rows(safe_candidate, comparator, available, blend_config)
        fold_improvements = []
        offset = 0
        for size in fold_sizes:
            fold_improvements.append(
                paired_validation_metrics(
                    blended[offset : offset + size],
                    comparator[offset : offset + size],
                    config["neutral_bps"],
                )["close_relative_improvement"]
            )
            offset += size
        blend_gate = development_acceptance(
            blended,
            comparator,
            config,
            fold_improvements=fold_improvements,
        )
        failed = [row["date"] for row in candidate if row["status"] != "OK"]
        alphas = [row["blend"]["alpha"] for row in blended if row["status"] == "OK"]
        markets[market] = {
            "raw_source_gate": _gate_summary(
                source["development_gate"]["instrument_gates"][market]
            ),
            "fair_gate": {
                **_gate_summary(fair_payload["development_gate"]),
                "comparator": fair["comparator_config"],
                "calibration": fair["calibration_config"],
                "candidate_failed_dates": failed,
                "comparator_fallback": fair_payload["fallback"],
            },
            "causal_shrinkage_diagnostic": {
                **_gate_summary(blend_gate),
                "candidate_failure_policy": (
                    "same-date comparator for diagnostic only; frozen rows unchanged"
                ),
                "alpha_mean": sum(alphas) / len(alphas),
                "alpha_last": blended[-1]["blend"],
            },
        }

    charged = sum(float(item["charged_seconds"]) for item in ledger["attempts"])
    return {
        "schema_version": "timesfm3-night-regression-residual-head-summary-v1",
        "as_of": "2026-10-27",
        "status": "DEVELOPMENT_GATE_FAIL",
        "production_switch": False,
        "research_scope": "development_only_not_independent_holdout",
        "run_id": RUN_ID,
        "dataset_id": dataset_id,
        "source_result_sha256": _sha256(source_path),
        "fair_result_sha256": _sha256(fair_path),
        "fair_predictions_sha256": _sha256(predictions_path),
        "config_sha256": _sha256(CONFIG),
        "training_method": (
            "causal night-regression GBUD residuals; frozen TimesFM 3 backbone; "
            "custom trainable shared output head"
        ),
        "official_timesfm3_finetuning": False,
        "elapsed_seconds": source["elapsed_seconds"],
        "peak_rss_mb": source["peak_rss_mb"],
        "checkpoints": checkpoints,
        "markets": markets,
        "budget": {
            "limit_seconds": ledger["total_limit_seconds"],
            "charged_seconds": charged,
            "remaining_seconds": ledger["total_limit_seconds"] - charged,
            "attempts": len(ledger["attempts"]),
        },
        "conclusion": {
            "development_gate_passed": False,
            "final_gate_passed": False,
            "default_model": "fixed night-regression comparator with range fallback",
            "search_action": "STOP_NO_FURTHER_DEVELOPMENT_TUNING",
            "reason": (
                "Residual-head and causal-shrinkage Close improvements are negative "
                "against the fixed fair comparator in both markets."
            ),
        },
        "limitations": [
            "The old final period is development data and was not reopened as an independent test.",
            "The one failed JOINT day remains in the frozen result and is disclosed.",
            "Historical TimesFM pretraining overlap cannot be excluded.",
            "Prediction error is separate from trading profitability.",
        ],
    }


def markdown(report: dict) -> str:
    lines = [
        "# Night-regression residual head result — 2026-10-27",
        "",
        "The fixed JOINT residual-head experiment completed three folds and 240 intended dates, but the fair development gate is **FAIL** for both markets.",
        "",
        "| Market | Candidate Close MAE | Fair comparator MAE | Improvement | Folds | CI95 | Gate |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for market, payload in report["markets"].items():
        gate = payload["fair_gate"]
        folds = " / ".join(f"{value:.2%}" for value in gate["fold_improvements"])
        ci = gate["close_improvement_ci95"]
        lines.append(
            f"| {market} | {gate['candidate_ohlc_mae_points']['close']:.2f} | "
            f"{gate['comparator_ohlc_mae_points']['close']:.2f} | "
            f"{gate['close_relative_improvement']:.2%} | {folds} | "
            f"[{ci[0]:.2%}, {ci[1]:.2%}] | `{gate['status']}` |"
        )
    lines.extend(["", "## Causal shrinkage diagnostic", ""])
    for market, payload in report["markets"].items():
        shrink = payload["causal_shrinkage_diagnostic"]
        lines.append(
            f"- **{market}**: Close improvement {shrink['close_relative_improvement']:.2%}; "
            f"folds {' / '.join(f'{value:.2%}' for value in shrink['fold_improvements'])}; "
            f"CI95 [{shrink['close_improvement_ci95'][0]:.2%}, "
            f"{shrink['close_improvement_ci95'][1]:.2%}]; gate `{shrink['status']}`."
        )
    lines.extend([
        "",
        "The shrinkage calculation uses only labels published before each 08:30 cutoff. For the single failed JOINT date it optimistically emits the same-date comparator; the frozen failure row is unchanged. This still does not improve either market.",
        "",
        "## Training and stop decision",
        "",
        f"The run used {report['elapsed_seconds']:.2f} seconds and peaked at {report['peak_rss_mb']:.2f} MiB. Three checkpoint hashes are recorded in the JSON aggregate. The phase-2 ledger has {report['budget']['remaining_seconds']:.2f} seconds left, but direct, blended and residual-head routes all fail the fixed comparator. The predeclared no-gain rule therefore stops further tuning on these development dates.",
        "",
        "The production default remains the fixed night-regression comparator with range fallback. `production_switch=false`; no final or profitability claim is made.",
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/tw-scenario quant timesfm-head-walk-forward \\",
        "  data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce \\",
        "  --instrument JOINT --config config/quant/timesfm_output_head_residual.json \\",
        "  --output-root reports/generated/quant/model-search-v2 --cache-dir data/raw/timesfm3",
        "```",
        "",
        "The generated artifact is immutable and will be reused rather than retrained when the identity matches.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    result = build()
    JSON_OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    MD_OUTPUT.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "json": str(JSON_OUTPUT.relative_to(ROOT)),
        "markdown": str(MD_OUTPUT.relative_to(ROOT)),
    }))
