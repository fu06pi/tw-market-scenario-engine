"""Fair point-baseline gate for saved TimesFM output-head experiments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import date
from pathlib import Path

from .backtest import load_dataset, walk_forward
from .contracts import OHLC, digest
from .head_calibration import calibrate_head_intervals
from .timesfm_head_training import development_acceptance, paired_validation_metrics

FAIR_COMPARATOR_VERSION = "fixed-point-comparator-v1"


def _validate_comparator_config(config: dict) -> None:
    required = {
        "version",
        "primary_model",
        "fallback_model",
        "representation",
        "selection_policy",
        "failure_policy",
    }
    if set(config) != required:
        raise ValueError("unexpected fair comparator config keys")
    if config["version"] != FAIR_COMPARATOR_VERSION:
        raise ValueError("unsupported fair comparator version")
    if config["selection_policy"] != "predeclared_primary_before_evaluation":
        raise ValueError("unsupported comparator selection policy")
    if config["failure_policy"] != "same_date_fallback_preserve_primary_error":
        raise ValueError("unsupported comparator failure policy")
    if config["primary_model"] == config["fallback_model"]:
        raise ValueError("primary and fallback comparator models must differ")
    if config["representation"] not in {"gbud", "relative"}:
        raise ValueError("unsupported comparator representation")


def fixed_comparator_rows(
    primary_rows: list[dict],
    fallback_rows: list[dict],
    config: dict,
) -> list[dict]:
    """Use a predeclared primary model and a deterministic same-date fallback."""
    _validate_comparator_config(config)
    if len(primary_rows) != len(fallback_rows):
        raise ValueError("primary and fallback row counts differ")
    output = []
    seen = set()
    for primary, fallback in zip(primary_rows, fallback_rows, strict=True):
        day = primary.get("date")
        if not isinstance(day, str) or day in seen or day != fallback.get("date"):
            raise ValueError("comparator rows must have identical unique ordered dates")
        seen.add(day)
        for field in ("instrument", "contract"):
            if primary.get(field) != fallback.get(field):
                raise ValueError(f"comparator {field} differs on {day}")
        if primary.get("status") == "OK":
            row = deepcopy(primary)
            row["comparator"] = {
                "model": config["primary_model"],
                "fallback_used": False,
            }
        elif fallback.get("status") == "OK":
            row = deepcopy(fallback)
            row["comparator"] = {
                "model": config["fallback_model"],
                "fallback_used": True,
                "primary_error": primary.get("error", "unknown primary failure"),
            }
        else:
            row = deepcopy(primary)
            row["comparator"] = {
                "model": config["primary_model"],
                "fallback_used": False,
                "primary_error": primary.get("error", "unknown primary failure"),
                "fallback_error": fallback.get("error", "unknown fallback failure"),
            }
        output.append(row)
    dates = [row["date"] for row in output]
    if dates != sorted(dates):
        raise ValueError("comparator dates must be chronological")
    return output


def _market_rows(saved: dict, key: str, market: str, joint: bool) -> list[dict]:
    rows = saved[key]
    if joint:
        if set(rows) != {"TAIEX", "TX"}:
            raise ValueError(f"joint fold has invalid {key} market set")
        return rows[market]
    if not isinstance(rows, list):
        raise TypeError(f"independent fold {key} must be a list")
    return rows


def _verify_saved_row(data, market: str, row: dict) -> None:
    if row.get("status") != "OK":
        return
    day = date.fromisoformat(row["date"])
    actual = data.actual(market, day)
    expected_actual = {field: getattr(actual, field) for field in OHLC}
    if (
        row.get("instrument") != market
        or row.get("contract") != data.contract(market, day)
        or row.get("previous_close") != data.reference(market, day).close
        or row.get("actual") != expected_actual
    ):
        raise ValueError("saved prediction label/reference differs from certified data")


def evaluate_saved_head_fair_gate(
    dataset_dir: Path,
    experiment_dir: Path,
    training_config: dict,
    comparator_config: dict,
    calibration_config: dict,
) -> tuple[dict, dict[str, dict[str, list[dict]]]]:
    """Rebuild a fixed comparator, calibrate both sides, and rerun every gate."""
    _validate_comparator_config(comparator_config)
    if not 2 <= int(calibration_config["minimum"]) <= int(calibration_config["window"]):
        raise ValueError("invalid fair-gate calibration window")
    experiment_dir = Path(experiment_dir)
    result_path = experiment_dir / "result.json"
    source = json.loads(result_path.read_text(encoding="utf-8"))
    identity = source["identity"]
    data, dataset_id = load_dataset(Path(dataset_dir))
    if identity["dataset_id"] != dataset_id or identity["config_hash"] != digest(
        training_config
    ):
        raise ValueError("dataset/config differs from frozen experiment")
    for field in ("timesfm_code_commit", "timesfm_weight_revision"):
        if identity[field] != training_config[field]:
            raise ValueError("pinned TimesFM revision mismatch")
    if comparator_config["primary_model"] not in training_config["stable_baselines"]:
        raise ValueError("primary comparator was not in the frozen baseline set")
    if comparator_config["fallback_model"] not in training_config["stable_baselines"]:
        raise ValueError("fallback comparator was not in the frozen baseline set")

    joint = identity["instrument"] == "JOINT"
    markets = ("TAIEX", "TX") if joint else (identity["instrument"],)
    if any(market not in {"TAIEX", "TX"} for market in markets):
        raise ValueError("unsupported experiment instrument")
    candidates = {market: [] for market in markets}
    comparators = {market: [] for market in markets}
    source_baselines = {market: [] for market in markets}
    fold_ranges = {market: [] for market in markets}
    source_fold_improvements = {market: [] for market in markets}
    fold_hashes = {}

    for index, fold in enumerate(source["fold_results"], 1):
        path = experiment_dir / f"fold-{index}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("identity_hash") != digest(identity) or saved.get("fold") != index:
            raise ValueError("fold identity mismatch")
        if saved.get("summary") != fold:
            raise ValueError("fold summary mismatch")
        checkpoint = Path(fold["checkpoint"]["path"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != fold["checkpoint"]["sha256"]:
            raise ValueError("checkpoint hash mismatch")
        expected = [
            str(day)
            for day in data.calendar.days
            if fold["validation_split"][0] <= str(day) <= fold["validation_split"][1]
        ]
        start = date.fromisoformat(fold["validation_split"][0])
        end = date.fromisoformat(fold["validation_split"][1])
        for market in markets:
            candidate = _market_rows(saved, "candidate_rows", market, joint)
            source_baseline = _market_rows(saved, "baseline_rows", market, joint)
            if [row.get("date") for row in candidate] != expected:
                raise ValueError("candidate fold dates missing, duplicated or unordered")
            if [row.get("date") for row in source_baseline] != expected:
                raise ValueError("source baseline fold dates missing, duplicated or unordered")
            primary = walk_forward(
                data,
                market,
                comparator_config["primary_model"],
                comparator_config["representation"],
                start,
                end,
                training_config,
            )
            fallback = walk_forward(
                data,
                market,
                comparator_config["fallback_model"],
                comparator_config["representation"],
                start,
                end,
                training_config,
            )
            comparator = fixed_comparator_rows(primary, fallback, comparator_config)
            if [row.get("date") for row in comparator] != expected:
                raise ValueError("rebuilt comparator dates differ from frozen fold")
            offset = len(candidates[market])
            candidates[market].extend(candidate)
            comparators[market].extend(comparator)
            source_baselines[market].extend(source_baseline)
            fold_ranges[market].append((offset, len(candidates[market])))
            result = fold["instrument_results"][market] if joint else fold
            source_fold_improvements[market].append(result["close_relative_improvement"])
        fold_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    market_reports = {}
    output_rows = {}
    for market in markets:
        for row in candidates[market] + source_baselines[market] + comparators[market]:
            _verify_saved_row(data, market, row)
        times = {
            row["date"]: data.actual(market, date.fromisoformat(row["date"])).available_at
            for row in candidates[market]
        }
        source_gate = development_acceptance(
            candidates[market],
            source_baselines[market],
            training_config,
            fold_improvements=source_fold_improvements[market],
        )
        expected_source_gate = (
            source["development_gate"]["instrument_gates"][market]
            if joint
            else source["development_gate"]
        )
        if source_gate != expected_source_gate:
            raise ValueError("saved source gate is not reproducible")
        candidate_calibrated = calibrate_head_intervals(
            candidates[market],
            times,
            window=int(calibration_config["window"]),
            minimum=int(calibration_config["minimum"]),
        )
        comparator_calibrated = calibrate_head_intervals(
            comparators[market],
            times,
            window=int(calibration_config["window"]),
            minimum=int(calibration_config["minimum"]),
        )
        fair_fold_improvements = [
            paired_validation_metrics(
                candidate_calibrated[start:end],
                comparator_calibrated[start:end],
                training_config["neutral_bps"],
            )["close_relative_improvement"]
            for start, end in fold_ranges[market]
        ]
        gate = development_acceptance(
            candidate_calibrated,
            comparator_calibrated,
            training_config,
            fold_improvements=fair_fold_improvements,
        )
        fallback_rows = [
            row for row in comparators[market] if row["comparator"].get("fallback_used")
        ]
        market_reports[market] = {
            "status": gate["status"],
            "source_gate": source_gate,
            "fair_fold_improvements": fair_fold_improvements,
            "development_gate": gate,
            "fallback": {
                "count": len(fallback_rows),
                "dates": [row["date"] for row in fallback_rows],
                "primary_errors": [row["comparator"]["primary_error"] for row in fallback_rows],
            },
            "calibration_status": {
                "candidate": dict(
                    Counter(row["calibration"]["status"] for row in candidate_calibrated)
                ),
                "comparator": dict(
                    Counter(row["calibration"]["status"] for row in comparator_calibrated)
                ),
            },
        }
        output_rows[market] = {
            "candidate": candidate_calibrated,
            "comparator": comparator_calibrated,
        }

    passed = all(item["status"] == "PASS" for item in market_reports.values())
    report = {
        "schema_version": "timesfm3-fair-point-comparator-evaluation-v1",
        "status": "DEVELOPMENT_GATE_PASS" if passed else "DEVELOPMENT_GATE_FAIL",
        "research_scope": "development_only_not_independent_holdout",
        "production_switch": False,
        "source_result": str(result_path),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "source_fold_sha256": fold_hashes,
        "source_identity": identity,
        "comparator_config": deepcopy(comparator_config),
        "calibration_config": deepcopy(calibration_config),
        "instrument_results": market_reports,
        "limitations": [
            "The old final period is development data and is not an independent holdout.",
            "Marginal interval coverage does not establish a joint OHLC path interval.",
            "A development pass would still require a newly frozen future holdout.",
        ],
    }
    return report, output_rows
