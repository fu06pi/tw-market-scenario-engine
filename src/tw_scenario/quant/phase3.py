"""Low-parameter causal residual research before any further TimesFM adaptation."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from .backtest import _git_state, load_dataset
from .contracts import OHLC, Prices, at, digest, write_once
from .forecast import predict
from .head_calibration import calibrate_head_intervals
from .timesfm_head_training import (
    development_acceptance,
    expanding_folds,
    joint_development_acceptance,
    paired_validation_metrics,
)


@dataclass(frozen=True)
class ResidualRecord:
    day: date
    features: np.ndarray
    target_bps: np.ndarray
    anchor: Prices
    actual: Prices
    previous_close: float
    contract: str
    label_available_at: datetime
    anchor_model: str
    primary_error: str | None = None


@dataclass(frozen=True)
class RidgeState:
    alpha: float
    window: int
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    training_rows: int
    training_start: date
    training_end: date


def _validate_config(config: dict) -> None:
    required = {
        "version", "seed", "train_start", "development_end", "future_holdout_start",
        "cutoff", "deadline", "timezone", "representation", "anchor_model",
        "anchor_fallback_model", "feature_contract", "target", "walk_forward_folds",
        "fold_validation_sessions", "minimum_train_sessions", "inner_validation_sessions",
        "minimum_fit_rows", "training_windows", "ridge_alphas",
        "max_abs_correction_bps", "window", "min_history", "calibration_window",
        "min_calibration", "neutral_bps", "promotion_close_improvement",
        "promotion_min_coverage", "bootstrap_block", "bootstrap_draws",
        "budget_cpu_seconds", "production_switch", "old_final_policy",
        "selection_policy", "future_holdout_policy",
    }
    optional = {"training_target_clip_bps"}
    if not required.issubset(config) or not set(config).issubset(required | optional):
        missing = sorted(required - set(config))
        extra = sorted(set(config) - required - optional)
        raise ValueError(f"phase3 config keys differ; missing={missing}, extra={extra}")
    versions = {
        "phase3-causal-price-residual-ridge-v1": "actual_minus_anchor_ohlc_bps",
        "phase3-causal-component-residual-ridge-v1": (
            "actual_minus_anchor_oc_extensions_bps"
        ),
        "phase3-causal-robust-component-residual-ridge-v1": (
            "actual_minus_anchor_oc_extensions_bps"
        ),
    }
    if versions.get(config["version"]) != config["target"]:
        raise ValueError("unsupported phase3 version/target pair")
    if (config["cutoff"], config["deadline"], config["timezone"]) != (
        "08:30", "08:40", "Asia/Taipei"
    ):
        raise ValueError("phase3 cutoff contract changed")
    if config["representation"] != "gbud":
        raise ValueError("phase3 anchor must use GBUD")
    if (config["anchor_model"], config["anchor_fallback_model"]) != (
        "night_regression", "range"
    ):
        raise ValueError("phase3 comparator must remain night_regression/range")
    if config["feature_contract"] != "verified_27_preopen_features_v1":
        raise ValueError("unsupported phase3 feature contract")
    if config["production_switch"] is not False:
        raise ValueError("phase3 cannot enable production")
    if config["old_final_policy"] != "development_only_never_independent_retest":
        raise ValueError("old final policy changed")
    if config["selection_policy"] != (
        "nested_pre_fold_close_mae_subject_to_ohl_noninferiority"
    ):
        raise ValueError("unsupported nested selection policy")
    if config["future_holdout_policy"] != "locked_checkpoint_only_no_reselection":
        raise ValueError("future holdout policy changed")
    if date.fromisoformat(config["development_end"]) >= date.fromisoformat(
        config["future_holdout_start"]
    ):
        raise ValueError("future holdout must begin after development")
    integers = (
        "walk_forward_folds", "fold_validation_sessions", "minimum_train_sessions",
        "inner_validation_sessions", "minimum_fit_rows", "window", "min_history",
        "calibration_window", "min_calibration", "bootstrap_block", "bootstrap_draws",
    )
    if any(type(config[key]) is not int or config[key] < 1 for key in integers):
        raise ValueError("phase3 integer limits must be positive")
    if not 2 <= config["min_calibration"] <= config["calibration_window"]:
        raise ValueError("invalid phase3 calibration limits")
    windows = config["training_windows"]
    alphas = config["ridge_alphas"]
    if (
        not isinstance(windows, list) or not windows
        or any(type(value) is not int or value < config["minimum_fit_rows"] for value in windows)
        or len(set(windows)) != len(windows)
    ):
        raise ValueError("invalid phase3 training windows")
    if (
        not isinstance(alphas, list) or not alphas
        or any(isinstance(value, bool) or float(value) <= 0 for value in alphas)
        or len({float(value) for value in alphas}) != len(alphas)
    ):
        raise ValueError("invalid phase3 ridge alphas")
    if float(config["max_abs_correction_bps"]) <= 0:
        raise ValueError("phase3 correction clip must be positive")
    robust = config["version"] == "phase3-causal-robust-component-residual-ridge-v1"
    if robust != ("training_target_clip_bps" in config):
        raise ValueError("robust phase3 version requires its frozen training target clip")
    if robust and float(config["training_target_clip_bps"]) <= 0:
        raise ValueError("phase3 training target clip must be positive")
    if float(config["budget_cpu_seconds"]) <= 0:
        raise ValueError("phase3 budget must be positive")


def _anchor(data, instrument: str, day: date, config: dict) -> tuple[Prices, str, str | None]:
    try:
        point, _, _, _ = predict(
            data, instrument, day, config["anchor_model"], config["representation"], config
        )
        return point, config["anchor_model"], None
    except Exception as primary:  # noqa: BLE001 - fixed fallback retains the primary error
        point, _, _, _ = predict(
            data,
            instrument,
            day,
            config["anchor_fallback_model"],
            config["representation"],
            config,
        )
        return point, config["anchor_fallback_model"], f"{type(primary).__name__}: {primary}"


def _materialize_records(data, instrument: str, days: list[date], config: dict):
    records: list[ResidualRecord] = []
    audit = []
    feature_count = len(data.research_feature_names())
    for day in days:
        try:
            features = data.research_features(instrument, day).astype(float)
            if features.shape != (feature_count,) or not np.isfinite(features).all():
                raise ValueError("invalid verified pre-open features")
            anchor, anchor_model, primary_error = _anchor(data, instrument, day, config)
            actual_bar = data.actual(instrument, day)
            actual = Prices(**{field: getattr(actual_bar, field) for field in OHLC})
            previous_close = float(data.reference(instrument, day).close)
            if config["target"] == "actual_minus_anchor_ohlc_bps":
                target_bps = (actual.array() - anchor.array()) / previous_close * 10_000
            else:
                anchor_upper = anchor.high - max(anchor.open, anchor.close)
                anchor_lower = min(anchor.open, anchor.close) - anchor.low
                actual_upper = actual.high - max(actual.open, actual.close)
                actual_lower = min(actual.open, actual.close) - actual.low
                target_bps = np.array([
                    actual.open - anchor.open,
                    actual.close - anchor.close,
                    actual_upper - anchor_upper,
                    actual_lower - anchor_lower,
                ]) / previous_close * 10_000
            if not np.isfinite(target_bps).all():
                raise ValueError("non-finite price residual target")
            raw_target_bps = target_bps.copy()
            if "training_target_clip_bps" in config:
                clip = float(config["training_target_clip_bps"])
                target_bps = np.clip(target_bps, -clip, clip)
            records.append(
                ResidualRecord(
                    day=day,
                    features=features,
                    target_bps=target_bps,
                    anchor=anchor,
                    actual=actual,
                    previous_close=previous_close,
                    contract=data.contract(instrument, day),
                    label_available_at=actual_bar.available_at,
                    anchor_model=anchor_model,
                    primary_error=primary_error,
                )
            )
            audit.append({
                "date": str(day),
                "status": "OK",
                "anchor_model": anchor_model,
                "primary_error": primary_error,
                "target_clipped": bool(np.any(target_bps != raw_target_bps)),
            })
        except Exception as exc:  # noqa: BLE001 - every unavailable date remains audited
            audit.append({
                "date": str(day),
                "status": "INELIGIBLE",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return records, audit


def _fit_ridge(records: list[ResidualRecord], *, alpha: float, window: int) -> RidgeState:
    selected = records[-window:]
    if not selected:
        raise ValueError("cannot fit phase3 Ridge without records")
    x = np.stack([record.features for record in selected])
    y = np.stack([record.target_bps for record in selected])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("phase3 train-only scaler is invalid")
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit((x - mean) / scale, y)
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if coef.shape != (4, x.shape[1]) or intercept.shape != (4,):
        raise RuntimeError("unexpected multi-output Ridge shape")
    return RidgeState(
        alpha=float(alpha),
        window=int(window),
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
        training_rows=len(selected),
        training_start=selected[0].day,
        training_end=selected[-1].day,
    )


def _corrected_prices(
    record: ResidualRecord,
    state: RidgeState,
    max_abs_correction_bps: float,
    target: str = "actual_minus_anchor_ohlc_bps",
) -> tuple[Prices, np.ndarray]:
    standardized = (record.features - state.mean) / state.scale
    correction = standardized @ state.coef.T + state.intercept
    correction = np.clip(
        np.asarray(correction, dtype=float),
        -float(max_abs_correction_bps),
        float(max_abs_correction_bps),
    )
    scaled = record.previous_close * correction / 10_000
    if target == "actual_minus_anchor_ohlc_bps":
        values = record.anchor.array() + scaled
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("phase3 correction produced non-positive price")
        open_, high, low, close = values
        point = Prices(
            open=float(open_),
            high=float(max(high, open_, close)),
            low=float(min(low, open_, close)),
            close=float(close),
        )
    elif target == "actual_minus_anchor_oc_extensions_bps":
        open_ = record.anchor.open + scaled[0]
        close = record.anchor.close + scaled[1]
        anchor_upper = record.anchor.high - max(record.anchor.open, record.anchor.close)
        anchor_lower = min(record.anchor.open, record.anchor.close) - record.anchor.low
        upper = max(0.0, anchor_upper + scaled[2])
        lower = max(0.0, anchor_lower + scaled[3])
        if not np.isfinite([open_, close, upper, lower]).all() or min(open_, close) <= 0:
            raise ValueError("phase3 component correction produced invalid price")
        point = Prices(
            open=float(open_),
            high=float(max(open_, close) + upper),
            low=float(min(open_, close) - lower),
            close=float(close),
        )
    else:
        raise ValueError("unsupported phase3 reconstruction target")
    return point, correction


def _correction_fields(target: str) -> tuple[str, str, str, str]:
    if target == "actual_minus_anchor_ohlc_bps":
        return OHLC
    if target == "actual_minus_anchor_oc_extensions_bps":
        return ("open", "close", "upper_extension", "lower_extension")
    raise ValueError("unsupported phase3 correction target")


def _row(
    record: ResidualRecord,
    point: Prices,
    correction: np.ndarray | None,
    target: str = "actual_minus_anchor_ohlc_bps",
) -> dict:
    correction_fields = _correction_fields(target)
    return {
        "date": str(record.day),
        "instrument": "TAIEX" if not record.contract else "TX",
        "contract": record.contract,
        "status": "OK",
        "previous_close": record.previous_close,
        "predicted": point.model_dump(),
        "actual": record.actual.model_dump(),
        "interval": None,
        "anchor": {
            "model": record.anchor_model,
            "primary_model": "night_regression",
            "fallback_model": "range",
            "fallback_used": record.anchor_model == "range",
            "primary_error": record.primary_error,
        },
        "correction_bps": (
            {
                field: float(correction[index])
                for index, field in enumerate(correction_fields)
            }
            if correction is not None
            else {field: 0.0 for field in correction_fields}
        ),
    }


def _failed_row(instrument: str, day: date, contract: str | None, exc: Exception) -> dict:
    return {
        "date": str(day),
        "instrument": instrument,
        "contract": contract,
        "status": "FAILED",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _anchor_row(record: ResidualRecord) -> dict:
    return _row(record, record.anchor, None)


def _candidate_rows(
    records: list[ResidualRecord],
    validation_days: list[date],
    *,
    alpha: float,
    window: int,
    minimum_fit_rows: int,
    max_abs_correction_bps: float,
    target: str,
    instrument: str,
) -> list[dict]:
    by_day = {record.day: record for record in records}
    output = []
    for day in validation_days:
        record = by_day.get(day)
        if record is None:
            output.append(_failed_row(instrument, day, None, ValueError("missing phase3 record")))
            continue
        try:
            prior = [candidate for candidate in records if candidate.day < day]
            if len(prior) < minimum_fit_rows:
                raise ValueError(
                    f"insufficient phase3 fit rows: {len(prior)} < {minimum_fit_rows}"
                )
            state = _fit_ridge(prior, alpha=alpha, window=window)
            point, correction = _corrected_prices(
                record, state, max_abs_correction_bps, target
            )
            output.append(_row(record, point, correction, target))
        except Exception as exc:  # noqa: BLE001 - preserve every validation failure
            output.append(_failed_row(instrument, day, record.contract, exc))
    return output


def _baseline_rows(
    records: list[ResidualRecord], validation_days: list[date], instrument: str
) -> list[dict]:
    by_day = {record.day: record for record in records}
    output = []
    for day in validation_days:
        record = by_day.get(day)
        if record is None:
            output.append(_failed_row(instrument, day, None, ValueError("missing anchor record")))
        else:
            output.append(_anchor_row(record))
    return output


def _select_spec(
    records: list[ResidualRecord],
    boundary: date,
    config: dict,
    instrument: str,
) -> dict:
    prior = [record for record in records if record.day < boundary]
    inner_sessions = int(config["inner_validation_sessions"])
    if len(prior) <= inner_sessions + int(config["minimum_fit_rows"]):
        raise ValueError("insufficient records for nested phase3 selection")
    inner_days = [record.day for record in prior[-inner_sessions:]]
    baseline = _baseline_rows(prior, inner_days, instrument)
    evaluated = []
    for window in config["training_windows"]:
        for alpha in config["ridge_alphas"]:
            candidate = _candidate_rows(
                prior,
                inner_days,
                alpha=float(alpha),
                window=int(window),
                minimum_fit_rows=int(config["minimum_fit_rows"]),
                max_abs_correction_bps=float(config["max_abs_correction_bps"]),
                target=config["target"],
                instrument=instrument,
            )
            comparison = paired_validation_metrics(
                candidate, baseline, float(config["neutral_bps"])
            )
            candidate_metrics = comparison.get("candidate_metrics", {})
            baseline_metrics = comparison.get("baseline_metrics", {})
            degradation = {}
            for field in ("open", "high", "low"):
                candidate_mae = candidate_metrics.get("ohlc", {}).get(field, {}).get(
                    "mae_points"
                )
                baseline_mae = baseline_metrics.get("ohlc", {}).get(field, {}).get(
                    "mae_points"
                )
                degradation[field] = (
                    candidate_mae / baseline_mae - 1
                    if candidate_mae is not None and baseline_mae and baseline_mae > 0
                    else None
                )
            safe = (
                comparison["paired_coverage"] >= float(config["promotion_min_coverage"])
                and all(value is not None and value <= 0.05 for value in degradation.values())
            )
            close_mae = candidate_metrics.get("ohlc", {}).get("close", {}).get(
                "mae_points", float("inf")
            )
            evaluated.append({
                "alpha": float(alpha),
                "window": int(window),
                "safe_ohl": safe,
                "paired_coverage": comparison["paired_coverage"],
                "close_mae_points": close_mae,
                "close_relative_improvement": comparison["close_relative_improvement"],
                "ohl_degradation": degradation,
            })
    selected = min(
        evaluated,
        key=lambda item: (
            not item["safe_ohl"],
            item["close_mae_points"],
            item["window"],
            item["alpha"],
        ),
    )
    return {
        "selection_boundary": str(boundary),
        "inner_validation": [str(inner_days[0]), str(inner_days[-1])],
        "selected": selected,
        "candidates": evaluated,
    }


def _state_payload(
    state: RidgeState,
    *,
    instrument: str,
    dataset_id: str,
    config: dict,
    feature_names: tuple[str, ...],
    calibration_rows: list[dict],
    label_available_at: dict[str, datetime],
) -> dict:
    residuals = []
    for row in calibration_rows:
        if row["status"] != "OK":
            continue
        residuals.append({
            "date": row["date"],
            "available_at": label_available_at[row["date"]].isoformat(),
            "normalized_signed_residual": {
                field: (
                    row["actual"][field] - row["predicted"][field]
                ) / row["previous_close"]
                for field in OHLC
            },
        })
    residuals = residuals[-int(config["calibration_window"]):]
    return {
        "schema_version": "phase3-price-residual-ridge-checkpoint-v1",
        "instrument": instrument,
        "dataset_id": dataset_id,
        "config_hash": digest(config),
        "feature_names": list(feature_names),
        "alpha": state.alpha,
        "window": state.window,
        "mean": state.mean.tolist(),
        "scale": state.scale.tolist(),
        "coef": state.coef.tolist(),
        "intercept": state.intercept.tolist(),
        "training_rows": state.training_rows,
        "training_start": str(state.training_start),
        "training_label_end": str(state.training_end),
        "max_abs_correction_bps": float(config["max_abs_correction_bps"]),
        "target": config["target"],
        "anchor_model": "night_regression",
        "anchor_fallback_model": "range",
        "calibration": {
            "window": int(config["calibration_window"]),
            "minimum": int(config["min_calibration"]),
            "residuals": residuals,
        },
        "future_holdout_start": config["future_holdout_start"],
        "research_only": True,
        "production_switch": False,
    }


def run_phase3_price_residual_ridge(
    dataset_dir: Path, config_path: Path, output_root: Path
) -> Path:
    """Run the frozen nested walk-forward experiment and save real Ridge checkpoints."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _validate_config(config)
    data, dataset_id = load_dataset(Path(dataset_dir))
    development_end = date.fromisoformat(config["development_end"])
    eligible_days = [
        day for day in data.calendar.days
        if date.fromisoformat(config["train_start"]) <= day <= development_end
    ]
    folds = expanding_folds(
        eligible_days,
        folds=int(config["walk_forward_folds"]),
        validation_sessions=int(config["fold_validation_sessions"]),
        minimum_train=int(config["minimum_train_sessions"]),
    )
    identity = {
        "name": "phase3_causal_price_residual_ridge",
        "dataset_id": dataset_id,
        "config_hash": digest(config),
        "code_version": _git_state(Path(config_path).parent.parent.parent),
        "feature_names": list(data.research_feature_names()),
        "folds": [
            {
                "train": [str(train[0]), str(train[-1])],
                "validation": [str(validation[0]), str(validation[-1])],
            }
            for train, validation in folds
        ],
        "future_holdout_start": config["future_holdout_start"],
        "old_final_reused_as_independent_test": False,
    }
    experiment_dir = Path(output_root) / "phase3-price-residual-ridge" / digest(identity)[:20]
    result_path = experiment_dir / "result.json"
    if result_path.exists():
        return result_path
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(config["budget_cpu_seconds"])
    markets = {}
    predictions = {}
    for instrument in ("TAIEX", "TX"):
        if time.monotonic() >= deadline:
            raise TimeoutError("phase3 experiment budget exhausted before market start")
        records, audit = _materialize_records(data, instrument, eligible_days, config)
        all_candidates: list[dict] = []
        all_baselines: list[dict] = []
        fold_results = []
        fold_ranges = []
        for fold_index, (_, validation_days) in enumerate(folds, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("phase3 experiment budget exhausted during folds")
            selection = _select_spec(records, validation_days[0], config, instrument)
            selected = selection["selected"]
            candidate = _candidate_rows(
                records,
                validation_days,
                alpha=selected["alpha"],
                window=selected["window"],
                minimum_fit_rows=int(config["minimum_fit_rows"]),
                max_abs_correction_bps=float(config["max_abs_correction_bps"]),
                target=config["target"],
                instrument=instrument,
            )
            baseline = _baseline_rows(records, validation_days, instrument)
            comparison = paired_validation_metrics(
                candidate, baseline, float(config["neutral_bps"])
            )
            start = len(all_candidates)
            all_candidates.extend(candidate)
            all_baselines.extend(baseline)
            fold_ranges.append((start, len(all_candidates)))
            fold_results.append({
                "fold": fold_index,
                "validation": [str(validation_days[0]), str(validation_days[-1])],
                "selection": selection,
                "comparison": comparison,
            })
        label_times = {
            str(record.day): record.label_available_at for record in records
            if str(record.day) in {row["date"] for row in all_candidates}
        }
        calibrated_candidate = calibrate_head_intervals(
            all_candidates,
            label_times,
            window=int(config["calibration_window"]),
            minimum=int(config["min_calibration"]),
        )
        calibrated_baseline = calibrate_head_intervals(
            all_baselines,
            label_times,
            window=int(config["calibration_window"]),
            minimum=int(config["min_calibration"]),
        )
        fold_improvements = [
            paired_validation_metrics(
                calibrated_candidate[start:end],
                calibrated_baseline[start:end],
                float(config["neutral_bps"]),
            )["close_relative_improvement"]
            for start, end in fold_ranges
        ]
        gate = development_acceptance(
            calibrated_candidate,
            calibrated_baseline,
            config,
            fold_improvements=fold_improvements,
        )
        final_selection = _select_spec(
            records, development_end.replace(day=development_end.day) + (date.resolution),
            config, instrument,
        )
        selected = final_selection["selected"]
        usable = [record for record in records if record.day <= development_end]
        final_state = _fit_ridge(
            usable, alpha=selected["alpha"], window=selected["window"]
        )
        checkpoint = _state_payload(
            final_state,
            instrument=instrument,
            dataset_id=dataset_id,
            config=config,
            feature_names=data.research_feature_names(),
            calibration_rows=calibrated_candidate,
            label_available_at=label_times,
        )
        checkpoint_path = experiment_dir / f"checkpoint-{instrument.lower()}.json"
        write_once(checkpoint_path, checkpoint)
        markets[instrument] = {
            "status": gate["status"],
            "materialization": {
                "requested": len(eligible_days),
                "usable": len(records),
                "failed": len(eligible_days) - len(records),
                "audit": audit,
            },
            "folds": fold_results,
            "development_gate": gate,
            "final_checkpoint_selection": final_selection,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "bytes": checkpoint_path.stat().st_size,
            },
        }
        predictions[instrument] = {
            "candidate": calibrated_candidate,
            "comparator": calibrated_baseline,
        }
    combined_gate = joint_development_acceptance({
        market: result["development_gate"] for market, result in markets.items()
    })
    result = {
        "schema_version": "phase3-price-residual-ridge-result-v1",
        "status": (
            "DEVELOPMENT_GATE_PASS_AWAITING_FUTURE_HOLDOUT"
            if combined_gate["status"] == "PASS"
            else "DEVELOPMENT_GATE_FAIL"
        ),
        "identity": identity,
        "training_scope": (
            "market-specific causal Ridge on verified pre-open features and price residuals"
        ),
        "timesfm_training_performed": False,
        "markets": markets,
        "development_gate": combined_gate,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "future_holdout_opened": False,
        "production_switch": False,
    }
    write_once(experiment_dir / "predictions.json", predictions)
    write_once(result_path, result)
    return result_path


def _load_checkpoint(path: Path, data, dataset_id: str, config: dict, instrument: str) -> dict:
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        checkpoint.get("schema_version") != "phase3-price-residual-ridge-checkpoint-v1"
        or checkpoint.get("instrument") != instrument
        or checkpoint.get("dataset_id") != dataset_id
        or checkpoint.get("config_hash") != digest(config)
        or checkpoint.get("feature_names") != list(data.research_feature_names())
        or checkpoint.get("research_only") is not True
        or checkpoint.get("production_switch") is not False
    ):
        raise ValueError("phase3 checkpoint provenance mismatch")
    return checkpoint


def predict_phase3_day(
    dataset_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    instrument: str,
    forecast_day: date,
) -> dict:
    """Load a locked Phase-3 checkpoint for one future pre-open research forecast."""
    if instrument not in {"TAIEX", "TX"}:
        raise ValueError("phase3 daily instrument must be TAIEX or TX")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _validate_config(config)
    data, dataset_id = load_dataset(Path(dataset_dir))
    checkpoint = _load_checkpoint(
        Path(checkpoint_path), data, dataset_id, config, instrument
    )
    if forecast_day < date.fromisoformat(checkpoint["future_holdout_start"]):
        raise ValueError("phase3 daily forecast is before the frozen future holdout")
    if date.fromisoformat(checkpoint["training_label_end"]) >= forecast_day:
        raise ValueError("phase3 checkpoint training labels overlap forecast day")
    features = data.research_features(instrument, forecast_day).astype(float)
    anchor, anchor_model, primary_error = _anchor(data, instrument, forecast_day, config)
    reference = float(data.reference(instrument, forecast_day).close)
    synthetic = ResidualRecord(
        day=forecast_day,
        features=features,
        target_bps=np.full(4, np.nan),
        anchor=anchor,
        actual=anchor,
        previous_close=reference,
        contract=data.contract(instrument, forecast_day),
        label_available_at=at(forecast_day, "18:00"),
        anchor_model=anchor_model,
        primary_error=primary_error,
    )
    state = RidgeState(
        alpha=float(checkpoint["alpha"]),
        window=int(checkpoint["window"]),
        mean=np.asarray(checkpoint["mean"], dtype=float),
        scale=np.asarray(checkpoint["scale"], dtype=float),
        coef=np.asarray(checkpoint["coef"], dtype=float),
        intercept=np.asarray(checkpoint["intercept"], dtype=float),
        training_rows=int(checkpoint["training_rows"]),
        training_start=date.fromisoformat(checkpoint["training_start"]),
        training_end=date.fromisoformat(checkpoint["training_label_end"]),
    )
    point, correction = _corrected_prices(
        synthetic,
        state,
        float(checkpoint["max_abs_correction_bps"]),
        checkpoint["target"],
    )
    cutoff = at(forecast_day)
    residuals = [
        item for item in checkpoint["calibration"]["residuals"]
        if date.fromisoformat(item["date"]) < forecast_day
        and datetime.fromisoformat(item["available_at"]) <= cutoff
    ][-int(checkpoint["calibration"]["window"]):]
    if len(residuals) < int(checkpoint["calibration"]["minimum"]):
        raise ValueError("phase3 checkpoint has insufficient causal calibration residuals")
    values = np.stack([
        [item["normalized_signed_residual"][field] for field in OHLC]
        for item in residuals
    ])
    lower, upper = np.quantile(values, [0.1, 0.9], axis=0, method="linear")
    point_values = point.array()
    lower = np.minimum(point_values, point_values + reference * lower)
    upper = np.maximum(point_values, point_values + reference * upper)
    if (lower <= 0).any() or not np.isfinite(upper).all():
        raise ValueError("phase3 daily interval is invalid")
    return {
        "schema_version": "phase3-price-residual-ridge-daily-v1",
        "date": str(forecast_day),
        "instrument": instrument,
        "contract": synthetic.contract,
        "cutoff": at(forecast_day).isoformat(),
        "previous_close": reference,
        "predicted": point.model_dump(),
        "interval": {
            field: [float(lower[index]), float(upper[index])]
            for index, field in enumerate(OHLC)
        },
        "correction_bps": {
            field: float(correction[index])
            for index, field in enumerate(_correction_fields(checkpoint["target"]))
        },
        "anchor": {
            "model": anchor_model,
            "fallback_used": anchor_model == "range",
            "primary_error": primary_error,
        },
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "calibration_samples": len(residuals),
        "research_only": True,
        "production_switch": False,
    }
