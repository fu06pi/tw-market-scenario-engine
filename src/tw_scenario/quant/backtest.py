from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from .contracts import OHLC, TZ, Calendar, at, digest, encode, write_once
from .evaluation import metrics
from .forecast import Dataset, calibrated_interval, predict
from .providers import load_bars
from .quality import quality_report


class DatasetQualityError(RuntimeError):
    """The immutable input dataset is unsafe for model comparison."""


def _row(data, instrument, day, prices, interval, previous, contract):
    actual = data.actual(instrument, day)
    return {
        "date": str(day),
        "instrument": instrument,
        "contract": contract,
        "status": "OK",
        "previous_close": previous,
        "predicted": prices.model_dump(),
        "actual": {name: getattr(actual, name) for name in OHLC},
        "interval": interval,
    }


def walk_forward(data: Dataset, instrument: str, model: str, representation: str,
                 start: date, end: date, config: dict) -> list[dict]:
    if config["calibration_window"] < 1:
        raise ValueError("calibration_window must be positive")
    residual_records = []
    output = []
    eligible_days = [day for day in data.calendar.days if day <= end]
    output_index = next(
        (index for index, day in enumerate(eligible_days) if day >= start),
        len(eligible_days),
    )
    # Calibration only needs pre-output forecasts whose labels can enter the
    # fixed trailing residual window. Include an extra history-length buffer so
    # isolated feature/session gaps cannot empty the warm-up silently.
    warmup_sessions = config["calibration_window"] + config.get("min_history", 0)
    warmup_index = max(0, output_index - warmup_sessions)
    for day in eligible_days[warmup_index:]:
        try:
            cutoff = at(day, config.get("cutoff", "08:30"))
            eligible_residuals = [
                record["residual"]
                for record in residual_records
                if record["target_date"] < day and record["available_at"] <= cutoff
            ][-config["calibration_window"] :]
            prices, target, previous, _ = predict(
                data, instrument, day, model, representation, config
            )
            interval = calibrated_interval(
                target,
                previous,
                representation,
                eligible_residuals,
                config["min_calibration"],
            )
            actual = data.actual(instrument, day)
            actual_target = encode(actual, previous, representation)
            if day >= start:
                output.append(_row(
                    data, instrument, day, prices, interval, previous,
                    data.contract(instrument, day),
                ))
            residual_records.append({
                "target_date": day,
                "available_at": actual.available_at,
                "residual": actual_target - target,
            })
        except Exception as exc:  # noqa: BLE001 - every failed forecast day must be retained
            if day >= start:
                output.append({
                    "date": str(day), "instrument": instrument,
                    "contract": _safe_contract(data, instrument, day),
                    "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                })
    return output


def _safe_contract(data, instrument, day):
    try:
        return data.contract(instrument, day)
    except ValueError:
        return None


def load_dataset(dataset_dir: Path) -> tuple[Dataset, str]:
    manifest = _read_manifest(dataset_dir)
    calendar = Calendar.model_validate_json(
        (dataset_dir / "calendar.json").read_text(encoding="utf-8")
    )
    bars = load_bars(dataset_dir / "bars.jsonl")
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise DatasetQualityError("dataset manifest has no dataset_id")
    if manifest.get("rows") != len(bars):
        raise DatasetQualityError(
            f"manifest row count {manifest.get('rows')} != loaded rows {len(bars)}"
        )
    bars_hash = digest([bar.model_dump(mode="json") for bar in bars])
    expected_hash = manifest.get("bars_hash", dataset_id)
    if expected_hash != bars_hash:
        raise DatasetQualityError("bars.jsonl content hash does not match immutable manifest")
    return Dataset(bars, calendar), dataset_id


def _read_manifest(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DatasetQualityError("dataset manifest must be a JSON object")
    return manifest


def _select_baseline(
    validation: dict,
    instrument: str,
    expected_dates: list[str],
    min_coverage: float,
    neutral_bps: float,
) -> tuple[dict, dict]:
    if not 0 < min_coverage <= 1:
        raise ValueError("min_validation_coverage must be in (0, 1]")
    expected = set(expected_dates)
    audit = {}
    eligible = []
    for key, result in validation.items():
        if not key.startswith(f"{instrument}/"):
            continue
        rows = [row for row in result["rows"] if row["date"] in expected]
        row_dates = [row["date"] for row in rows]
        duplicate_dates = sorted(
            day for day, count in Counter(row_dates).items() if count > 1
        )
        successful = {
            row["date"]: row for row in rows
            if row["status"] == "OK" and row["date"] not in duplicate_dates
        }
        coverage = len(successful) / len(expected) if expected else 0.0
        reason = None
        if duplicate_dates:
            reason = "duplicate_validation_dates"
        elif coverage < min_coverage:
            reason = "below_min_validation_coverage"
        elif not result["metrics"].get("ohlc"):
            reason = "no_validation_metrics"
        audit[key] = {
            "expected_days": len(expected),
            "success_days": len(successful),
            "coverage": coverage,
            "duplicate_dates": duplicate_dates,
            "eligible": reason is None,
            "ineligible_reason": reason,
        }
        if reason is None:
            eligible.append((key, successful))
    if not eligible:
        raise RuntimeError(
            f"no {instrument} baseline met validation coverage {min_coverage:.1%}"
        )

    paired_dates = sorted(set.intersection(*(set(rows) for _, rows in eligible)))
    paired_coverage = len(paired_dates) / len(expected) if expected else 0.0
    if paired_coverage < min_coverage:
        raise RuntimeError(
            f"{instrument} paired validation coverage {paired_coverage:.1%} "
            f"is below {min_coverage:.1%}"
        )

    scored = []
    for key, rows in eligible:
        paired_metrics = metrics([rows[day] for day in paired_dates], neutral_bps)
        close_mae = paired_metrics["ohlc"]["close"]["mae_points"]
        audit[key]["paired_close_mae"] = close_mae
        scored.append((close_mae, key, paired_metrics))
    close_mae, key, paired_metrics = min(scored, key=lambda item: (item[0], item[1]))
    _, representation, model = key.split("/")
    return {
        "model": model,
        "representation": representation,
        "validation_close_mae": close_mae,
        "validation_expected_days": len(expected),
        "validation_paired_days": len(paired_dates),
        "validation_paired_coverage": paired_coverage,
        "validation_paired_dates_hash": digest(paired_dates),
        "validation_paired_metrics": paired_metrics,
    }, audit


def run_baseline_suite(dataset_dir: Path, config_path: Path, output_root: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data, dataset_id = load_dataset(dataset_dir)
    manifest = _read_manifest(dataset_dir)
    requested_start = date.fromisoformat(config["train_start"])
    requested_end = date.fromisoformat(config["test_end"])
    data_quality = quality_report(
        data.bars,
        data.calendar,
        requested_start,
        requested_end,
        manifest=manifest,
    )
    if data_quality["status"] != "PASS_RETROSPECTIVE":
        raise DatasetQualityError(
            "dataset quality gate failed: "
            f"missing={len(data_quality['missing'])}, "
            f"duplicates={len(data_quality['duplicates'])}, "
            f"issues={len(data_quality['issues'])}, "
            f"calendar_gaps={len(data_quality['calendar_gaps'])}, "
            f"manifest_issues={len(data_quality['manifest_issues'])}"
        )
    validation_start = date.fromisoformat(config["validation_start"])
    test_start = date.fromisoformat(config["test_start"])
    test_end = date.fromisoformat(config["test_end"])
    validation_dates = [
        day for day in data.calendar.days if validation_start <= day < test_start
    ]
    if not validation_dates:
        raise ValueError("validation split contains no verified exchange dates")
    validation_end = validation_dates[-1]
    expected_validation_dates = [str(day) for day in validation_dates]
    started = datetime.now(TZ)
    clock = time.monotonic()
    validation = {}
    for instrument in ("TAIEX", "TX"):
        for representation in config["representations"]:
            for model in config["baselines"]:
                key = f"{instrument}/{representation}/{model}"
                rows = walk_forward(
                    data, instrument, model, representation,
                    validation_start, validation_end, config,
                )
                validation[key] = {"rows": rows, "metrics": metrics(rows, config["neutral_bps"])}
                if time.monotonic() - clock > config["budget_cpu_seconds"]:
                    raise TimeoutError("frozen baseline research budget exhausted")
    selected = {}
    selection_audit = {}
    min_coverage = float(config.get("min_validation_coverage", 0.95))
    for instrument in ("TAIEX", "TX"):
        selected[instrument], selection_audit[instrument] = _select_baseline(
            validation,
            instrument,
            expected_validation_dates,
            min_coverage,
            config["neutral_bps"],
        )
    final_test = {}
    for instrument, choice in selected.items():
        rows = walk_forward(
            data, instrument, choice["model"], choice["representation"],
            test_start, test_end, config,
        )
        final_test[instrument] = {"rows": rows, "metrics": metrics(rows, config["neutral_bps"])}
    finished = datetime.now(TZ)
    code_version = _git_state(config_path.parent.parent.parent)
    summary = {
        "experiment_type": "baseline_selection_then_single_final_test",
        "dataset_id": dataset_id,
        "data_quality": data_quality,
        "config_hash": digest(config),
        "code_version": code_version,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": time.monotonic() - clock,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "splits": {
            "validation": [str(validation_start), str(validation_end)],
            "final_test": [str(test_start), str(test_end)],
        },
        "selected_baselines": selected,
        "selection_audit": selection_audit,
        "validation": validation,
        "final_test": final_test,
        "acceptance_gate": {
            "status": "NOT_EVALUATED_NO_LOCKED_TIMESFM_CANDIDATE",
            "close_mae_required_improvement": 0.05,
            "ohl_degradation_limit": 0.05,
        },
    }
    experiment_id = digest({
        "dataset_id": dataset_id, "config_hash": summary["config_hash"],
        "code_version": code_version, "started_at": str(started),
    })[:20]
    output = output_root / experiment_id / "baseline_results.json"
    write_once(output, summary)
    return output


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def _git_state(root: Path) -> dict:
    """Hash committed, modified, staged and untracked research code without logging content."""
    try:
        head = _git(root, "rev-parse", "HEAD").decode().strip()
        status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        diff = _git(root, "diff", "--binary", "HEAD", "--")
        untracked = _git(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
            "tests",
            "config",
            "pyproject.toml",
            "uv.lock",
        ).split(b"\0")
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
        return {"head": "unknown", "error": type(exc).__name__}

    state = hashlib.sha256()
    state.update(diff)
    untracked_paths = []
    for raw_path in sorted(path for path in untracked if path):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = root / relative
        if not path.is_file():
            continue
        untracked_paths.append(relative)
        state.update(b"\0UNTRACKED\0")
        state.update(raw_path)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                state.update(chunk)
    return {
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "worktree_sha256": state.hexdigest(),
        "untracked_code_paths": untracked_paths,
    }


def _git_head(root: Path) -> str:
    """Compatibility helper for callers that only need the committed revision."""
    return _git_state(root)["head"]
