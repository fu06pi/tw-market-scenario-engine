from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from .contracts import TZ, digest, write_once

DEFAULT_CONFIG = Path("config/quant/research.json")
DEFAULT_STAGING_CONFIG = Path("config/quant/staging.json")
DEFAULT_REPORT_ROOT = Path("reports/generated/quant")
SENSITIVE_CONFIG_KEYS = {"api_key", "credential", "password", "secret", "token"}


class CollectionSource(str, Enum):
    FINMIND = "finmind"
    OFFICIAL = "official"


app = typer.Typer(
    no_args_is_help=True,
    help="Reproducible TAIEX/TX research workflows; production switching is disabled.",
)
daily_app = typer.Typer(
    no_args_is_help=True,
    help="Validate, freeze, narrate, and review write-once OHLC daily records.",
)
app.add_typer(daily_app, name="daily")


def _abort(message: str, code: int = 2) -> None:
    typer.echo(f"ERROR: {message}", err=True)
    raise typer.Exit(code=code)


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "schema validation failed; inspect the local input file"
    message = str(exc)
    for name in ("FINMIND_TOKEN", "TEJ_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"required file is missing: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"file is not valid readable JSON: {path}") from None


def _has_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in SENSITIVE_CONFIG_KEYS or normalized.endswith("_api_key"):
                return True
            if normalized.endswith("_token") or _has_sensitive_key(nested):
                return True
    elif isinstance(value, list):
        return any(_has_sensitive_key(item) for item in value)
    return False


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} must use YYYY-MM-DD") from None


def _parse_datetime(value: str | None, field: str) -> datetime:
    if value is None:
        return datetime.now(TZ)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from None
    if parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed


def _load_config(path: Path) -> dict:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise TypeError("research config must be a JSON object")
    if _has_sensitive_key(raw):
        raise ValueError("research config must not contain credential fields")
    required = {
        "version",
        "train_start",
        "validation_start",
        "test_start",
        "test_end",
        "capture_not_before",
        "cutoff",
        "deadline",
        "postclose_not_before",
        "timezone",
        "budget_cpu_seconds",
        "timesfm_code_commit",
        "timesfm_weight_revision",
        "production_switch",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"research config is missing required fields: {', '.join(missing)}")
    if raw["production_switch"] is not False:
        raise ValueError("production_switch must remain false")
    if (raw["cutoff"], raw["deadline"], raw["timezone"]) != (
        "08:30",
        "08:40",
        "Asia/Taipei",
    ):
        raise ValueError("cutoff, deadline, and timezone must remain 08:30/08:40/Asia-Taipei")
    if raw["capture_not_before"] != "08:20" or raw["postclose_not_before"] != "18:00":
        raise ValueError("daily live windows must remain 08:20 pre-open and 18:00 post-close")
    train = _parse_date(str(raw["train_start"]), "train_start")
    validation = _parse_date(str(raw["validation_start"]), "validation_start")
    test = _parse_date(str(raw["test_start"]), "test_start")
    test_end = _parse_date(str(raw["test_end"]), "test_end")
    if not train < validation < test <= test_end:
        raise ValueError("research splits must satisfy train < validation < test <= test_end")
    budget = raw["budget_cpu_seconds"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError("budget_cpu_seconds must be positive")
    return raw


def _config_summary(path: Path, config: dict) -> dict:
    return {
        "status": "VALID_RESEARCH_CONFIG",
        "path": str(path),
        "config_hash": digest(config),
        "version": config["version"],
        "splits": {
            "train_start": config["train_start"],
            "validation_start": config["validation_start"],
            "test_start": config["test_start"],
            "test_end": config["test_end"],
        },
        "cutoff": config["cutoff"],
        "capture_not_before": config["capture_not_before"],
        "deadline": config["deadline"],
        "postclose_not_before": config["postclose_not_before"],
        "timezone": config["timezone"],
        "budget_cpu_seconds": config["budget_cpu_seconds"],
        "timesfm_code_commit": config["timesfm_code_commit"],
        "timesfm_weight_revision": config["timesfm_weight_revision"],
        "production_switch": False,
    }


def _echo_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str))


def _credential_locations(env_file: Path) -> dict:
    result = {
        "FINMIND_TOKEN": "environment" if os.environ.get("FINMIND_TOKEN") else "absent",
        "TEJ_API_KEY": "environment" if os.environ.get("TEJ_API_KEY") else "absent",
        "values_exposed": False,
    }
    if not env_file.exists():
        return result
    if env_file.is_symlink() or not env_file.is_file():
        result["private_env_file"] = "INVALID_NOT_REGULAR"
        return result
    if stat.S_IMODE(env_file.stat().st_mode) & 0o077:
        result["private_env_file"] = "INVALID_PERMISSIONS"
        return result
    try:
        names = {
            line.split("=", 1)[0].strip()
            for line in env_file.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.lstrip().startswith("#")
        }
    except (OSError, UnicodeDecodeError):
        result["private_env_file"] = "INVALID_UNREADABLE"
        return result
    result["private_env_file"] = "VALID_PRIVATE_FILE"
    for name in ("FINMIND_TOKEN", "TEJ_API_KEY"):
        if result[name] == "absent" and name in names:
            result[name] = "private_env_file"
    return result


def _write_json_once_or_same(path: Path, payload: dict) -> tuple[Path, bool]:
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise FileExistsError(f"refusing to replace different immutable output: {path}")
        return path, True
    write_once(path, payload)
    return path, False


@app.command("config")
def show_config(
    config_path: Annotated[
        Path, typer.Option("--config", help="Frozen research JSON.")
    ] = DEFAULT_CONFIG,
) -> None:
    """Validate and show the non-secret frozen research contract."""
    try:
        config = _load_config(config_path)
    except (TypeError, ValueError) as exc:
        _abort(_safe_error(exc))
    _echo_json(_config_summary(config_path, config))


@app.command("inspect")
def inspect_environment(
    config_path: Annotated[
        Path, typer.Option("--config", help="Frozen research JSON.")
    ] = DEFAULT_CONFIG,
    dataset_dir: Annotated[
        Path | None,
        typer.Option("--dataset-dir", help="Optional immutable processed dataset to verify."),
    ] = None,
    env_file: Annotated[
        Path, typer.Option("--env-file", help="Private env file location.")
    ] = Path(".env"),
    timesfm_cache: Annotated[
        Path, typer.Option("--timesfm-cache", help="TimesFM cache location.")
    ] = Path("data/raw/timesfm3"),
) -> None:
    """Inspect config, credential presence, dataset integrity, and TimesFM resources."""
    try:
        config = _load_config(config_path)
        from .timesfm3 import runtime_probe

        probe = runtime_probe(timesfm_cache, int(config["memory_probe_limit_mb"]))
        dataset = {"status": "NOT_REQUESTED"}
        if dataset_dir is not None:
            from .backtest import load_dataset

            loaded, dataset_id = load_dataset(dataset_dir)
            dataset = {
                "status": "INTEGRITY_VERIFIED_QUALITY_NOT_RUN",
                "path": str(dataset_dir),
                "dataset_id": dataset_id,
                "rows": len(loaded.bars),
                "calendar_days": len(loaded.calendar.days),
            }
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "INSPECTED",
            "config": _config_summary(config_path, config),
            "credentials": _credential_locations(env_file),
            "dataset": dataset,
            "timesfm_preflight": asdict(probe),
            "production_actions_performed": False,
        }
    )


@app.command("collect")
def collect_data(
    start: Annotated[str, typer.Option("--start", help="First requested date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option("--end", help="Last complete date (YYYY-MM-DD).")],
    source: Annotated[
        CollectionSource, typer.Option("--source", case_sensitive=False)
    ] = CollectionSource.FINMIND,
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    include_tx: Annotated[bool, typer.Option("--include-tx/--taiex-only")] = True,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Run the bounded immutable FinMind or official collection workflow."""
    try:
        _load_config(config_path)
        start_date = _parse_date(start, "start")
        end_date = _parse_date(end, "end")
        if start_date > end_date:
            raise ValueError("start must not be after end")
        if end_date >= datetime.now(TZ).date():
            raise ValueError("collection accepts complete prior dates only")
        from . import collection

        if source is CollectionSource.FINMIND:
            collector = getattr(collection, "collect_finmind", None)
            if collector is None:
                raise RuntimeError("FinMind collection capability is unavailable")
        else:
            collector = collection.collect_official
        output = Path(collector(start_date, end_date, data_root, include_tx=include_tx))
        required = [output / name for name in ("bars.jsonl", "calendar.json", "manifest.json")]
        if not output.is_dir() or any(not path.is_file() for path in required):
            raise RuntimeError("collector did not produce a complete immutable dataset")
        from .backtest import load_dataset

        loaded, dataset_id = load_dataset(output)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "COLLECTED_INTEGRITY_VERIFIED",
            "source": source.value,
            "dataset_dir": str(output),
            "dataset_id": dataset_id,
            "rows": len(loaded.bars),
            "quality_gate": "NOT_RUN",
            "production_switch": False,
        }
    )


@app.command("quality")
def run_quality(
    dataset_dir: Annotated[
        Path, typer.Argument(help="Immutable processed dataset directory.")
    ],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    start: Annotated[
        str | None, typer.Option("--start", help="Override requested DQ start.")
    ] = None,
    end: Annotated[
        str | None, typer.Option("--end", help="Override requested DQ end.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Immutable JSON report path.")
    ] = None,
) -> None:
    """Run the full dataset-quality gate and preserve failures as evidence."""
    try:
        config = _load_config(config_path)
        from .backtest import load_dataset
        from .quality import quality_report

        loaded, dataset_id = load_dataset(dataset_dir)
        manifest = _read_json(dataset_dir / "manifest.json")
        start_date = _parse_date(start or config["train_start"], "start")
        end_date = _parse_date(end or config["test_end"], "end")
        report = quality_report(
            loaded.bars,
            loaded.calendar,
            start_date,
            end_date,
            manifest=manifest,
        )
        envelope = {
            **report,
            "dataset_id": dataset_id,
            "config_hash": digest(config),
            "requested_start": str(start_date),
            "requested_end": str(end_date),
        }
        report_hash = digest(envelope)
        output = output or (
            DEFAULT_REPORT_ROOT / "data-quality" / f"{dataset_id}-{report_hash[:12]}.json"
        )
        output, reused = _write_json_once_or_same(output, envelope)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": report["status"],
            "dataset_id": dataset_id,
            "report_hash": report_hash,
            "output": str(output),
            "reused": reused,
        }
    )
    if report["status"] != "PASS_RETROSPECTIVE":
        raise typer.Exit(code=2)


@app.command("reconcile")
def reconcile_sources(
    dataset_dir: Annotated[
        Path, typer.Argument(help="Uncertified immutable FinMind dataset directory.")
    ],
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    output: Annotated[
        Path | None, typer.Option("--output", help="Immutable reconciliation report.")
    ] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Reconcile selected TAIEX/TX rows with official data and certify session exceptions."""
    try:
        config = _load_config(config_path)
        from .collection import certify_finmind_with_official

        certified, report = certify_finmind_with_official(dataset_dir, data_root)
        report_hash = digest(report)
        output = output or (
            DEFAULT_REPORT_ROOT
            / "reconciliation"
            / f"{certified.name}-{report_hash[:12]}.json"
        )
        output, reused = _write_json_once_or_same(output, report)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": report["status"],
        "parent_dataset_id": report["parent_dataset_id"],
        "certified_dataset_dir": str(certified),
        "certified_dataset_id": certified.name,
        "verified_session_exceptions": len(report["verified_session_exceptions"]),
        "config_hash": digest(config),
        "report_hash": report_hash,
        "output": str(output),
        "reused": reused,
    })


def _existing_backtest(output_root: Path, dataset_id: str, config_hash: str) -> Path | None:
    if not output_root.exists():
        return None
    for path in sorted(output_root.glob("*/baseline_results.json")):
        try:
            result = _read_json(path)
        except ValueError:
            continue
        if (
            isinstance(result, dict)
            and result.get("dataset_id") == dataset_id
            and result.get("config_hash") == config_hash
            and result.get("final_test")
        ):
            return path
    return None


@app.command("backtest")
def run_backtest(
    dataset_dir: Annotated[
        Path, typer.Argument(help="DQ-passing immutable processed dataset.")
    ],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Immutable experiment root.")
    ] = DEFAULT_REPORT_ROOT / "experiments",
) -> None:
    """Select baselines on validation, then open the fixed final test once."""
    try:
        config = _load_config(config_path)
        from .backtest import load_dataset, run_baseline_suite

        _, dataset_id = load_dataset(dataset_dir)
        config_hash = digest(config)
        existing = _existing_backtest(output_root, dataset_id, config_hash)
        if existing is not None:
            _echo_json(
                {
                    "status": "REUSED_LOCKED_FINAL_TEST",
                    "dataset_id": dataset_id,
                    "config_hash": config_hash,
                    "output": str(existing),
                }
            )
            return
        output = run_baseline_suite(dataset_dir, config_path, output_root)
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "BASELINES_SELECTED_FINAL_TEST_LOCKED",
            "dataset_id": dataset_id,
            "config_hash": config_hash,
            "output": str(output),
        }
    )


@app.command("timesfm-preflight")
def timesfm_preflight(
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("data/raw/timesfm3"),
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional immutable JSON evidence.")
    ] = None,
) -> None:
    """Probe CPU resources and dependencies without loading or downloading weights."""
    try:
        config = _load_config(config_path)
        from .timesfm3 import (
            CODE_COMMIT,
            WEIGHT_REPO,
            WEIGHT_REVISION,
            gradient_smoke_contract,
            runtime_probe,
        )

        if config["timesfm_code_commit"] != CODE_COMMIT:
            raise ValueError("config TimesFM code commit does not match the adapter")
        if config["timesfm_weight_revision"] != WEIGHT_REVISION:
            raise ValueError("config TimesFM weight revision does not match the adapter")
        probe = runtime_probe(cache_dir, int(config["memory_probe_limit_mb"]))
        evidence = {
            **asdict(probe),
            "code_commit": CODE_COMMIT,
            "weight_repo": WEIGHT_REPO,
            "weight_revision": WEIGHT_REVISION,
            "weights_loaded": False,
            "weights_downloaded": False,
            "gradient_smoke": gradient_smoke_contract(),
            "production_switch": False,
        }
        if output is not None:
            _, reused = _write_json_once_or_same(output, evidence)
            evidence["output"] = str(output)
            evidence["reused"] = reused
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(evidence)
    if probe.status != "READY_TO_LOAD":
        raise typer.Exit(code=2)


@app.command("timesfm-gradient-smoke")
def timesfm_gradient_smoke(
    dataset_dir: Annotated[Path, typer.Argument(help="Certified immutable dataset.")],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Ignored local experiment root.")
    ] = DEFAULT_REPORT_ROOT / "experiments",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Pinned local TimesFM weight cache.")
    ] = Path("data/raw/timesfm3"),
) -> None:
    """Run one custom output-head-only gradient step on a train-period sample."""
    try:
        config = _load_config(config_path)
        from .timesfm3 import CODE_COMMIT, WEIGHT_REVISION
        from .timesfm_gradient_smoke import run_output_head_gradient_smoke

        if config["timesfm_code_commit"] != CODE_COMMIT:
            raise ValueError("config TimesFM code commit does not match the adapter")
        if config["timesfm_weight_revision"] != WEIGHT_REVISION:
            raise ValueError("config TimesFM weight revision does not match the adapter")
        output = run_output_head_gradient_smoke(
            dataset_dir, config_path, cache_dir, output_root
        )
        result = _read_json(output)
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": result["status"],
            "name": result["name"],
            "official_timesfm3_finetuning": False,
            "production_candidate": False,
            "gradient_norm_l2": result["gradient_norm_l2"],
            "parameter_delta_l2": result["parameter_delta_l2"],
            "forecast_delta_l2": result["forecast_delta_l2"],
            "output": str(output),
        }
    )


@app.command("timesfm-head-train")
def timesfm_head_train(
    dataset_dir: Annotated[Path, typer.Argument(help="Certified immutable dataset.")],
    instrument: Annotated[
        str, typer.Option("--instrument", help="TAIEX, TX, or JOINT output head.")
    ],
    candidate_index: Annotated[
        int, typer.Option("--candidate-index", help="Zero-based frozen training-grid index.")
    ] = 0,
    config_path: Annotated[
        Path, typer.Option("--config", help="Output-head development config.")
    ] = Path("config/quant/timesfm_output_head.json"),
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Ignored local experiment root.")
    ] = DEFAULT_REPORT_ROOT / "experiments",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Pinned local TimesFM weight cache.")
    ] = Path("data/raw/timesfm3"),
) -> None:
    """Train a custom TimesFM 3 output head without opening a final test."""
    try:
        config = _load_config(config_path)
        from .timesfm3 import CODE_COMMIT, WEIGHT_REVISION
        from .timesfm_head_training import run_output_head_training

        if config["timesfm_code_commit"] != CODE_COMMIT:
            raise ValueError("config TimesFM code commit does not match the adapter")
        if config["timesfm_weight_revision"] != WEIGHT_REVISION:
            raise ValueError("config TimesFM weight revision does not match the adapter")
        output = run_output_head_training(
            dataset_dir,
            config_path,
            cache_dir,
            output_root,
            instrument.upper(),
            candidate_index,
        )
        result = _read_json(output)
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": result["status"],
        "instrument": result["identity"]["instrument"],
        "candidate_index": result["identity"]["candidate_index"],
        "official_timesfm3_finetuning": False,
        "final_test_opened": False,
        "production_candidate": False,
        "checkpoint_sha256": result["checkpoint"]["sha256"],
        "output": str(output),
    })


@app.command("timesfm-head-walk-forward")
def timesfm_head_walk_forward(
    dataset_dir: Annotated[Path, typer.Argument(help="Certified immutable dataset.")],
    instrument: Annotated[
        str, typer.Option("--instrument", help="TAIEX, TX, or JOINT output head.")
    ],
    candidate_index: Annotated[
        int, typer.Option("--candidate-index", help="Zero-based frozen training-grid index.")
    ] = 0,
    config_path: Annotated[
        Path, typer.Option("--config", help="Output-head development config.")
    ] = Path("config/quant/timesfm_output_head.json"),
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Ignored local experiment root.")
    ] = DEFAULT_REPORT_ROOT / "experiments",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Pinned local TimesFM weight cache.")
    ] = Path("data/raw/timesfm3"),
) -> None:
    """Run expanding development folds for the custom TimesFM output head."""
    try:
        config = _load_config(config_path)
        from .timesfm3 import CODE_COMMIT, WEIGHT_REVISION
        from .timesfm_head_training import run_output_head_walk_forward

        if config["timesfm_code_commit"] != CODE_COMMIT:
            raise ValueError("config TimesFM code commit does not match the adapter")
        if config["timesfm_weight_revision"] != WEIGHT_REVISION:
            raise ValueError("config TimesFM weight revision does not match the adapter")
        output = run_output_head_walk_forward(
            dataset_dir,
            config_path,
            cache_dir,
            output_root,
            instrument.upper(),
            candidate_index,
        )
        result = _read_json(output)
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": result["status"],
        "instrument": result["identity"]["instrument"],
        "candidate_index": result["identity"]["candidate_index"],
        "official_timesfm3_finetuning": False,
        "final_test_opened": False,
        "production_candidate": False,
        "development_gate": result["development_gate"]["status"],
        "output": str(output),
    })


@app.command("timesfm-head-daily")
def timesfm_head_daily(
    dataset_dir: Annotated[Path, typer.Argument(help="Immutable pre-open dataset.")],
    bundle: Annotated[Path, typer.Option("--bundle", help="Result/checkpoint paths for both markets.")],
    trading_date: Annotated[str, typer.Option("--date", help="Exchange trading date YYYY-MM-DD.")],
    output: Annotated[Path, typer.Option("--output", help="Write-once research snapshot JSON.")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("data/raw/timesfm3"),
) -> None:
    """Reload trained heads and freeze research OHLC; never switches production."""
    try:
        from .head_daily import run_head_daily

        path = run_head_daily(dataset_dir, bundle, date.fromisoformat(trading_date), output, cache_dir)
        frozen = _read_json(path)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json({"status": "RESEARCH_SNAPSHOT_FROZEN", "output": str(path),
                "forecast_id": frozen["forecast_id"], "snapshot_hash": frozen["snapshot_hash"],
                "timeliness": frozen["timeliness"], "production_switch": False,
                "interval_calibration": "UNVERIFIED"})


@app.command("timesfm-head-select")
def timesfm_head_select(
    results: Annotated[
        list[Path], typer.Argument(help="Completed indexed walk-forward results.")
    ],
    output: Annotated[Path, typer.Option("--output", help="Immutable selection JSON.")],
    max_no_gain_rounds: Annotated[
        int, typer.Option("--max-no-gain-rounds", help="Bounded search stop rule.")
    ] = 3,
) -> None:
    """Select a development candidate without opening a historical final test."""
    try:
        from .timesfm_head_training import select_output_head_candidate

        candidates = [_read_json(path) for path in results]
        result = select_output_head_candidate(candidates, max_no_gain_rounds)
        output, reused = _write_json_once_or_same(output, result)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": result["status"],
        "instrument": result["instrument"],
        "selected": result["selected"],
        "final_test_opened": False,
        "production_candidate": False,
        "output": str(output),
        "reused": reused,
    })


@app.command("timesfm-validate")
def timesfm_validate(
    dataset_dir: Annotated[Path, typer.Argument(help="Certified immutable dataset.")],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Immutable TimesFM experiment root.")
    ] = DEFAULT_REPORT_ROOT / "experiments",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Pinned local TimesFM weight cache.")
    ] = Path("data/raw/timesfm3"),
) -> None:
    """Run validation-only TimesFM candidates and freeze per-market selections."""
    try:
        _load_config(config_path)
        from .timesfm_backtest import run_timesfm_validation

        output = run_timesfm_validation(dataset_dir, config_path, output_root, cache_dir)
        result = _read_json(output)
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": "TIMESFM_CANDIDATES_LOCKED_VALIDATION_ONLY",
        "dataset_id": result["dataset_id"],
        "selected_candidates": result["selected_candidates"],
        "elapsed_seconds": result["elapsed_seconds"],
        "final_test_opened": result["final_test_opened"],
        "output": str(output),
    })


@app.command("timesfm-final")
def timesfm_final(
    validation_path: Annotated[
        Path, typer.Argument(help="Write-once TimesFM validation selection.")
    ],
    baseline_path: Annotated[
        Path, typer.Option("--baseline", help="Locked baseline result with final rows.")
    ],
    dataset_dir: Annotated[
        Path, typer.Option("--dataset-dir", help="Matching certified immutable dataset.")
    ],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Pinned local TimesFM weight cache.")
    ] = Path("data/raw/timesfm3"),
) -> None:
    """Evaluate the validation-locked TimesFM candidates on the final test once."""
    try:
        _load_config(config_path)
        from .timesfm_backtest import run_timesfm_final

        output = run_timesfm_final(
            validation_path, baseline_path, dataset_dir, config_path, cache_dir
        )
        result = _read_json(output)
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": "TIMESFM_FINAL_TEST_RECORDED",
        "dataset_id": result["dataset_id"],
        "acceptance_gate": result["acceptance_gate"]["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(output),
    })


@app.command("final-stability")
def final_stability(
    baseline_json: Annotated[Path, typer.Argument(help="Locked baseline final artifact.")],
    candidate_json: Annotated[Path, typer.Argument(help="Locked candidate final artifact.")],
    output: Annotated[Path, typer.Option("--output")],
    events_json: Annotated[
        Path | None,
        typer.Option("--events", help="Optional frozen date-to-event-label JSON."),
    ] = None,
) -> None:
    """Describe closed final stability without selecting or retuning a model."""
    try:
        from .evaluation import build_final_stability_audit

        baseline = _read_json(baseline_json)
        candidate = _read_json(candidate_json)
        events = None if events_json is None else _read_json(events_json)
        if events is not None and (
            not isinstance(events, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in events.items())
        ):
            raise TypeError("events JSON must map ISO dates to non-empty labels")
        result = build_final_stability_audit(
            baseline,
            candidate,
            event_labels=events,
        )
        output, reused = _write_json_once_or_same(output, result)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": result["status"],
            "candidate_acceptance_status": result["candidate_acceptance_status"],
            "output": str(output),
            "reused": reused,
        }
    )


@app.command("timesfm-joint-validate")
def timesfm_joint_validate(
    independent_validation_path: Annotated[
        Path, typer.Argument(help="Locked independent-market TimesFM validation.")
    ],
    dataset_dir: Annotated[
        Path, typer.Option("--dataset-dir", help="Matching certified immutable dataset.")
    ],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Immutable TimesFM experiment root.")
    ] = DEFAULT_REPORT_ROOT / "experiments",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Pinned local TimesFM weight cache.")
    ] = Path("data/raw/timesfm3"),
) -> None:
    """Compare a TAIEX+TX joint candidate on validation without opening final test."""
    try:
        _load_config(config_path)
        from .timesfm_backtest import run_timesfm_joint_validation

        output = run_timesfm_joint_validation(
            independent_validation_path,
            dataset_dir,
            config_path,
            output_root,
            cache_dir,
        )
        result = _read_json(output)
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": "TIMESFM_JOINT_CANDIDATE_LOCKED_VALIDATION_ONLY",
        "dataset_id": result["dataset_id"],
        "selected_candidates": result["selected_candidates"],
        "elapsed_seconds": result["elapsed_seconds"],
        "final_test_opened": result["final_test_opened"],
        "output": str(output),
    })


@daily_app.command("predict")
def daily_predict(
    body_json: Annotated[
        Path, typer.Argument(help="Model-produced OHLCForecastSnapshotBodyV1 JSON.")
    ],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Validate a model-produced body and emit a sealed candidate without persisting it."""
    try:
        _load_config(config_path)
        from .daily import OHLCForecastSnapshotBodyV1, seal_snapshot

        body = OHLCForecastSnapshotBodyV1.model_validate(_read_json(body_json))
        if body.data_quality.status == "FAIL":
            raise ValueError("daily prediction blocked because data_quality is FAIL")
        snapshot = seal_snapshot(body)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(snapshot.model_dump(mode="json"))


@daily_app.command("build-body")
def daily_build_body(
    trading_date: Annotated[str, typer.Option("--date")],
    dataset_dir: Annotated[Path, typer.Option("--dataset-dir")],
    baseline_path: Annotated[Path, typer.Option("--baseline")],
    final_summary_path: Annotated[Path, typer.Option("--final-summary")],
    generated_at: Annotated[str, typer.Option("--generated-at")],
    persisted_at: Annotated[str, typer.Option("--persisted-at")],
    output: Annotated[Path, typer.Option("--output")],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Build a dual-market body from validation-locked baselines after TimesFM gate failure."""
    try:
        config = _load_config(config_path)
        baseline = _read_json(baseline_path)
        final_summary = _read_json(final_summary_path)
        if final_summary.get("status") != "FAIL":
            raise ValueError("baseline fallback requires a recorded final acceptance failure")
        from .backtest import _git_state, _read_manifest, load_dataset
        from .daily_runner import build_locked_baseline_body

        data, dataset_id = load_dataset(dataset_dir)
        if baseline.get("dataset_id") != dataset_id:
            raise ValueError("baseline lock does not match dataset")
        body = build_locked_baseline_body(
            data=data,
            manifest=_read_manifest(dataset_dir),
            baseline_lock=baseline,
            config=config,
            trading_date=_parse_date(trading_date, "date"),
            generated_at=_parse_datetime(generated_at, "generated_at"),
            persisted_at=_parse_datetime(persisted_at, "persisted_at"),
            code_version=digest(_git_state(config_path.parent.parent.parent)),
            final_summary_hash=digest(final_summary),
        )
        output, reused = _write_json_once_or_same(output, body.model_dump(mode="json"))
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": "DAILY_BODY_READY_RETROSPECTIVE_SHADOW",
        "forecast_id": body.forecast_id,
        "timeliness": body.timeliness,
        "output": str(output),
        "reused": reused,
        "production_switch": False,
    })


@daily_app.command("capture-preopen")
def daily_capture_preopen(
    trading_date: Annotated[str, typer.Option("--date")],
    base_dataset_dir: Annotated[Path, typer.Option("--base-dataset-dir")],
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Capture the FinMind increment before 08:30 and produce an immutable input audit."""
    try:
        config = _load_config(config_path)
        day = _parse_date(trading_date, "date")
        cutoff = datetime.combine(
            day, datetime.strptime(config["cutoff"], "%H:%M").time(), TZ
        )
        from .daily_input import capture_preopen_dataset

        output, report = capture_preopen_dataset(
            base_dataset_dir=base_dataset_dir,
            trading_date=day,
            cutoff_at=cutoff,
            root=data_root,
        )
        report_path = (
            DEFAULT_REPORT_ROOT
            / "daily"
            / "input-audits"
            / str(day)
            / f"{output.name}.json"
        )
        report_path, reused = _write_json_once_or_same(report_path, report)
        if report["status"] != "PASS":
            raise ValueError(f"pre-open capture failed input audit; evidence={report_path}")
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({
        "status": "PREOPEN_INPUT_CAPTURED",
        "trading_date": str(day),
        "daily_dataset_dir": str(output),
        "input_audit": str(report_path),
        "audit_reused": reused,
        "production_switch": False,
    })


@daily_app.command("run-preopen")
def daily_run_preopen(
    trading_date: Annotated[str, typer.Option("--date")],
    daily_dataset_dir: Annotated[Path, typer.Option("--daily-dataset-dir")],
    baseline_path: Annotated[Path, typer.Option("--baseline")],
    final_summary_path: Annotated[Path, typer.Option("--final-summary")],
    observed_headers_json: Annotated[Path, typer.Option("--observed-headers")],
    start_row: Annotated[int, typer.Option("--start-row", min=2)],
    news_json: Annotated[Path | None, typer.Option("--news")] = None,
    started_at: Annotated[str | None, typer.Option("--started-at")] = None,
    output_root: Annotated[
        Path, typer.Option("--output-root")
    ] = DEFAULT_REPORT_ROOT / "daily",
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Build, freeze and narrate one cutoff-certified staging forecast; never write Sheets."""
    from .daily_orchestration import DailyRunFailed, run_preopen_staging

    try:
        config = _load_config(config_path)
        headers = _read_json(observed_headers_json)
        if not isinstance(headers, list) or not all(isinstance(item, str) for item in headers):
            raise TypeError("observed headers JSON must be a list of strings")
        news = [] if news_json is None else _read_json(news_json)
        if not isinstance(news, list):
            raise TypeError("news JSON must be a list")
        evidence_path = run_preopen_staging(
            daily_dataset_dir=daily_dataset_dir,
            baseline_path=baseline_path,
            final_summary_path=final_summary_path,
            config=config,
            config_path=config_path,
            trading_date=_parse_date(trading_date, "date"),
            started_at=_parse_datetime(started_at, "started_at"),
            output_root=output_root,
            start_row=start_row,
            observed_headers=tuple(headers),
            news=news,
        )
        result = _read_json(evidence_path)
    except DailyRunFailed as exc:
        _abort(_safe_error(exc))
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({**result, "evidence": str(evidence_path)})


@daily_app.command("live-preopen")
def daily_live_preopen(
    trading_date: Annotated[str, typer.Option("--date")],
    base_dataset_dir: Annotated[Path, typer.Option("--base-dataset-dir")],
    baseline_path: Annotated[Path, typer.Option("--baseline")],
    final_summary_path: Annotated[Path, typer.Option("--final-summary")],
    news_json: Annotated[Path | None, typer.Option("--news")] = None,
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    output_root: Annotated[
        Path, typer.Option("--output-root")
    ] = DEFAULT_REPORT_ROOT / "daily",
    staging_config_path: Annotated[
        Path, typer.Option("--staging-config")
    ] = DEFAULT_STAGING_CONFIG,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Run the single staging-only live path from pre-cutoff capture through Sheet readback."""
    from .daily_orchestration import DailyRunFailed, run_live_preopen

    try:
        config = _load_config(config_path)
        news = [] if news_json is None else _read_json(news_json)
        if not isinstance(news, list):
            raise TypeError("news JSON must be a list")
        writer = _staging_writer(staging_config_path)
        evidence_path = run_live_preopen(
            trading_date=_parse_date(trading_date, "date"),
            base_dataset_dir=base_dataset_dir,
            baseline_path=baseline_path,
            final_summary_path=final_summary_path,
            config=config,
            config_path=config_path,
            data_root=data_root,
            output_root=output_root,
            writer=writer,
            news=news,
        )
        result = _read_json(evidence_path)
    except DailyRunFailed as exc:
        _abort(_safe_error(exc))
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({**result, "evidence": str(evidence_path)})


@daily_app.command("freeze")
def daily_freeze(
    snapshot_json: Annotated[
        Path, typer.Argument(help="Sealed OHLCForecastSnapshotV1 JSON.")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", help="Write-once snapshot path.")
    ] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Persist a sealed daily snapshot with exclusive create and verified readback."""
    try:
        _load_config(config_path)
        from .daily import OHLCForecastSnapshotV1, freeze_snapshot, load_snapshot

        snapshot = OHLCForecastSnapshotV1.model_validate(_read_json(snapshot_json))
        if snapshot.data_quality.status == "FAIL":
            raise ValueError("daily freeze blocked because data_quality is FAIL")
        output = output or (
            DEFAULT_REPORT_ROOT / "daily" / "snapshots" / f"{snapshot.forecast_id}.json"
        )
        reused = False
        if output.exists():
            if load_snapshot(output) != snapshot:
                raise FileExistsError(f"refusing to replace different snapshot: {output}")
            reused = True
        else:
            freeze_snapshot(snapshot, output)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "ALREADY_FROZEN" if reused else "FROZEN",
            "forecast_id": snapshot.forecast_id,
            "snapshot_hash": snapshot.snapshot_hash,
            "timeliness": snapshot.timeliness,
            "output": str(output),
        }
    )


def _daily_record_path(root: Path, kind: str, record: Any) -> Path:
    contract = record.contract if record.contract is not None else "cash"
    return root / kind / record.forecast_id / f"{record.instrument}-{contract}.json"


@daily_app.command("review")
def daily_review(
    snapshot_json: Annotated[
        Path, typer.Argument(help="Previously frozen daily snapshot.")
    ],
    realization_json: Annotated[
        Path, typer.Argument(help="One append-only OHLCRealizationV1 record.")
    ],
    record_root: Annotated[
        Path, typer.Option("--record-root")
    ] = DEFAULT_REPORT_ROOT / "daily",
    evaluated_at: Annotated[str | None, typer.Option("--evaluated-at")] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Append a realization and its evaluation without changing the forecast."""
    try:
        _load_config(config_path)
        from .daily import (
            OHLCEvaluationV1,
            OHLCRealizationV1,
            append_evaluation,
            append_realization,
            evaluate_realization,
            load_snapshot,
        )

        snapshot = load_snapshot(snapshot_json)
        realization = OHLCRealizationV1.model_validate(_read_json(realization_json))
        evaluation = evaluate_realization(
            snapshot, realization, _parse_datetime(evaluated_at, "evaluated_at")
        )
        records = (
            (
                realization,
                _daily_record_path(record_root, "realizations-v1", realization),
                append_realization,
                OHLCRealizationV1,
            ),
            (
                evaluation,
                _daily_record_path(record_root, "evaluations-v1", evaluation),
                append_evaluation,
                OHLCEvaluationV1,
            ),
        )
        paths = []
        reused = []
        for record, path, append, record_type in records:
            if path.exists():
                existing = record_type.model_validate(_read_json(path))
                if existing != record:
                    raise FileExistsError(f"refusing to replace different daily record: {path}")
                reused.append(True)
            else:
                append(record, record_root)
                reused.append(False)
            paths.append(path)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "REVIEW_APPENDED",
            "forecast_id": snapshot.forecast_id,
            "realization": str(paths[0]),
            "evaluation": str(paths[1]),
            "reused": reused,
        }
    )


@daily_app.command("narrative")
def daily_narrative(
    snapshot_json: Annotated[
        Path, typer.Argument(help="Frozen OHLCForecastSnapshotV1 JSON.")
    ],
    news_json: Annotated[
        Path | None, typer.Option("--news", help="Optional cutoff-safe news list.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Immutable narrative JSON.")
    ] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Generate the deterministic V0 narrative from exact frozen numbers."""
    try:
        _load_config(config_path)
        from .daily import build_v0_narrative, load_snapshot

        snapshot = load_snapshot(snapshot_json)
        news = [] if news_json is None else _read_json(news_json)
        if not isinstance(news, list):
            raise TypeError("news JSON must be a list")
        narrative = build_v0_narrative(snapshot, news)
        output = output or (
            DEFAULT_REPORT_ROOT / "daily" / "narratives" / f"{snapshot.forecast_id}.json"
        )
        output, reused = _write_json_once_or_same(output, narrative.model_dump(mode="json"))
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "NARRATIVE_READY",
            "forecast_id": snapshot.forecast_id,
            "output": str(output),
            "reused": reused,
        }
    )


@daily_app.command("sheet-dry-run")
def daily_sheet_dry_run(
    snapshot_json: Annotated[
        Path, typer.Argument(help="Frozen OHLCForecastSnapshotV1 JSON.")
    ],
    narrative_json: Annotated[
        Path, typer.Argument(help="Narrative generated from that snapshot.")
    ],
    observed_headers_json: Annotated[
        Path,
        typer.Option("--observed-headers", help="Captured A:P headers; no Sheet access occurs."),
    ],
    start_row: Annotated[int, typer.Option("--start-row", min=2)],
    output: Annotated[
        Path | None, typer.Option("--output", help="Immutable dry-run batch JSON.")
    ] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Build a local A:P Sheet batch; this command never writes Google Sheets."""
    try:
        _load_config(config_path)
        from .daily import (
            V0NarrativePayload,
            build_v0_sheet_dry_run,
            load_snapshot,
            save_sheet_dry_run,
        )

        snapshot = load_snapshot(snapshot_json)
        narrative = V0NarrativePayload.model_validate(_read_json(narrative_json))
        observed_headers = _read_json(observed_headers_json)
        if not isinstance(observed_headers, list) or not all(
            isinstance(item, str) for item in observed_headers
        ):
            raise TypeError("observed headers JSON must be a list of strings")
        batch = build_v0_sheet_dry_run(snapshot, narrative, observed_headers, start_row)
        output = output or (
            DEFAULT_REPORT_ROOT / "daily" / "sheet-dry-runs" / f"{snapshot.forecast_id}.json"
        )
        reused = False
        if output.exists():
            if _read_json(output) != batch.model_dump(mode="json"):
                raise FileExistsError(f"refusing to replace different Sheet dry-run: {output}")
            reused = True
        else:
            save_sheet_dry_run(batch, output)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "SHEET_DRY_RUN_ONLY",
            "forecast_id": snapshot.forecast_id,
            "write_range": batch.write_range,
            "output": str(output),
            "reused": reused,
            "production_sheet_written": False,
        }
    )


@daily_app.command("sheet-postclose-dry-run")
def daily_sheet_postclose_dry_run(
    snapshot_json: Annotated[
        Path, typer.Argument(help="Frozen OHLCForecastSnapshotV1 JSON.")
    ],
    morning_batch_json: Annotated[
        Path, typer.Argument(help="Morning V0SheetDryRunBatch written to staging.")
    ],
    taiex_realization_json: Annotated[Path, typer.Argument()],
    taiex_evaluation_json: Annotated[Path, typer.Argument()],
    tx_realization_json: Annotated[Path, typer.Argument()],
    tx_evaluation_json: Annotated[Path, typer.Argument()],
    reviewed_at: Annotated[str, typer.Option("--reviewed-at")],
    output: Annotated[
        Path | None, typer.Option("--output", help="Immutable post-close batch JSON.")
    ] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Build a local G:N,P-only update bound to immutable morning cells."""
    try:
        _load_config(config_path)
        from .daily import (
            OHLCEvaluationV1,
            OHLCRealizationV1,
            V0SheetDryRunBatch,
            build_v0_postclose_sheet_batch,
            load_snapshot,
            save_postclose_sheet_dry_run,
        )

        snapshot = load_snapshot(snapshot_json)
        morning_batch = V0SheetDryRunBatch.model_validate(_read_json(morning_batch_json))
        realizations = (
            OHLCRealizationV1.model_validate(_read_json(taiex_realization_json)),
            OHLCRealizationV1.model_validate(_read_json(tx_realization_json)),
        )
        evaluations = (
            OHLCEvaluationV1.model_validate(_read_json(taiex_evaluation_json)),
            OHLCEvaluationV1.model_validate(_read_json(tx_evaluation_json)),
        )
        batch = build_v0_postclose_sheet_batch(
            snapshot,
            morning_batch,
            realizations,
            evaluations,
            _parse_datetime(reviewed_at, "reviewed_at"),
        )
        output = output or (
            DEFAULT_REPORT_ROOT
            / "daily"
            / "sheet-postclose-dry-runs"
            / f"{snapshot.forecast_id}.json"
        )
        reused = False
        if output.exists():
            if _read_json(output) != batch.model_dump(mode="json"):
                raise FileExistsError(
                    f"refusing to replace different post-close Sheet batch: {output}"
                )
            reused = True
        else:
            save_postclose_sheet_dry_run(batch, output)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json(
        {
            "status": "SHEET_POSTCLOSE_DRY_RUN_ONLY",
            "forecast_id": snapshot.forecast_id,
            "update_columns": batch.update_columns,
            "rows": [row.row_number for row in batch.rows],
            "output": str(output),
            "reused": reused,
            "production_sheet_written": False,
        }
    )


def _staging_writer(staging_config_path: Path):
    staging = _read_json(staging_config_path)
    if not isinstance(staging, dict) or staging.get("production_switch") is not False:
        raise ValueError("staging config must keep production_switch=false")
    required = {"staging_sheet_id", "production_sheet_id", "sheet_name"}
    if not all(isinstance(staging.get(key), str) and staging[key] for key in required):
        raise ValueError("staging config is incomplete")
    from .sheets import StagingSheetWriter

    return StagingSheetWriter(
        sheet_id=staging["staging_sheet_id"],
        staging_sheet_id=staging["staging_sheet_id"],
        production_sheet_id=staging["production_sheet_id"],
    )


@daily_app.command("sheet-preflight-staging")
def daily_sheet_preflight_staging(
    staging_config_path: Annotated[
        Path, typer.Option("--staging-config")
    ] = DEFAULT_STAGING_CONFIG,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Read staging headers and next rows without writing any cell."""
    try:
        _load_config(config_path)
        staging = _read_json(staging_config_path)
        writer = _staging_writer(staging_config_path)
        result = writer.preflight(staging["sheet_name"])
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json({**result, "status": "STAGING_READ_ONLY_PREFLIGHT_VERIFIED"})


@daily_app.command("sheet-write-staging")
def daily_sheet_write_staging(
    batch_json: Annotated[Path, typer.Argument(help="Validated morning Sheet batch JSON.")],
    receipt: Annotated[Path | None, typer.Option("--receipt")] = None,
    staging_config_path: Annotated[
        Path, typer.Option("--staging-config")
    ] = DEFAULT_STAGING_CONFIG,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Write A:F,O to the configured staging Sheet and verify exact readback."""
    try:
        _load_config(config_path)
        from .daily import V0SheetDryRunBatch
        from .sheets import save_receipt

        batch = V0SheetDryRunBatch.model_validate(_read_json(batch_json))
        writer = _staging_writer(staging_config_path)
        result = writer.write_morning(batch)
        receipt = receipt or (
            DEFAULT_REPORT_ROOT
            / "daily"
            / "sheet-receipts"
            / f"{batch.forecast_id}-morning.json"
        )
        receipt = save_receipt(result, receipt)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json({**result, "status": "STAGING_WRITE_VERIFIED", "receipt": str(receipt)})


@daily_app.command("sheet-postclose-write-staging")
def daily_sheet_postclose_write_staging(
    batch_json: Annotated[Path, typer.Argument(help="Validated post-close batch JSON.")],
    receipt: Annotated[Path | None, typer.Option("--receipt")] = None,
    staging_config_path: Annotated[
        Path, typer.Option("--staging-config")
    ] = DEFAULT_STAGING_CONFIG,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Write G:N,P to staging after verifying the immutable morning cells."""
    try:
        _load_config(config_path)
        from .daily import V0PostcloseSheetBatch
        from .sheets import save_receipt

        batch = V0PostcloseSheetBatch.model_validate(_read_json(batch_json))
        writer = _staging_writer(staging_config_path)
        result = writer.write_postclose(batch)
        receipt = receipt or (
            DEFAULT_REPORT_ROOT
            / "daily"
            / "sheet-receipts"
            / f"{batch.forecast_id}-postclose.json"
        )
        receipt = save_receipt(result, receipt)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _abort(_safe_error(exc))
    _echo_json({**result, "status": "STAGING_POSTCLOSE_VERIFIED", "receipt": str(receipt)})


@daily_app.command("live-postclose")
def daily_live_postclose(
    snapshot_json: Annotated[Path, typer.Argument(help="Frozen morning snapshot JSON.")],
    daily_dataset_dir: Annotated[Path, typer.Option("--daily-dataset-dir")],
    morning_batch_json: Annotated[Path, typer.Option("--morning-batch")],
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    output_root: Annotated[
        Path, typer.Option("--output-root")
    ] = DEFAULT_REPORT_ROOT / "daily",
    staging_config_path: Annotated[
        Path, typer.Option("--staging-config")
    ] = DEFAULT_STAGING_CONFIG,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    """Capture closing OHLC, append errors, and verify the staging G:N,P update."""
    from .daily_orchestration import DailyRunFailed, run_live_postclose

    try:
        config = _load_config(config_path)
        writer = _staging_writer(staging_config_path)
        evidence_path = run_live_postclose(
            snapshot_path=snapshot_json,
            daily_dataset_dir=daily_dataset_dir,
            morning_batch_path=morning_batch_json,
            data_root=data_root,
            output_root=output_root,
            writer=writer,
            config=config,
        )
        result = _read_json(evidence_path)
    except DailyRunFailed as exc:
        _abort(_safe_error(exc))
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        _abort(_safe_error(exc))
    _echo_json({**result, "evidence": str(evidence_path)})


@app.command('timesfm-head-calibrate')
def timesfm_head_calibrate(
    dataset_dir: Path,
    experiment_dir: Path,
    output_dir: Path,
    config: Path = Path('config/quant/timesfm_output_head_short.json'),
    calibration_config: Path = Path('config/quant/head_interval_recalibration.json'),
):
    """Evaluate frozen causal marginal calibration on saved development folds."""
    from .head_calibration import evaluate_saved_head_intervals

    report, rows = evaluate_saved_head_intervals(
        dataset_dir, experiment_dir, json.loads(config.read_text()),
        json.loads(calibration_config.read_text()))
    write_once(output_dir / 'result.json', report)
    write_once(output_dir / 'predictions.json', rows)
    typer.echo(json.dumps({'result': str(output_dir / 'result.json'),
                           'status': report['development_gate']['status'],
                           'coverage': report['development_gate']['interval_coverage']}))


@app.command('timesfm-head-replay')
def timesfm_head_replay(
    dataset_dir: Path,
    bundle: Path,
    trading_date: str,
    output_dir: Path,
    config: Path = Path('config/quant/timesfm_output_head_short.json'),
    cache_dir: Path = Path('data/raw/timesfm3'),
    budget_root: Path = Path('reports/generated/quant/experiments'),
):
    """Verify real checkpoint daily replay within the existing cumulative budget."""
    from .head_budget import HeadBudget
    from .head_daily import verify_head_replay

    settings = _read_json(config)
    with HeadBudget(budget_root / 'output-head-search-budget.json',
                    digest({'operation': 'daily_replay', 'bundle': _read_json(bundle),
                            'date': trading_date}), settings['search_budget_seconds'],
                    settings['budget_cpu_seconds']) as budget:
        report_path = output_dir / 'replay-result.json'
        report = verify_head_replay(dataset_dir, bundle, date.fromisoformat(trading_date),
                                   output_dir / 'snapshot.json', cache_dir, report_path)
        budget.result_path = report_path
    _echo_json(report)
