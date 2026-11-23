"""Read-only audit ledger and conservative, version-separated inclusion policy."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from datetime import date, datetime, time
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from .metrics import evaluate_snapshot, realized_direction
from .models import MarketRealization, PredictionSnapshot
from .policy import ScoringPolicy
from .snapshot import document_hash, verify_document, write_immutable_json
from .split_forecast import TAIPEI, ObservedBar, SplitForecast, evaluate_split


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_persistence(root: Path, path: Path) -> dict | None:
    """Derive file-birth time, never trust a snapshot's self-reported generated_at.

    Git commit metadata is evidence in this repository, not an independent
    attestation of when a remote audience could first access the forecast.
    A working-tree edit or a later content-changing commit disqualifies the file.
    """
    relative = str(path.relative_to(root))
    args = ["git", "-C", str(root)]
    try:
        history = subprocess.run(
            [*args, "log", "--follow", "--format=%H\t%cI", "--", relative],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if not history:
            return None
        oldest_sha, recorded_at = history[-1].split("\t")
        original = subprocess.run(
            [*args, "show", f"{oldest_sha}:{relative}"], check=True, capture_output=True
        ).stdout
        changed = original != path.read_bytes()
        # Even an edit later reverted must not be treated as an immutable record.
        for entry in history[:-1]:
            sha = entry.split("\t")[0]
            historical = subprocess.run(
                [*args, "show", f"{sha}:{relative}"], check=True, capture_output=True
            ).stdout
            changed = changed or historical != original
        return {
            "recorded_at": recorded_at,
            "source": "repository_git_history",
            "commit": oldest_sha,
            "content_changed": changed,
            "limitation": "Git metadata is not independent public-publication attestation.",
        }
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None


def assess_eligibility(
    payload: dict,
    integrity: dict,
    *,
    has_actual: bool,
    manifest: dict,
    evaluation: dict,
    persistence: dict | None,
) -> dict:
    """A True historical flag never overrides contrary evidence or another False."""
    is_v2 = payload.get("schema_version") == "ohlc-split-v2"
    identity = payload["inputs"] if is_v2 else payload
    reasons = []
    warnings = []
    if not integrity["valid"]:
        reasons.append("integrity_failed")
    if not has_actual:
        reasons.append("missing_actual")
    flags = []
    for label, source in (("manifest", manifest), ("evaluation", evaluation)):
        if not isinstance(source, dict):
            raise TypeError(f"{label} metadata must be a JSON object")
        flag = source.get("aggregate_metric_eligibility")
        if isinstance(flag, bool):
            flags.append(flag)
            if not flag:
                reasons.append(f"{label}_explicit_exclusion")
        status = str(source.get("snapshot_status", source.get("status", ""))).lower()
        if (
            source.get("reconstructed")
            or source.get("late_generated")
            or any(
                token in status
                for token in ("reconstruct", "backfill", "late_generated", "postopen")
            )
        ):
            reasons.append(f"{label}_late_or_reconstructed")
        samples = source.get("sample_eligibility", {})
        if not isinstance(samples, dict):
            raise TypeError(f"{label} sample_eligibility must be a JSON object")
        for key in ("formal_hash_verified_aggregate", "formal_legacy_direction_metrics"):
            if samples.get(key) is False:
                reasons.append(f"{label}_{key}_excluded")
        if source.get("trading_date", identity["trading_date"]) != identity["trading_date"]:
            reasons.append(f"{label}_date_mismatch")
    if True in flags and False in flags:
        warnings.append("conflicting_historical_inclusion_flags")
    cutoff = datetime.fromisoformat(identity["data_cutoff_at"])
    generated = datetime.fromisoformat(identity["generated_at"])
    if cutoff.tzinfo is None or generated.tzinfo is None:
        reasons.append("timezone_missing")
    else:
        cutoff, generated = cutoff.astimezone(TAIPEI), generated.astimezone(TAIPEI)
        deadline = time(8, 40) if is_v2 else time(9, 0)
        feature_deadline = time(8, 30) if is_v2 else time(8, 50)
        if (
            cutoff.date().isoformat() != identity["trading_date"]
            or cutoff.time() > feature_deadline
        ):
            reasons.append("cutoff_contract_failed")
        if (
            generated.date().isoformat() != identity["trading_date"]
            or generated < cutoff
            or generated.time() >= deadline
        ):
            reasons.append("generation_contract_failed")
        if persistence is None:
            reasons.append("missing_persistence_evidence")
        else:
            persisted = datetime.fromisoformat(persistence["recorded_at"])
            if persisted.tzinfo is None:
                reasons.append("persistence_timezone_missing")
            else:
                persisted = persisted.astimezone(TAIPEI)
                if (
                    persisted.date().isoformat() != identity["trading_date"]
                    or persisted.time() >= deadline
                    or persisted < cutoff
                ):
                    reasons.append("persistence_contract_failed")
                if persisted < generated:
                    reasons.append("generation_after_recorded_persistence")
            if persistence.get("content_changed"):
                reasons.append("historical_prediction_changed")
    cohort = "split_v2_preopen" if is_v2 else "legacy_preopen"
    return {
        "eligible": not reasons,
        "cohort": cohort,
        "reasons": sorted(set(reasons)),
        "warnings": warnings,
        "new_ohlc_acceptance_proven": False,
    }


def _legacy_scores(payload: dict, actual: dict, policy: ScoringPolicy) -> dict:
    prediction = PredictionSnapshot.model_validate(payload)
    realization = MarketRealization.model_validate(actual)
    scores = evaluate_snapshot(prediction, realization, policy)
    scores["open_direction_predicted"] = prediction.targets.open_direction.value
    scores["close_direction_predicted"] = prediction.targets.close_direction.value
    gap = prediction.targets.expected_open_gap_bps
    close = prediction.targets.expected_close_return_bps
    # This makes legacy gap/body errors inspectable, without pretending an
    # independently trained body head generated the original prediction.
    if gap is not None and close is not None and gap > -10_000 and close > -10_000:
        body = ((1 + close / 10_000) / (1 + gap / 10_000) - 1) * 10_000
        observed = (realization.close / realization.open - 1) * 10_000
        scores.update(
            session_body_predicted_bps=body,
            session_body_realized_bps=observed,
            session_body_mae_bps=abs(body - observed),
            session_body_source="derived_legacy_targets_not_independent_head",
        )
    return scores


def audit_repository(root: str | Path, policy: ScoringPolicy | None = None) -> dict:
    root = Path(root).resolve()
    policy = policy or ScoringPolicy()
    rows = []
    paths = sorted((root / "predictions").glob("*.json"))
    paths += sorted((root / "predictions" / "v2").glob("*.json"))
    for path in paths:
        row = {"prediction_path": str(path.relative_to(root)), "scores": None}
        try:
            payload = read_json(path)
            if not isinstance(payload, dict):
                raise TypeError("prediction must be a JSON object")
            if payload.get("schema_version") not in (None, "ohlc-split-v2"):
                raise ValueError("unsupported prediction schema")
            is_v2 = payload.get("schema_version") == "ohlc-split-v2"
            identity = payload["inputs"] if is_v2 else payload
            trading_date = identity["trading_date"]
            if date.fromisoformat(trading_date).isoformat() != trading_date:
                raise ValueError("noncanonical trading date")
            market = identity.get("market", "TAIEX")
            contract = identity.get("contract")
            # Keep identifiable forecasts in their version's denominator even
            # when a later actual/manifest/evaluation validation fails.
            row.update(
                trading_date=trading_date,
                market=market,
                contract=contract,
                schema_version=payload.get("schema_version", "legacy"),
                model_version=payload["model_version"],
                feature_version=identity.get("feature_version"),
                prompt_version=payload.get("prompt_version"),
                config_hash=document_hash(payload["config"]) if is_v2 else None,
            )
            # V2 actuals must carry an unambiguous market and contract identity.
            actual_name = f"{trading_date}-{market}" + (f"-{contract}" if contract else "")
            actual_path = (
                root / "realizations" / "v2" / f"{actual_name}.json"
                if is_v2
                else (root / "realizations" / f"{trading_date}.json")
            )
            manifest_path = root / "data/manifests" / f"{trading_date}.json"
            evaluation_path = root / "reports/evaluations" / f"{trading_date}.json"
            # Legacy day-wide flags cannot veto or promote a separate V2 forecast.
            manifest = read_json(manifest_path) if manifest_path.exists() and not is_v2 else {}
            evaluation = (
                read_json(evaluation_path) if evaluation_path.exists() and not is_v2 else {}
            )
            integrity = verify_document(payload).to_dict()
            if is_v2:
                SplitForecast.model_validate(payload)
            else:
                PredictionSnapshot.model_validate(payload)
            actual = read_json(actual_path) if actual_path.exists() else None
            if actual is not None:
                if is_v2:
                    scores = evaluate_split(
                        SplitForecast.model_validate(payload),
                        ObservedBar.model_validate(actual),
                        policy,
                    )
                    scores["open_direction_hit"] = scores["open_gap_direction_hit"]
                    scores["close_direction_hit"] = scores["close_return_direction_hit"]
                    for target, component in (("open", "open_gap"), ("close", "close_return")):
                        for kind in ("predicted", "realized"):
                            scores[f"{target}_direction_{kind}"] = realized_direction(
                                scores[f"{component}_{kind}_bps"], policy.flat_threshold_bps
                            ).value
                else:
                    scores = _legacy_scores(payload, actual, policy)
                row["scores"] = scores
            persistence = git_persistence(root, path)
            eligibility = assess_eligibility(
                payload,
                integrity,
                has_actual=actual is not None,
                manifest=manifest,
                evaluation=evaluation,
                persistence=persistence,
            )
            if path.stem != (actual_name if is_v2 else trading_date):
                eligibility["eligible"] = False
                eligibility["reasons"].append("filename_date_mismatch")
            row.update(integrity=integrity, persistence=persistence, eligibility=eligibility)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            row.update(
                eligibility={"eligible": False, "cohort": "invalid", "reasons": ["invalid_record"]},
                error=str(exc),
            )
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        if "model_version" not in row:
            continue
        key = tuple(
            row.get(k)
            for k in (
                "schema_version",
                "market",
                "contract",
                "model_version",
                "feature_version",
                "prompt_version",
                "config_hash",
            )
        )
        groups[key].append(row)
    aggregates = []
    for key, group in groups.items():
        included = [r for r in group if r["eligibility"]["eligible"] and r["scores"] is not None]
        metrics = {}
        confusion = {}
        for target in ("open", "close"):
            matrix = {p: {r: 0 for r in ("up", "flat", "down")} for p in ("up", "flat", "down")}
            for row in included:
                scores = row["scores"]
                matrix[scores[f"{target}_direction_predicted"]][
                    scores[f"{target}_direction_realized"]
                ] += 1
            confusion[target] = {"rows": "predicted", "columns": "realized", "counts": matrix}
        for metric in (
            "open_direction_hit",
            "close_direction_hit",
            "open_gap_mae_bps",
            "session_body_mae_bps",
            "close_return_mae_bps",
        ):
            values = [r["scores"][metric] for r in included if r["scores"].get(metric) is not None]
            metrics[metric] = {"mean": mean(values) if values else None, "n": len(values)}
        aggregates.append(
            dict(
                zip(
                    (
                        "schema_version",
                        "market",
                        "contract",
                        "model_version",
                        "feature_version",
                        "prompt_version",
                        "config_hash",
                    ),
                    key,
                )
            )
            | {
                "observed_forecast_count": len(group),
                "eligible_count": len(included),
                "excluded_count": len(group) - len(included),
                "metrics": metrics,
                "direction_confusion_matrices": confusion,
            }
        )
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return {
        "audit_version": "eligibility-v2",
        "source_git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("pydantic", "numpy", "scikit-learn", "PyYAML", "typer")
            },
        },
        "implementation_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "audit.py",
                "snapshot.py",
                "policy.py",
                "metrics.py",
                "models.py",
                "split_forecast.py",
            )
        },
        "scoring_policy": policy.model_dump(),
        "scope": "repository records; actual market data not externally revalidated",
        "coverage_scope": "observed files only; scheduled trading-day coverage not established",
        "persistence_caveat": "Git metadata is not independent publication-time attestation",
        "prediction_count": len(rows),
        "eligible_count": sum(r["eligibility"]["eligible"] for r in rows),
        "rows": rows,
        "aggregates": aggregates,
    }


def write_audit(report: dict, root: str | Path, output: str | Path) -> Path:
    root, output = Path(root).resolve(), Path(output).resolve()
    for directory in ("predictions", "realizations", "data/manifests", "reports/evaluations"):
        if output.is_relative_to(root / directory):
            raise ValueError("audit output must not be written into historical record directories")
    return write_immutable_json(report, output)
