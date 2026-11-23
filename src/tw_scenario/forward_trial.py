"""Immutable public-comment receipts for a bounded, isolated research trial.

GitHub comment created_at/updated_at attest payload persistence, not model quality
or independent verification of a publisher's historical availability.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, time, timedelta

from .morning_adapter import HEADERS, morning_export, validate_forecast
from .news_comparison import NewsDecision, evaluate_pairs, validate_decision
from .shadow import seal
from .snapshot import verify_document
from .split_forecast import TAIPEI

MARKER = "TW_FORWARD_RESEARCH_RECEIPT_V1"


def comment_body(payload):
    if not verify_document(payload).valid:
        raise ValueError("invalid receipt payload hash")
    body = MARKER + "\n```json\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n```"
    if len(body) > 62000:
        raise ValueError("receipt exceeds safe GitHub comment size")
    return body


def parse_comment(body):
    prefix = MARKER + "\n```json\n"
    if not body.startswith(prefix) or not body.endswith("\n```"):
        raise ValueError("not an exact research receipt")
    payload = json.loads(body[len(prefix):-4])
    if not verify_document(payload).valid:
        raise ValueError("remote receipt content hash failed")
    return payload


def package_preopen(forecast, local_receipt, decisions, source_commit, protocol_sha256, at):
    validate_forecast(forecast)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or not re.fullmatch(r"[0-9a-f]{64}", protocol_sha256):
        raise ValueError("explicit code/protocol hashes required")
    if at.tzinfo is None or at.astimezone(TAIPEI).date().isoformat() != forecast["trading_date"]:
        raise ValueError("actual packaging clock must be on the forecast date")
    deadline = datetime.combine(at.astimezone(TAIPEI).date(), time(8, 40), TAIPEI)
    if at >= deadline:
        raise ValueError("packaging missed 08:40; never backfill")
    if local_receipt["snapshot_hash"] != forecast["snapshot_hash"]:
        raise ValueError("local receipt refers to another forecast")
    generated = datetime.fromisoformat(forecast["generated_at"])
    persisted = datetime.fromisoformat(local_receipt["local_persisted_at"])
    cutoff = datetime.fromisoformat(forecast["data_cutoff_at"]).astimezone(TAIPEI)
    if (cutoff.date().isoformat() != forecast["trading_date"] or cutoff.time() > time(8, 30)
            or not cutoff <= generated <= persisted <= at):
        raise ValueError("numerical cutoff/generation/persistence chronology invalid")
    seen = set()
    for raw in decisions:
        if not verify_document(raw).valid:
            raise ValueError("news decision hash failed")
        decision = NewsDecision(**{k: v for k, v in raw.items() if k in NewsDecision.model_fields})
        validate_decision(forecast, decision)
        if decision.generated_at > at or decision.market in seen:
            raise ValueError("future/duplicate news decision")
        if len(decision.news) > 5 or any(n.published_at < at - timedelta(hours=72) for n in decision.news):
            raise ValueError("news falls outside the locked 72h / five-source window")
        seen.add(decision.market)
    return seal({"schema_version": "forward-trial-bundle-v1", "kind": "preopen",
                 "target_date": forecast["trading_date"], "packaged_at": at.isoformat(),
                 "source_commit": source_commit, "protocol_sha256": protocol_sha256,
                 "forecast": forecast, "local_receipt": local_receipt, "news_decisions": decisions,
                 "morning_export": morning_export(forecast, HEADERS),
                 "missing_news_markets": sorted(set(forecast["markets"]) - seen),
                 "formal_accuracy_claim": False, "production_switch": False})


def assess_comment(metadata, repository, issue_number, expected_protocol, expected_commit):
    payload = parse_comment(metadata["body"])
    issue_url = f"https://api.github.com/repos/{repository}/issues/{issue_number}"
    if metadata.get("issue_url") != issue_url:
        raise ValueError("receipt belongs to another research ledger")
    if metadata.get("user", {}).get("login") != repository.split("/")[0]:
        raise ValueError("receipt author does not match the expected repository owner")
    if payload.get("protocol_sha256") != expected_protocol or payload.get("source_commit") != expected_commit:
        raise ValueError("code/protocol differs from the locked trial")
    created, updated = (datetime.fromisoformat(metadata[k]) for k in ("created_at", "updated_at"))
    if created.tzinfo is None or updated.tzinfo is None or updated < created:
        raise ValueError("invalid server clock evidence")
    if payload.get("kind") != "preopen":
        return {"timing_verified": False, "reason": "not a preopen prediction", "comment_id": metadata["id"]}
    # Revalidate the full embedded bundle, not just self-declared eligibility flags.
    rebuilt = package_preopen(payload["forecast"], payload["local_receipt"], payload["news_decisions"],
                              expected_commit, expected_protocol, datetime.fromisoformat(payload["packaged_at"]))
    if rebuilt != payload:
        raise ValueError("embedded numerical/report bundle is inconsistent")
    day = datetime.fromisoformat(payload["target_date"]).date()
    deadline = datetime.combine(day, time(8, 40), TAIPEI)
    packaged = datetime.fromisoformat(payload["packaged_at"])
    # GitHub reports seconds; permit rounding only for relative chronology, never deadline.
    if (created.astimezone(TAIPEI).date() != day or packaged > created + timedelta(seconds=1)
            or created >= deadline or updated >= deadline):
        raise ValueError("remote persistence/edit occurred after deadline or on another date")
    return {"timing_verified": True, "comment_id": metadata["id"], "created_at": metadata["created_at"],
            "updated_at": metadata["updated_at"], "payload_hash": payload["snapshot_hash"],
            "forecast_hash": payload["forecast"]["snapshot_hash"],
            "scope": "server persistence of this exact payload; not accuracy or independent source verification"}


def aggregate_trial_comments(comments, repository, issue_number, protocol_hash, source_commit, expected_dates=None):
    preopens, closes, issues = {}, {}, []
    for metadata in comments:
        if not metadata.get("body", "").startswith(MARKER):
            continue
        if (metadata.get("issue_url") != f"https://api.github.com/repos/{repository}/issues/{issue_number}"
                or metadata.get("user", {}).get("login") != repository.split("/")[0]):
            continue
        try:
            payload = parse_comment(metadata["body"])
            if payload.get("kind") == "preopen":
                proof = assess_comment(metadata, repository, issue_number, protocol_hash, source_commit)
                if proof["timing_verified"]:
                    if expected_dates is not None and payload["target_date"] not in expected_dates:
                        raise ValueError("preopen date outside scheduled population")
                    preopens.setdefault(payload["target_date"], []).append((metadata, payload))
            elif payload.get("kind") == "close":
                if (metadata.get("issue_url") != f"https://api.github.com/repos/{repository}/issues/{issue_number}"
                        or metadata.get("user", {}).get("login") != repository.split("/")[0]
                        or payload.get("source_commit") != source_commit
                        or payload.get("protocol_sha256") != protocol_hash
                        or metadata["updated_at"] != metadata["created_at"]):
                    raise ValueError("close receipt identity/version/edit mismatch")
                closes.setdefault(payload["preopen_comment_id"], []).append((metadata, payload))
            elif payload.get("kind") in ("late_failure", "missing_preopen", "source_failure"):
                issues.append({"comment_id": metadata["id"], "date": payload.get("target_date"),
                               "reason": payload.get("reason", payload["kind"])})
        except (ValueError, KeyError, TypeError) as exc:
            issues.append({"comment_id": metadata.get("id"), "reason": str(exc)})
    forecasts, decisions, actual_index = [], [], {}
    for day, records in sorted(preopens.items()):
        if len(records) != 1:
            issues.append({"date": day, "reason": "duplicate preopen receipts; entire date excluded"})
            continue
        metadata, payload = records[0]
        forecasts.append(payload["forecast"]); decisions.extend(payload["news_decisions"])
        # First complete official outcome receipt, by server publication time.
        candidates = sorted(closes.get(metadata["id"], []), key=lambda pair: pair[0]["created_at"])
        valid = [pair for pair in candidates if len(pair[1].get("official_actuals", [])) == 2
                 and not pair[1].get("actual_failures")]
        if not valid:
            issues.append({"date": day, "reason": "no complete close receipt"})
            continue
        close_metadata, close_payload = valid[0]
        for actual in close_payload["official_actuals"]:
            if (actual["trading_date"] != day or datetime.fromisoformat(actual["available_at"]) >
                    datetime.fromisoformat(close_metadata["created_at"]) + timedelta(seconds=1)):
                raise ValueError("actual date/availability conflicts with server close receipt")
            key = (day, actual["market"], actual["contract"])
            if key in actual_index:
                raise ValueError("duplicate actual identity")
            actual_index[key] = actual
    report = evaluate_pairs(forecasts, decisions, list(actual_index.values()))
    report.update({"receipt_issues": issues, "verified_preopen_days": len(forecasts),
                   "received_preopen_dates": len(preopens), "protocol_sha256": protocol_hash,
                   "rejected_preopen_days": len(preopens) - len(forecasts),
                   "source_commit": source_commit, "rule": "first complete close receipt; all failures retained"})
    report["expected_scheduled_days"] = len(expected_dates) if expected_dates is not None else None
    report["expected_scheduled_pairs"] = 2 * len(expected_dates) if expected_dates is not None else None
    report["missing_preopen_dates"] = sorted(set(expected_dates) - {f["trading_date"] for f in forecasts}) \
        if expected_dates is not None else None
    report["overall_scheduled_pair_coverage"] = report["paired_n"] / (2 * len(expected_dates)) \
        if expected_dates else None
    return report
