"""Package frozen research records; no GitHub credentials or live Sheet writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

from scripts.monitor_shadow import monitor
from tw_scenario.forward_trial import (
    aggregate_trial_comments,
    assess_comment,
    comment_body,
    package_preopen,
    parse_comment,
)
from tw_scenario.morning_adapter import close_export, verify_review_readback
from tw_scenario.news_comparison import evaluate_pairs, freeze_decision
from tw_scenario.postclose import collect_day
from tw_scenario.shadow import now, seal
from tw_scenario.snapshot import write_immutable_json


def read(path):
    return json.loads(path.read_text())


def check_protocol(path):
    config = read(path)
    checkpoint = Path(config["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("locked numerical checkpoint changed")
    prompt = Path(config["news_prompt"])
    if hashlib.sha256(prompt.read_bytes()).hexdigest() != config["news_prompt_sha256"]:
        raise ValueError("locked news prompt changed")
    return config, hashlib.sha256(path.read_bytes()).hexdigest()


def preopen(run, decisions, protocol, output):
    config, digest = check_protocol(protocol)
    at = now()
    if not date.fromisoformat(config["start_date"]) <= at.date() <= date.fromisoformat(config["end_date"]):
        raise ValueError("outside bounded research trial")
    if (run / "failure.json").exists():
        raise ValueError("failed numerical run cannot become a forward prediction")
    forecast = read(run / "forecast.json")
    if (forecast["checkpoint_hash"] != read(Path(config["checkpoint"]))["snapshot_hash"]
            or any(d["default_baseline"] != config["quantitative_method"] for d in forecast["markets"].values())):
        raise ValueError("numerical snapshot is not the locked checkpoint/comparator")
    frozen_news = []
    for path in decisions:
        payload = read(path)
        if payload["prompt_sha256"] != config["news_prompt_sha256"]:
            raise ValueError("news prompt differs from locked protocol")
        if payload["quantitative_method"] != config["quantitative_method"]:
            raise ValueError("quantitative comparator differs")
        frozen_news.append(freeze_decision(forecast, payload, now(), run / f"news-{payload['market']}.json"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    bundle = package_preopen(forecast, read(run / "local-receipt.json"), frozen_news, commit, digest, now())
    write_immutable_json(bundle, output)
    output.with_suffix(".comment.txt").write_text(comment_body(bundle), encoding="utf-8")
    print(json.dumps({"kind": "preopen", "missing_news": bundle["missing_news_markets"], "payload": str(output)}))


def close(metadata_path, protocol, source_commit, repository, issue_number, output):
    config, digest = check_protocol(protocol)
    metadata = read(metadata_path)
    proof = assess_comment(metadata, repository, issue_number, digest, source_commit)
    if not proof["timing_verified"]:
        raise ValueError("no verified preopen payload; never use a late/failure receipt")
    bundle = parse_comment(metadata["body"])
    if any(d["prompt_sha256"] != config["news_prompt_sha256"] for d in bundle["news_decisions"]):
        raise ValueError("remote news prompt differs from locked protocol")
    if bundle["target_date"] != now().date().isoformat():
        raise ValueError("only today's exact preopen receipt is allowed")
    output.mkdir(parents=True, exist_ok=False)
    forecast = bundle["forecast"]
    if forecast["checkpoint_hash"] != read(Path(config["checkpoint"]))["snapshot_hash"]:
        raise ValueError("remote numerical checkpoint differs from the trial")
    contract = forecast["markets"]["TX_selected_monthly"]["inputs"]["contract"]
    actual_bundle = collect_day(now().date(), contract, output / "official")
    run = output / "numeric-run"
    write_immutable_json(forecast, run / "forecast.json")
    write_immutable_json(bundle["local_receipt"], run / "local-receipt.json")
    numeric = monitor([run], actual_bundle["actuals"])
    news = evaluate_pairs([forecast], bundle["news_decisions"], actual_bundle["actuals"])
    review = None
    if actual_bundle["status"] == "complete":
        morning = bundle["morning_export"]
        rows = [r["values"][:] for r in morning["records"]]
        review = close_export(forecast, morning, rows, actual_bundle["actuals"], now())
        for row, patch in zip(rows, review["records"]):
            for column, value in patch["patch_by_zero_based_column"].items(): row[int(column)] = value
        verify_review_readback(review, rows)
        write_immutable_json(review, output / "review-patch.json")
    report = seal({"schema_version": "forward-trial-close-v1", "kind": "close",
                   "target_date": bundle["target_date"], "recorded_at": now().isoformat(),
                   "protocol_sha256": digest, "source_commit": source_commit,
                   "preopen_comment_id": metadata["id"], "preopen_timing": proof,
                   "official_actuals": actual_bundle["actuals"], "actual_failures": actual_bundle["failures"],
                   "numeric_monitor": numeric, "news_comparison": news,
                   "local_report_roundtrip_verified": review is not None,
                   "real_sheet_write": False, "formal_acceptance_claim": False})
    write_immutable_json(report, output / "close.json")
    (output / "close.comment.txt").write_text(comment_body(report), encoding="utf-8")
    print(json.dumps({"kind": "close", "actual_status": actual_bundle["status"], "paired_news_n": news["paired_n"]}))


def aggregate(comments_path, protocol, source_commit, repository, issue_number, output):
    config, digest = check_protocol(protocol)
    first, last = date.fromisoformat(config["start_date"]), min(now().date(), date.fromisoformat(config["end_date"]))
    expected = [(first + timedelta(days=i)).isoformat() for i in range(max(0, (last - first).days + 1))
                if (first + timedelta(days=i)).weekday() < 5]
    report = aggregate_trial_comments(read(comments_path), repository, issue_number, digest, source_commit, expected)
    payload = seal({"kind": "aggregate", "target_date": now().date().isoformat(), "recorded_at": now().isoformat(),
                    "source_commit": source_commit, "protocol_sha256": digest, "comparison": report,
                    "formal_acceptance_claim": False})
    write_immutable_json(payload, output)
    output.with_suffix(".comment.txt").write_text(comment_body(payload), encoding="utf-8")
    print(json.dumps({"kind": "aggregate", "scheduled_days": len(expected), "paired_news_n": report["paired_n"]}))


def failure(protocol, source_commit, kind, reason, output):
    _, digest = check_protocol(protocol)
    payload = seal({"kind": kind, "target_date": now().date().isoformat(), "recorded_at": now().isoformat(),
                    "source_commit": source_commit, "protocol_sha256": digest, "reason": reason,
                    "formal_acceptance_claim": False, "preopen_prediction_created": False})
    write_immutable_json(payload, output)
    output.with_suffix(".comment.txt").write_text(comment_body(payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preopen")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--decisions", type=Path, nargs="*", default=[])
    p.add_argument("--protocol", type=Path, default=Path("config/forward_news_trial_v1.json"))
    p.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("close")
    c.add_argument("--metadata", type=Path, required=True)
    c.add_argument("--protocol", type=Path, default=Path("config/forward_news_trial_v1.json"))
    c.add_argument("--source-commit", required=True)
    c.add_argument("--repository", default="fu06pi/tw-market-scenario-engine")
    c.add_argument("--issue-number", type=int, required=True)
    c.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("--comments", type=Path, required=True)
    g.add_argument("--protocol", type=Path, default=Path("config/forward_news_trial_v1.json"))
    g.add_argument("--source-commit", required=True)
    g.add_argument("--repository", default="fu06pi/tw-market-scenario-engine")
    g.add_argument("--issue-number", type=int, required=True)
    g.add_argument("--output", type=Path, required=True)
    f = sub.add_parser("failure")
    f.add_argument("--protocol", type=Path, default=Path("config/forward_news_trial_v1.json"))
    f.add_argument("--source-commit", required=True)
    f.add_argument("--kind", choices=["late_failure", "missing_preopen", "source_failure"], required=True)
    f.add_argument("--reason", required=True)
    f.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preopen": preopen(args.run, args.decisions, args.protocol, args.output)
    elif args.command == "close": close(args.metadata, args.protocol, args.source_commit, args.repository, args.issue_number, args.output)
    elif args.command == "aggregate": aggregate(args.comments, args.protocol, args.source_commit, args.repository, args.issue_number, args.output)
    else: failure(args.protocol, args.source_commit, args.kind, args.reason, args.output)
