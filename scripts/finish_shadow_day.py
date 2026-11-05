"""Download today's actual shadow artifacts, fetch outcomes, and generate reviews.

Runs on an independent post-close schedule. The original report/Sheet schedules
are untouched. A missing pre-open run remains missing; never generate it here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, time, timedelta
from pathlib import Path

from scripts.monitor_shadow import monitor
from scripts.verify_shadow_artifact import assess
from tw_scenario.morning_adapter import (
    close_export,
    verify_readback,
    verify_review_readback,
)
from tw_scenario.postclose import collect_day
from tw_scenario.shadow import now, seal
from tw_scenario.snapshot import verify_document, write_immutable_json
from tw_scenario.split_forecast import TAIPEI


def api(url, binary=False):
    if not url.startswith("https://api.github.com/repos/"):
        raise ValueError("unsupported GitHub API destination")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "tw-scenario-shadow-close/1"}
    if token := os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    if not binary:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        urllib.request.build_opener(NoRedirect).open(request, timeout=20)
    except urllib.error.HTTPError as exc:
        if exc.code != 302:
            raise
        destination = exc.headers["Location"]
    else:
        raise ValueError("expected signed GitHub artifact redirect")
    if not destination.startswith("https://"):
        raise ValueError("artifact download must use HTTPS")
    # New request: never forward GitHub credentials to blob storage.
    with urllib.request.urlopen(destination, timeout=30) as response:
        archive = response.read(15_000_001)
    if len(archive) > 15_000_000:
        raise ValueError("oversized shadow artifact")
    return archive


def materialize(metadata, archive, directory):
    """Check remote digest and bound paths/sizes before extracting any bytes."""
    if metadata.get("digest") != "sha256:" + hashlib.sha256(archive).hexdigest():
        raise ValueError("downloaded artifact digest mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        entries = zipped.infolist()
        if len(entries) > 100 or sum(e.file_size for e in entries) > 15_000_000:
            raise ValueError("oversized uncompressed shadow artifact")
        paths = [e.filename for e in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate archive paths")
        for entry in entries:
            path = Path(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError("unsafe artifact path")
            if entry.is_dir():
                continue
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(zipped.read(entry))


def download_today(repository, output, at):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository identity")
    day = at.astimezone(TAIPEI).date()
    start = datetime.combine(day, time(0), TAIPEI)
    stop = start + timedelta(days=1)
    created = f"{start.isoformat()}..{stop.isoformat()}"
    base = f"https://api.github.com/repos/{repository}"
    index = api(base + "/actions/workflows/preopen-shadow.yml/runs?" + urllib.parse.urlencode({
        "created": created, "per_page": 100,
    }))
    if index["total_count"] > 100:
        raise ValueError("bounded run listing exceeded; no partial denominator")
    runs, evidence = [], []
    for run in sorted(index["workflow_runs"], key=lambda r: r["id"]):
        directory = output / f"run-{run['id']}"
        try:
            if not start <= datetime.fromisoformat(run["created_at"]) < stop:
                raise ValueError("API returned a run from another trading date")
            if run["status"] != "completed":
                raise ValueError("preopen workflow still incomplete")
            payload = api(base + f"/actions/runs/{run['id']}/artifacts?per_page=100")
            if payload["total_count"] > 100:
                raise ValueError("artifact listing truncated")
            matches = [a for a in payload["artifacts"]
                       if a["name"].startswith(f"preopen-shadow-{run['id']}-")]
            if len(matches) != 1:
                raise ValueError("missing/duplicate preopen artifacts; do not choose a rerun")
            metadata = matches[0]
            if metadata.get("expired") or metadata.get("workflow_run", {}).get("id") != run["id"]:
                raise ValueError("expired artifact or workflow identity mismatch")
            archive = api(base + f"/actions/artifacts/{metadata['id']}/zip", binary=True)
            materialize(metadata, archive, directory)
            write_immutable_json({"run": run, "artifact": metadata, "received_at": now().isoformat()},
                                 directory / "download-evidence.json")
            if (directory / "failure.json").exists():
                evidence.append({"run_id": run["id"], "status": "preopen_failed"})
            else:
                forecast = json.loads((directory / "forecast.json").read_text())
                receipt = json.loads((directory / "local-receipt.json").read_text())
                if forecast["trading_date"] != day.isoformat():
                    raise ValueError("artifact forecast is from another date")
                remote = assess(metadata, archive, forecast, receipt, str(metadata["id"]),
                                metadata["digest"], repository)
                write_immutable_json(remote, directory / "close-remote-readback.json")
                evidence.append({"run_id": run["id"], **remote})
        except Exception as exc:  # noqa: BLE001 -- preserve failed run in the denominator
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / "failure.json").exists():
                write_immutable_json({"status": "failed", "recorded_at": now().isoformat(),
                                      "reason": str(exc), "eligible_for_formal_accuracy": False},
                                     directory / "failure.json")
            evidence.append({"run_id": run["id"], "status": "download_or_timing_failed", "reason": str(exc)})
        runs.append(directory)
    return runs, evidence


def finish_local(runs, output, at):
    """No historical forecast synthesis. Only frozen matching forecasts get labels."""
    actual_index, failures, cache, reviews = {}, [], {}, []
    for directory in runs:
        if (directory / "failure.json").exists():
            continue
        try:
            forecast = json.loads((directory / "forecast.json").read_text())
            if not verify_document(forecast).valid:
                raise ValueError("invalid downloaded numerical snapshot")
            if forecast["trading_date"] != at.astimezone(TAIPEI).date().isoformat():
                raise ValueError("post-close must match today's forecast, never yesterday's")
            contract = forecast["markets"]["TX_selected_monthly"]["inputs"]["contract"]
            key = (forecast["trading_date"], contract)
            if key not in cache:
                cache[key] = collect_day(at.astimezone(TAIPEI).date(), contract,
                                         output / "sources" / (key[0] + "-" + contract))
                failures.extend(cache[key]["failures"])
            bundle = cache[key]
            for actual in bundle["actuals"]:
                identity = (actual["trading_date"], actual["market"], actual["contract"])
                if identity in actual_index:
                    previous = actual_index[identity]
                    if any(previous[f] != actual[f] for f in ("open", "high", "low", "close")):
                        raise ValueError("official outcomes conflict across downloads")
                else:
                    actual_index[identity] = actual
            morning_path = directory / "morning-export.json"
            if not morning_path.exists():
                reviews.append({"run": str(directory), "status": "morning_export_missing",
                                "reason": "never backfill a report as if written before open"})
                continue
            morning = json.loads(morning_path.read_text())
            observed_rows = [r["values"] for r in morning["records"]]
            verify_readback(morning, observed_rows)
            patch = close_export(forecast, morning, observed_rows, bundle["actuals"], now())
            # Local round trip only; this is not evidence of a Google Sheets write.
            updated = [list(row) for row in observed_rows]
            for row, entry in zip(updated, patch["records"]):
                for column, value in entry["patch_by_zero_based_column"].items():
                    row[int(column)] = value
            verify_review_readback(patch, updated)
            write_immutable_json(patch, output / "reviews" / f"{directory.name}.json")
            reviews.append({"run": str(directory), "status": "local_review_verified",
                            "live_sheet_write": False})
        except Exception as exc:  # noqa: BLE001 -- keep actual/review failures, never silently pass
            failures.append({"run": str(directory), "failure_type": type(exc).__name__, "reason": str(exc)})
    actuals = list(actual_index.values())
    write_immutable_json(actuals, output / "actuals.json")
    report = monitor(runs, actuals)
    write_immutable_json(report, output / "monitor.json")
    return {"actual_count": len(actuals), "failures": failures, "reviews": reviews,
            "monitor_scored_groups": len(report["groups"]),
            "monitor_failures": report["failed_runs"], "pending_labels": report["pending_labels"]}


def aggregate_recent(repository, current_runs, current_actuals, output, at):
    """Accumulate daily artifacts without recursively storing old archives.

    Past artifacts are diagnostics, not new independent time attestations. The
    original preopen bytes and remote evidence remain in the originating run.
    """
    base = f"https://api.github.com/repos/{repository}"
    index = api(base + "/actions/workflows/shadow-close.yml/runs?status=completed&per_page=50")
    issues, runs, actuals = [], list(current_runs), list(current_actuals)
    today = at.astimezone(TAIPEI).date()
    for run in index["workflow_runs"]:
        day = datetime.fromisoformat(run["created_at"]).astimezone(TAIPEI).date()
        if not today - timedelta(days=60) <= day < today:
            continue
        try:
            payload = api(base + f"/actions/runs/{run['id']}/artifacts?per_page=100")
            if payload["total_count"] > 100:
                raise ValueError("historical close artifact listing truncated")
            matches = [m for m in payload["artifacts"]
                       if m["name"].startswith(f"shadow-close-{run['id']}-")]
            if len(matches) != 1:
                raise ValueError("missing/duplicate historical close artifacts")
            metadata = matches[0]
            if metadata.get("expired") or metadata.get("workflow_run", {}).get("id") != run["id"]:
                raise ValueError("historical artifact expired or belongs to another workflow run")
            raw = api(base + f"/actions/artifacts/{metadata['id']}/zip", binary=True)
            if metadata.get("digest") != "sha256:" + hashlib.sha256(raw).hexdigest():
                raise ValueError("historical close artifact digest mismatch")
            # Only fixed JSON members: never extract raw quotes or nested histories.
            directory = output / f"close-{run['id']}"
            directory.mkdir(parents=True, exist_ok=False)
            with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
                names = zipped.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("duplicate historical archive paths")
                chosen = [n for n in names if n in ("actuals.json", "job.json") or re.fullmatch(
                    r"runs/run-\d+/(forecast|local-receipt|failure)\.json", n)]
                if len(chosen) > 50 or sum(zipped.getinfo(n).file_size for n in chosen) > 2_000_000:
                    raise ValueError("historical diagnostic members oversized")
                for name in chosen:
                    target = directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(zipped.read(name))
            if not (directory / "job.json").exists():
                raise ValueError("historical close receipt missing")
            job = json.loads((directory / "job.json").read_text())
            if not verify_document(job).valid:
                raise ValueError("historical close receipt hash failed")
            if (directory / "actuals.json").exists():
                actuals.extend(json.loads((directory / "actuals.json").read_text()))
            old_runs = sorted((directory / "runs").glob("run-*"))
            runs.extend(old_runs)
            if not old_runs or job["status"] != "diagnostic_complete":
                issues.append({"run_id": run["id"], "status": job["status"]})
        except Exception as exc:  # noqa: BLE001 -- retain all historical download failures
            issues.append({"run_id": run["id"], "reason": str(exc)})
    deduplicated = {}
    for record in sorted(actuals, key=lambda r: r["available_at"]):
        identity = (record["trading_date"], record["market"], record["contract"])
        if identity in deduplicated:
            if any(deduplicated[identity][f] != record[f] for f in ("open", "high", "low", "close")):
                raise ValueError("conflicting actual revisions; cumulative scoring stopped")
        else:
            deduplicated[identity] = record
    # Re-running a close job can archive the very same preopen run again. Those
    # references are deduplicated by run ID plus exact numerical/receipt bytes;
    # distinct forecasts on the same day remain subject to monitor's rejection.
    references, unique_runs, repeated = set(), [], []
    for directory in runs:
        hashes = tuple(hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       if (directory / name).exists() else None
                       for name in ("forecast.json", "local-receipt.json", "failure.json"))
        identity = (directory.name, hashes)
        if identity in references:
            repeated.append(str(directory))
        else:
            references.add(identity)
            unique_runs.append(directory)
    report = monitor(unique_runs, list(deduplicated.values()))
    report.update({"history_issues": issues, "history_lookup_limit": 50,
                   "history_listing_may_be_truncated": index["total_count"] > 50,
                   "history_calendar_days": 60, "all_time_accuracy_claim": False,
                   "repeated_archive_references": repeated})
    return report


def main(repository, output):
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": "shadow-close-job-v1", "recorded_at": now().isoformat(),
              "formal_accuracy_eligible": False, "live_sheet_write": False}
    try:
        at = now()
        if at.time() < time(14):
            raise ValueError("post-close job must wait until 14:00 Taipei")
        runs, evidence = download_today(repository, output / "runs", at)
        result["remote_evidence"] = evidence
        if not runs:
            result.update({"status": "missing_preopen", "attempted_runs": 0,
                           "reason": "no preopen workflow run; no forecast/accuracy created"})
        else:
            result.update(finish_local(runs, output, at))
            current_actuals = json.loads((output / "actuals.json").read_text())
            try:
                accumulated = aggregate_recent(repository, runs, current_actuals,
                                               output.parent / "history", at)
                write_immutable_json(accumulated, output / "longitudinal-monitor.json")
                result["longitudinal_groups"] = len(accumulated["groups"])
                result["history_issues"] = accumulated["history_issues"]
            except Exception as exc:  # noqa: BLE001 -- daily diagnostics survive history API failure
                result["history_issues"] = [{"reason": str(exc)}]
            bad = (result["failures"] or result["monitor_failures"] or result["pending_labels"]
                   or result["history_issues"] or any(
                       r["status"] != "local_review_verified" for r in result["reviews"]))
            result["status"] = "failed_or_incomplete" if bad else "diagnostic_complete"
    except Exception as exc:  # noqa: BLE001 -- preserve setup/API failures in the job artifact
        result.update({"status": "failed", "failure_type": type(exc).__name__, "reason": str(exc)})
    write_immutable_json(seal(result), output / "job.json")
    print(json.dumps({k: result[k] for k in ("status", "recorded_at")}))
    return 0 if result["status"] == "diagnostic_complete" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="fu06pi/tw-market-scenario-engine")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.repository, args.output))
