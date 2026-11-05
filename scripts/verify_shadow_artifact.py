"""Read GitHub's artifact API; a local snapshot hash alone is not time evidence."""

import argparse
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from tw_scenario.shadow import now
from tw_scenario.snapshot import verify_document, write_immutable_json


def assess(metadata, archive, forecast, local_receipt, expected_id, expected_digest, repository):
    if not verify_document(forecast).valid:
        raise ValueError("invalid saved forecast hash")
    if metadata["id"] != int(expected_id) or metadata.get("expired"):
        raise ValueError("artifact identity/expiry mismatch")
    if metadata["url"] != f"https://api.github.com/repos/{repository}/actions/artifacts/{expected_id}":
        raise ValueError("artifact is from another repository")
    digest = "sha256:" + hashlib.sha256(archive).hexdigest()
    if not expected_digest.startswith("sha256:"):
        expected_digest = "sha256:" + expected_digest
    if digest != metadata.get("digest") or digest != expected_digest:
        raise ValueError("artifact archive digest mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        stored = json.loads(zipped.read("forecast.json"))
        stored_receipt = json.loads(zipped.read("local-receipt.json"))
        if stored != forecast or stored_receipt != local_receipt:
            raise ValueError("remote bytes differ from locally frozen numbers/receipt")
        if "failure.json" in zipped.namelist():
            raise ValueError("failed attempt cannot become an on-time forecast")
    deadline = datetime.fromisoformat(forecast["deadline"])
    created = datetime.fromisoformat(metadata["created_at"])
    uploaded = datetime.fromisoformat(metadata["updated_at"])
    generated = datetime.fromisoformat(forecast["generated_at"])
    persisted = datetime.fromisoformat(local_receipt["local_persisted_at"])
    if local_receipt["snapshot_hash"] != forecast["snapshot_hash"]:
        raise ValueError("receipt hash differs")
    if not generated <= persisted <= created <= uploaded < deadline:
        raise ValueError("remote numerical persistence missed deadline or chronology disagrees")
    return {"remote_persistence_verified": True, "artifact_id": metadata["id"],
            "created_at": metadata["created_at"], "updated_at": metadata["updated_at"],
            "artifact_digest": digest, "forecast_hash": forecast["snapshot_hash"],
            "scope": "this artifact only; calendar/labels/model accuracy not certified"}


def main(run_dir, output):
    result = {"recorded_at": now().isoformat(), "remote_persistence_verified": False}
    try:
        artifact_id = os.environ["ARTIFACT_ID"]
        repository = os.environ["REPOSITORY"]
        token = os.environ["GH_TOKEN"]
        # The token is sent only to api.github.com, never through redirect chains.
        url = f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}"
        request = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        })
        with urllib.request.urlopen(request, timeout=20) as response:
            metadata = json.load(response)
        request = urllib.request.Request(url + "/zip", headers={"Authorization": f"Bearer {token}"})

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        try:
            urllib.request.build_opener(NoRedirect).open(request, timeout=20)
        except urllib.error.HTTPError as exc:
            if exc.code != 302:
                raise
            signed_url = exc.headers["Location"]
        else:
            raise ValueError("expected a GitHub signed archive redirect")
        with urllib.request.urlopen(signed_url, timeout=30) as response:
            archive = response.read(15_000_001)
        if len(archive) > 15_000_000:
            raise ValueError("oversized shadow archive")
        forecast = json.loads((run_dir / "forecast.json").read_text())
        receipt = json.loads((run_dir / "local-receipt.json").read_text())
        result.update(assess(metadata, archive, forecast, receipt, artifact_id,
                             os.environ["ARTIFACT_DIGEST"], repository))
        result["metadata"] = metadata
    except Exception as exc:  # noqa: BLE001 -- persist failed API/readback attempts
        result.update({"failure_type": type(exc).__name__, "reason": str(exc)})
    write_immutable_json(result, output)
    return 0 if result["remote_persistence_verified"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.run_dir, args.output))
