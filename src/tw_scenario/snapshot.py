from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import PredictionSnapshot

HASH_VERSION = "sha256-json-v2"


def document_hash(payload: dict[str, Any]) -> str:
    """Hash the persisted JSON representation, excluding only its hash field.

    V2 writes its version and typed numeric representation before hashing. Legacy
    documents are checked as stored; never coerce integers or rewrite old hashes.
    """
    body = {k: v for k, v in payload.items() if k != "snapshot_hash"}
    raw = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class IntegrityResult:
    valid: bool
    method: str
    raw_match: bool
    normalized_match: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def verify_document(payload: dict[str, Any]) -> IntegrityResult:
    """Read-only, explicit legacy compatibility. A hash is not timing evidence."""
    expected = payload.get("snapshot_hash")
    raw_match = bool(expected) and document_hash(payload) == expected
    version = payload.get("hash_version")
    if version is not None:
        valid = version == HASH_VERSION and raw_match
        return IntegrityResult(valid, version if valid else "unsupported_or_invalid", raw_match)
    # New schemas must never silently fall back to an unversioned legacy hash.
    if "schema_version" in payload:
        return IntegrityResult(False, "missing_hash_version", raw_match)
    snapshot = PredictionSnapshot.model_validate(payload)
    normalized_match = verify_snapshot(snapshot)

    # Pydantic's legacy model ignores extras. Do not let unsigned added fields
    # pass solely because normalization discarded them.
    def no_discarded_keys(original, normalized):
        if isinstance(original, dict) and isinstance(normalized, dict):
            return set(original) <= set(normalized) and all(
                no_discarded_keys(v, normalized[k]) for k, v in original.items()
            )
        if isinstance(original, list) and isinstance(normalized, list):
            return len(original) == len(normalized) and all(
                no_discarded_keys(a, b) for a, b in zip(original, normalized)
            )
        return True

    no_unknown_fields = no_discarded_keys(payload, snapshot.model_dump(mode="json"))
    valid = raw_match or (normalized_match and no_unknown_fields)
    method = "legacy_raw" if raw_match else "legacy_normalized" if valid else "invalid"
    return IntegrityResult(valid, method, raw_match, normalized_match)


def write_immutable_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Exclusive creation also prevents two writers replacing the same snapshot."""
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return path


def freeze_document(payload: dict[str, Any], path: str | Path) -> Path:
    body = dict(payload)
    body["hash_version"] = HASH_VERSION
    body["snapshot_hash"] = document_hash(body)
    return write_immutable_json(body, path)


def canonical_payload(snapshot: PredictionSnapshot) -> dict:
    payload = snapshot.model_dump(mode="json")
    payload.pop("snapshot_hash", None)
    return payload


def compute_snapshot_hash(snapshot: PredictionSnapshot) -> str:
    raw = json.dumps(
        canonical_payload(snapshot), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def freeze_snapshot(snapshot: PredictionSnapshot, path: str | Path) -> Path:
    snapshot.snapshot_hash = compute_snapshot_hash(snapshot)
    return write_immutable_json(snapshot.model_dump(mode="json"), path)


def verify_snapshot(snapshot: PredictionSnapshot) -> bool:
    if not snapshot.snapshot_hash:
        return False
    return snapshot.snapshot_hash == compute_snapshot_hash(snapshot)
