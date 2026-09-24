from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import PredictionSnapshot


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable prediction already exists: {path}")
    snapshot.snapshot_hash = compute_snapshot_hash(snapshot)
    path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def verify_snapshot(snapshot: PredictionSnapshot) -> bool:
    if not snapshot.snapshot_hash:
        return False
    return snapshot.snapshot_hash == compute_snapshot_hash(snapshot)
