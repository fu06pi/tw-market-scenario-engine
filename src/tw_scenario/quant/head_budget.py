"""Durable wall-clock accounting for bounded output-head research searches."""
from __future__ import annotations

import fcntl
import inspect
import json
import math
import os
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from .contracts import digest

_ACTIVE_DEADLINE = ContextVar("head_search_deadline", default=None)


def budget_deadline(fallback: float) -> float:
    current = _ACTIVE_DEADLINE.get()
    return min(current, fallback) if current is not None else fallback


class HeadBudget:
    """Reserve before work; an unclean exit retains its full reservation as spent."""

    def __init__(self, path: Path, run_id: str, total_seconds: float, run_seconds: float):
        if any(not math.isfinite(v) or v <= 0 for v in (total_seconds, run_seconds)):
            raise ValueError("budget limits must be finite and positive")
        self.path, self.run_id = path, run_id
        self.total_seconds, self.run_seconds = total_seconds, run_seconds
        self.result_path = None

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.ledger, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.ledger = (json.loads(self.path.read_text()) if self.path.exists() else {
                "schema_version": "output-head-search-budget-v1",
                "clock": "elapsed_wall_seconds", "total_limit_seconds": self.total_seconds,
                "run_limit_seconds": self.run_seconds, "attempts": [],
            })
            if (self.ledger["total_limit_seconds"] != self.total_seconds
                    or self.ledger["run_limit_seconds"] != self.run_seconds):
                raise ValueError("cannot change a frozen search budget")
            attempts = self.ledger["attempts"]
            used = sum(a["charged_seconds"] for a in attempts)
            run_used = sum(a["charged_seconds"] for a in attempts if a["run_id"] == self.run_id)
            self.allowance = min(self.total_seconds - used, self.run_seconds - run_used)
            if self.allowance <= 0:
                raise TimeoutError("cumulative output-head research budget exhausted")
            self.started = time.monotonic()
            self.deadline = self.started + self.allowance
            self.attempt = {"run_id": self.run_id, "attempt": len(attempts) + 1,
                            "started_unix": time.time(), "reserved_seconds": self.allowance,
                            "charged_seconds": self.allowance,
                            "status": "RESERVED_UNCLEAN_EXIT_CHARGES_FULL_ALLOWANCE"}
            attempts.append(self.attempt)
            self._save()
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.attempt.update(charged_seconds=max(0.0, time.monotonic() - self.started),
                                status="COMPLETED" if exc_type is None else "FAILED",
                                error_type=exc_type.__name__ if exc_type else None,
                                result_path=str(self.result_path) if self.result_path else None)
            self._save()
        finally:
            self.lock.close()
        return False


def bounded_head_run(function):
    """Share a frozen search ledger across markets, candidate indexes and restarts."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        config = json.loads(values["config_path"].read_text())
        # Check resources before reserving time; a blocked preflight is not training.
        from .timesfm3 import runtime_probe
        probe = runtime_probe(values["cache_dir"], int(config["memory_probe_limit_mb"]))
        if probe.status != "READY_TO_LOAD":
            raise RuntimeError(f"TimesFM 3 load blocked by preflight: {probe.status}")
        manifest = json.loads((values["dataset_dir"] / "manifest.json").read_text())
        identity = {"operation": function.__name__, "dataset_id": manifest["dataset_id"],
                    "config_hash": digest(config), "instrument": values["instrument"],
                    "candidate_index": values["candidate_index"]}
        ledger = values["output_root"] / "output-head-search-budget.json"
        with HeadBudget(ledger, digest(identity), float(config["search_budget_seconds"]),
                        float(config["budget_cpu_seconds"])) as budget:
            token = _ACTIVE_DEADLINE.set(budget.deadline)
            try:
                result = function(*args, **kwargs)
                budget.result_path = result
                return result
            finally:
                _ACTIVE_DEADLINE.reset(token)
    return wrapped
