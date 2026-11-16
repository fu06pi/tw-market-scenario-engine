"""Fail-closed CPU numeric-runtime contract for reproducible model replay.

The checkpoint hash alone is insufficient for sub-millipoint replay.  CPU
reductions can change by a few float32 ulps when the PyTorch build, effective
thread counts, or deterministic-kernel settings differ.  Training artifacts
must persist :func:`capture_torch_cpu_runtime`; replay must require an exact
match with :func:`require_matching_torch_cpu_runtime` before loading weights.

This contract prevents an unsupported comparison.  It does not claim that two
different machines with matching metadata are necessarily bitwise identical;
an independent checkpoint reload remains required evidence.
"""
from __future__ import annotations

import hashlib
import platform
from typing import Any

import numpy as np

SCHEMA_VERSION = "torch-cpu-numeric-runtime-v1"
REPRODUCIBLE_INTRAOP_THREADS = 1
REPRODUCIBLE_INTEROP_THREADS = 1


class NumericRuntimeMismatch(RuntimeError):
    """Raised when training and replay do not share a verified numeric path."""


def _torch(torch_module=None):
    if torch_module is not None:
        return torch_module
    import torch

    return torch


def _call(obj: Any, path: str, default=None):
    """Read a nested attribute and call it when callable."""
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return default
        current = getattr(current, name)
    return current() if callable(current) else current


def capture_torch_cpu_runtime(torch_module=None) -> dict:
    """Return the versioned effective runtime identity used by model math."""
    torch = _torch(torch_module)
    build = str(_call(torch, "__config__.show", ""))
    capability = _call(torch, "backends.cpu.get_cpu_capability", None)
    git_version = _call(torch, "version.git_version", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "device": "cpu",
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "torch_git_version": git_version,
        "torch_build_sha256": hashlib.sha256(build.encode("utf-8")).hexdigest(),
        "cpu_capability": capability,
        "intraop_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            _call(torch, "is_deterministic_algorithms_warn_only_enabled", False)
        ),
        "float32_matmul_precision": _call(
            torch, "get_float32_matmul_precision", None
        ),
        "mkldnn_enabled": bool(_call(torch, "backends.mkldnn.enabled", False)),
        "mkldnn_deterministic": bool(
            _call(torch, "backends.mkldnn.deterministic", False)
        ),
    }


def configure_reproducible_torch_cpu(torch_module=None) -> dict:
    """Configure a single-thread deterministic CPU path before model work.

    PyTorch only permits changing the inter-op pool before parallel work starts.
    If another caller initialized a different pool first, fail instead of
    silently recording a runtime that the process did not actually use.
    """
    torch = _torch(torch_module)
    torch.set_num_threads(REPRODUCIBLE_INTRAOP_THREADS)
    try:
        torch.set_num_interop_threads(REPRODUCIBLE_INTEROP_THREADS)
    except RuntimeError as exc:
        if torch.get_num_interop_threads() != REPRODUCIBLE_INTEROP_THREADS:
            raise NumericRuntimeMismatch(
                "PyTorch inter-op pool was initialized before the reproducible "
                "CPU runtime contract"
            ) from exc
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.deterministic = True
    runtime = capture_torch_cpu_runtime(torch)
    required = {
        "intraop_threads": REPRODUCIBLE_INTRAOP_THREADS,
        "interop_threads": REPRODUCIBLE_INTEROP_THREADS,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "mkldnn_deterministic": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": runtime.get(key)}
        for key, expected in required.items()
        if runtime.get(key) != expected
    }
    if mismatches:
        raise NumericRuntimeMismatch(
            f"failed to configure reproducible PyTorch CPU runtime: {mismatches}"
        )
    return runtime


def require_matching_torch_cpu_runtime(expected: dict, torch_module=None) -> dict:
    """Fail unless the current effective runtime exactly matches training."""
    if not isinstance(expected, dict) or expected.get("schema_version") != SCHEMA_VERSION:
        raise NumericRuntimeMismatch(
            "checkpoint lacks a supported PyTorch CPU numeric-runtime contract"
        )
    actual = capture_torch_cpu_runtime(torch_module)
    expected_keys = set(actual)
    if set(expected) != expected_keys:
        missing = sorted(expected_keys - set(expected))
        extra = sorted(set(expected) - expected_keys)
        raise NumericRuntimeMismatch(
            f"numeric-runtime schema fields differ: missing={missing}, extra={extra}"
        )
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in sorted(expected_keys)
        if expected[key] != actual[key]
    }
    if mismatches:
        raise NumericRuntimeMismatch(
            f"checkpoint numeric runtime differs from replay: {mismatches}"
        )
    return actual
