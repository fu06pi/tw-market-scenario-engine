from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

CODE_COMMIT = "e51928e27119cb17bebc005be2696b75e0a9e688"
PACKAGE_VERSION = "3.0.2"
WEIGHT_REPO = "google/timesfm-3.0-pytorch"
WEIGHT_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
WEIGHT_BYTES = 1_322_898_824
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ForecastOutput(Protocol):
    forecast: np.ndarray | None
    quantiles: np.ndarray | None


class Forecaster(Protocol):
    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastOutput: ...


@dataclass(frozen=True)
class RuntimeProbe:
    status: Literal[
        "READY_TO_LOAD",
        "DEPENDENCY_MISSING",
        "BLOCKED_INSUFFICIENT_MEMORY",
        "BLOCKED_INSUFFICIENT_DISK",
    ]
    available_ram_mb: int
    swap_free_mb: int
    disk_free_mb: int
    timesfm3_installed: bool
    torch_installed: bool
    device: Literal["cpu"] = "cpu"
    per_core_batch_size: Literal[1] = 1


@dataclass(frozen=True)
class TimesFM3Prediction:
    mode: Literal["zero_shot_univariate", "zero_shot_multivariate"]
    point: np.ndarray
    quantiles: np.ndarray
    code_commit: str = CODE_COMMIT
    weight_repo: str = WEIGHT_REPO
    weight_revision: str = WEIGHT_REVISION

    def __post_init__(self) -> None:
        if self.point.shape != (4,):
            raise ValueError("TimesFM OHLC target point must have shape (4,)")
        if self.quantiles.shape != (4, len(QUANTILES)):
            raise ValueError("TimesFM OHLC target quantiles must have shape (4, 9)")
        if not np.all(np.isfinite(self.point)) or not np.all(np.isfinite(self.quantiles)):
            raise ValueError("TimesFM output must be finite")


@dataclass(frozen=True)
class TimesFM3JointPrediction:
    mode: Literal["zero_shot_joint_multivariate"]
    point: np.ndarray
    quantiles: np.ndarray
    code_commit: str = CODE_COMMIT
    weight_repo: str = WEIGHT_REPO
    weight_revision: str = WEIGHT_REVISION

    def __post_init__(self) -> None:
        if self.point.shape != (8,) or self.quantiles.shape != (8, len(QUANTILES)):
            raise ValueError("joint TAIEX/TX output must have shapes (8,) and (8, 9)")
        if not np.all(np.isfinite(self.point)) or not np.all(np.isfinite(self.quantiles)):
            raise ValueError("joint TimesFM output must be finite")


def _meminfo() -> tuple[int, int]:
    values: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) // 1024
    return values.get("MemAvailable", 0), values.get("SwapFree", 0)


def runtime_probe(cache_dir: Path, minimum_available_ram_mb: int = 3072) -> RuntimeProbe:
    """Inspect resources without importing torch or loading/downloading model weights."""
    available_ram_mb, swap_free_mb = _meminfo()
    disk_free_mb = shutil.disk_usage(cache_dir.resolve().parent).free // (1024 * 1024)
    timesfm3_installed = importlib.util.find_spec("timesfm3") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    if available_ram_mb < minimum_available_ram_mb:
        status = "BLOCKED_INSUFFICIENT_MEMORY"
    elif disk_free_mb < (WEIGHT_BYTES * 2) // (1024 * 1024):
        status = "BLOCKED_INSUFFICIENT_DISK"
    elif not (timesfm3_installed and torch_installed):
        status = "DEPENDENCY_MISSING"
    else:
        status = "READY_TO_LOAD"
    return RuntimeProbe(
        status=status,
        available_ram_mb=available_ram_mb,
        swap_free_mb=swap_free_mb,
        disk_free_mb=disk_free_mb,
        timesfm3_installed=timesfm3_installed,
        torch_installed=torch_installed,
    )


def installed_provenance() -> dict:
    """Fail closed unless the installed package records the pinned Git commit."""
    distribution = importlib.metadata.distribution("timesfm")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return {
        "package_version": distribution.version,
        "direct_url": direct_url.get("url"),
        "vcs_commit": commit,
        "version_matches": distribution.version == PACKAGE_VERSION,
        "commit_matches": commit == CODE_COMMIT,
    }


def load_pinned_forecaster(cache_dir: Path, local_files_only: bool = False) -> Forecaster:
    probe = runtime_probe(cache_dir)
    if probe.status != "READY_TO_LOAD":
        raise RuntimeError(f"TimesFM 3 load blocked by preflight: {probe.status}")
    provenance = installed_provenance()
    if not provenance["version_matches"] or not provenance["commit_matches"]:
        raise RuntimeError("TimesFM package provenance does not match the frozen code commit")
    from timesfm3 import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(
        WEIGHT_REPO,
        device="cpu",
        revision=WEIGHT_REVISION,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        per_core_batch_size=1,
    )


class TimesFM3Adapter:
    """Pinned inference-only TimesFM 3 adapter for four transformed OHLC targets."""

    def __init__(self, forecaster: Forecaster):
        self.forecaster = forecaster

    @staticmethod
    def _history(values: np.ndarray, columns: int = 4) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != columns or values.shape[0] < 2:
            raise ValueError(f"history must have shape (time>=2, {columns})")
        if not np.all(np.isfinite(values)):
            raise ValueError("TimesFM history cannot contain missing/non-finite values")
        return values

    @staticmethod
    def _kwargs() -> dict:
        # Financial transformed targets can be negative, so benchmark make_positive is invalid.
        return {
            "return_quantiles": True,
            "use_symmetric_averaging": False,
            "make_positive": False,
            "sort_quantiles": True,
            "use_znorm": False,
            "padding_mode": "none",
        }

    @staticmethod
    def _output(output: ForecastOutput, expected_point: tuple[int, ...],
                expected_quantiles: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if output.forecast is None or output.quantiles is None:
            raise RuntimeError("TimesFM returned no point or quantile forecast")
        point = np.asarray(output.forecast, dtype=float)
        quantiles = np.asarray(output.quantiles, dtype=float)
        if point.shape != expected_point or quantiles.shape != expected_quantiles:
            raise RuntimeError(
                f"unexpected TimesFM shapes: point={point.shape}, quantiles={quantiles.shape}"
            )
        return point, quantiles

    def zero_shot_univariate(self, history: np.ndarray) -> TimesFM3Prediction:
        values = self._history(history)
        point, quantiles = [], []
        for index in range(4):
            output = self.forecaster.predict(
                values[:, index], horizon=1, **self._kwargs()
            )
            p, q = self._output(output, (1,), (1, len(QUANTILES)))
            point.append(p[0])
            quantiles.append(q[0])
        return TimesFM3Prediction(
            mode="zero_shot_univariate",
            point=np.asarray(point),
            quantiles=np.asarray(quantiles),
        )

    def zero_shot_multivariate(
        self,
        history: np.ndarray,
        past_features: np.ndarray | None = None,
        forecast_features: np.ndarray | None = None,
    ) -> TimesFM3Prediction:
        values = self._history(history)
        kwargs = self._kwargs()
        if (past_features is None) != (forecast_features is None):
            raise ValueError("past and forecast-period covariates must be supplied together")
        if past_features is not None and forecast_features is not None:
            past = np.asarray(past_features, dtype=np.float32)
            future = np.asarray(forecast_features, dtype=np.float32)
            if past.ndim != 2 or past.shape[0] != values.shape[0]:
                raise ValueError("past_features must have shape (time, features)")
            if future.shape != (past.shape[1],):
                raise ValueError("forecast_features must have one known value per feature")
            if not np.all(np.isfinite(past)) or not np.all(np.isfinite(future)):
                raise ValueError("TimesFM covariates cannot contain missing/non-finite values")
            kwargs["past_future_covariates"] = np.concatenate(
                [past.T, future[:, None]], axis=1
            )
        output = self.forecaster.predict(values.T, horizon=1, **kwargs)
        point, quantiles = self._output(output, (4, 1), (4, 1, len(QUANTILES)))
        return TimesFM3Prediction(
            mode="zero_shot_multivariate",
            point=point[:, 0],
            quantiles=quantiles[:, 0, :],
        )

    def zero_shot_joint_multivariate(
        self,
        history: np.ndarray,
        past_features: np.ndarray,
        forecast_features: np.ndarray,
    ) -> TimesFM3JointPrediction:
        values = self._history(history, columns=8)
        past = np.asarray(past_features, dtype=np.float32)
        future = np.asarray(forecast_features, dtype=np.float32)
        if past.ndim != 2 or past.shape[0] != values.shape[0]:
            raise ValueError("joint past_features must have shape (time, features)")
        if future.shape != (past.shape[1],):
            raise ValueError("joint forecast_features must match feature columns")
        if not np.all(np.isfinite(past)) or not np.all(np.isfinite(future)):
            raise ValueError("joint covariates cannot contain missing/non-finite values")
        output = self.forecaster.predict(
            values.T,
            horizon=1,
            past_future_covariates=np.concatenate([past.T, future[:, None]], axis=1),
            **self._kwargs(),
        )
        point, quantiles = self._output(output, (8, 1), (8, 1, len(QUANTILES)))
        return TimesFM3JointPrediction(
            mode="zero_shot_joint_multivariate",
            point=point[:, 0],
            quantiles=quantiles[:, 0, :],
        )


def gradient_smoke_contract() -> dict:
    """Describe the only allowed custom-training claim before any gradient run occurs."""
    return {
        "status": "NOT_RUN",
        "name": "custom_output_head_gradient_smoke",
        "official_timesfm3_finetuning": False,
        "required_evidence": [
            "finite_loss",
            "nonzero_output_head_gradient",
            "output_head_parameter_delta",
            "forecast_delta",
            "save_reload_equivalence",
        ],
    }
