"""One shared official TimesFM 3 decode path for inference and experimental head tuning."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

MODEL = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
CODE_COMMIT = "e51928e27119cb17bebc005be2696b75e0a9e688"


class Forecaster:
    def __init__(self):
        if importlib.metadata.version("timesfm") != "3.0.2":
            raise ValueError("requires audited timesfm==3.0.2")
        torch.set_num_threads(4)
        torch.manual_seed(1729)
        self.evaluator = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=MODEL,
                revision=REVISION,
                cache_dir="/workspace/.cache/huggingface/hub",
                local_files_only=True,
                device="cpu",
                per_core_batch_size=1,
            )
        )
        self.model = self.evaluator.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.original_head = {
            k: v.detach().clone() for k, v in self.model.output_head.state_dict().items()
        }
        self.quantiles = torch.tensor(self.model.quantiles)

    def reset(self):
        self.model.output_head.load_state_dict(self.original_head)

    def logits(
        self, x: np.ndarray | torch.Tensor, grad: bool = False, covariates: np.ndarray | None = None
    ):
        target = torch.as_tensor(x, dtype=torch.float32)
        if target.ndim != 3 or target.shape[1] != 4 or target.shape[2] % 32:
            raise ValueError("expected batch x 4 variates x context multiple of 32")
        if not torch.isfinite(target).all():
            raise ValueError("nonfinite model input")
        kwargs = {}
        if covariates is not None:
            cov = torch.as_tensor(covariates, dtype=torch.float32)
            if cov.shape != (len(target), 1, target.shape[2] + 1):
                raise ValueError("known-night covariate dimensions differ")
            if torch.isinf(cov).any():
                raise ValueError("infinite covariate")
            kwargs = {
                "past_future_covariates": torch.nan_to_num(cov),
                "past_future_mask": torch.isnan(cov),
            }
        if grad:
            # Official decode is decorated no_grad. Invoke its fixed-source implementation
            # with autograd explicitly enabled, retaining identical padding/CPM/detrending.
            return self.model.decode.__wrapped__(self.model, target, horizon=1, **kwargs)[
                :, :, 0, :
            ]
        return self.model.decode(target, horizon=1, **kwargs)[:, :, 0, :]

    def predict(self, x: np.ndarray, batch: int = 8, covariates: np.ndarray | None = None):
        outputs = []
        self.model.eval()
        for start in range(0, len(x), batch):
            outputs.append(
                self.logits(
                    x[start : start + batch],
                    covariates=None if covariates is None else covariates[start : start + batch],
                )
                .detach()
                .numpy()
            )
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite forecast")
        return result

    def verify_paths(self, x: np.ndarray, covariates: np.ndarray | None = None) -> dict:
        self.model.eval()
        with torch.no_grad():
            official = self.logits(x[:1], covariates=None if covariates is None else covariates[:1])
            train_path = self.logits(
                x[:1], grad=True, covariates=None if covariates is None else covariates[:1]
            )
        if not torch.equal(official, train_path):
            raise ValueError("training/official decode mismatch")
        if covariates is not None:
            return {
                "official_vs_autograd_decode": "exact_equal",
                "shape": list(official.shape),
                "public_evaluator_comparison": "core decode masks retained; wrapper interpolation deliberately not used",
                "finite": bool(torch.isfinite(official).all()),
            }
        public = next(
            self.evaluator.predict_batch(
                [x[0]],
                horizon=1,
                return_quantiles=True,
                use_symmetric_averaging=False,
                make_positive=False,
                sort_quantiles=False,
            )
        )
        public_values = np.asarray(public.quantiles)[:, 0, :]
        difference = float(np.max(np.abs(public_values - official[0].numpy())))
        if difference > 1e-6:
            raise ValueError("public evaluator/core decode mismatch")
        return {
            "official_vs_autograd_decode": "exact_equal",
            "public_evaluator_max_abs_error": difference,
            "shape": list(official.shape),
            "finite": True,
        }

    def load_head(self, path: Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("weight_revision") != REVISION:
            raise ValueError("checkpoint/base weight mismatch")
        self.model.output_head.load_state_dict(checkpoint["head"], strict=True)
        return checkpoint
