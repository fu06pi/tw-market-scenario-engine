"""Synthetic execution-contract tests; not pretrained TimesFM accuracy evidence."""
import hashlib
import json
import time
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from tw_scenario.quant import head_daily
from tw_scenario.quant.cli import app
from tw_scenario.quant.contracts import TZ
from tw_scenario.quant.daily import load_snapshot
from tw_scenario.quant.timesfm_head_training import (
    CODE_COMMIT,
    WEIGHT_REPO,
    WEIGHT_REVISION,
    HeadSample,
    HeadTrainingSpec,
    train_output_head,
)


@pytest.mark.parametrize("joint", [False, True])
def test_synthetic_trained_checkpoint_to_dual_market_frozen_snapshot(tmp_path, monkeypatch, joint):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file

    class TinyModel(torch.nn.Module):
        input_patch_len = 32
        output_patch_len = 64
        quantiles = tuple(np.arange(1, 10) / 10)

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Parameter(torch.tensor(1.0))
            self.output_head = torch.nn.Linear(1, 9)

        def forward(self, inputs, patch_cpm_mask):
            return {"logits": self.output_head(
                inputs["values"][:, :, :, :1] * self.backbone).unsqueeze(-2)}

    model = TinyModel()
    sample = HeadSample(day=date(2025, 1, 2), history=np.full((32, 8 if joint else 4), .01, dtype=np.float32),
                        past_features=np.ones((32, 4 if joint else 2), dtype=np.float32),
                        forecast_features=np.ones(4 if joint else 2, dtype=np.float32),
                        target=np.zeros(8 if joint else 4, dtype=np.float32), previous_close=100., actual={}, contract="")
    training = train_output_head(model, [sample], [sample],
                                HeadTrainingSpec(32, .001, 0, max_epochs=1),
                                seed=4655, deadline=time.monotonic() + 30)
    checkpoint = tmp_path / "head.safetensors"
    save_file(model.output_head.state_dict(), str(checkpoint))
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {"schema_version": "timesfm3-custom-output-head-training-v1",
              "weight_repo": WEIGHT_REPO, "training": training,
              "checkpoint": {"sha256": checkpoint_hash},
              "identity": {"timesfm_code_commit": CODE_COMMIT,
                           "timesfm_weight_revision": WEIGHT_REVISION,
                           "train_end": "2025-01-02", "early_stop_split": ["2025-01-02", "2025-01-02"],
                           "feature_names": ({m: [f"{m.lower()}__{n}" for n in ("a", "b")]
                                              for m in ("TAIEX", "TX")} if joint else ["a", "b"]),
                           "spec": {"context": 32}}}
    bundle = {}
    for market in (("JOINT",) if joint else ("TAIEX", "TX")):
        market_result = {**result, "identity": {**result["identity"], "instrument": market}}
        (tmp_path / f"{market}.json").write_text(json.dumps(market_result))
        bundle[market] = {"result": f"{market}.json", "checkpoint": checkpoint.name}
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    (tmp_path / "manifest.json").write_text('{"dataset_id":"synthetic"}')

    class Preopen:
        calendar = SimpleNamespace(previous=lambda day: date(2025, 1, 3))

        def research_feature_names(self):
            return ["a", "b"]

        def research_history(self, *args):
            return sample.history[:, :4], sample.past_features[:, :2], [date(2025, 1, 3)]

        def research_features(self, *args):
            return sample.forecast_features[:2]

        def reference(self, market, day):
            return SimpleNamespace(close=100. if market == "TAIEX" else 200., source_hash="a" * 64,
                                   available_at=datetime(2025, 1, 3, 14, tzinfo=TZ))

        def contract(self, market, day):
            return "202501" if market == "TX" else ""

        def get(self, *args):
            return SimpleNamespace(source_hash="b" * 64,
                                   available_at=datetime(2025, 1, 6, 5, tzinfo=TZ))

        def actual(self, *args):
            pytest.fail("forecast must not read day labels")

    monkeypatch.setattr(head_daily, "load_dataset", lambda path: (Preopen(), "synthetic"))
    monkeypatch.setattr(head_daily, "runtime_probe", lambda path: SimpleNamespace(status="READY_TO_LOAD"))
    monkeypatch.setattr(head_daily, "load_pinned_forecaster", lambda *a, **k: SimpleNamespace(model=model))
    output = tmp_path / "snapshot.json"
    args = ["timesfm-head-daily", str(tmp_path), "--bundle", str(bundle_path),
            "--date", "2025-01-06", "--output", str(output)]
    response = CliRunner().invoke(app, args)
    assert response.exit_code == 0, response.output
    snapshot = load_snapshot(output)
    assert [f.instrument for f in snapshot.forecasts] == ["TAIEX", "TX"]
    assert snapshot.forecasts[1].previous_close == 200
    assert snapshot.forecasts[1].contract == "202501"
    assert all(f.model.checkpoint_id == checkpoint_hash for f in snapshot.forecasts)
    assert snapshot.timeliness == "LATE"
    assert snapshot.data_quality.status == "WARN"
    assert "uncalibrated" in snapshot.forecasts[0].intervals.method
    original = output.read_bytes()
    assert CliRunner().invoke(app, args).exit_code != 0
    assert output.read_bytes() == original
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    failed_output = tmp_path / "failed.json"
    response = CliRunner().invoke(app, [*args[:-1], str(failed_output)])
    assert response.exit_code != 0
    assert not failed_output.exists()
