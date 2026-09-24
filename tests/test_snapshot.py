from pathlib import Path

from tw_scenario.models import PredictionSnapshot
from tw_scenario.snapshot import freeze_snapshot, verify_snapshot


def test_sample_contains_required_narrative():
    sample = Path("examples/sample_prediction.json")
    snapshot = PredictionSnapshot.model_validate_json(sample.read_text(encoding="utf-8"))
    assert snapshot.narrative.open_hypothesis
    assert snapshot.narrative.intraday_hypothesis
    assert snapshot.narrative.close_hypothesis


def test_freeze_and_verify(tmp_path: Path):
    sample = Path("examples/sample_prediction.json")
    snapshot = PredictionSnapshot.model_validate_json(sample.read_text(encoding="utf-8"))
    output = tmp_path / "prediction.json"
    freeze_snapshot(snapshot, output)
    frozen = PredictionSnapshot.model_validate_json(output.read_text(encoding="utf-8"))
    assert verify_snapshot(frozen)
