from __future__ import annotations

import json
from pathlib import Path

import typer

from .metrics import evaluate_snapshot
from .models import MarketRealization, PredictionSnapshot
from .snapshot import freeze_snapshot, verify_snapshot

app = typer.Typer(no_args_is_help=True)


@app.command()
def freeze(input_json: Path, output_json: Path):
    """Freeze a new immutable prediction snapshot."""
    snapshot = PredictionSnapshot.model_validate_json(input_json.read_text(encoding="utf-8"))
    path = freeze_snapshot(snapshot, output_json)
    typer.echo(str(path))


@app.command()
def verify(prediction_json: Path):
    """Verify the SHA-256 integrity of a prediction snapshot."""
    snapshot = PredictionSnapshot.model_validate_json(prediction_json.read_text(encoding="utf-8"))
    typer.echo("OK" if verify_snapshot(snapshot) else "FAILED")


@app.command()
def evaluate(prediction_json: Path, realization_json: Path):
    """Evaluate one frozen prediction against realized market data."""
    snapshot = PredictionSnapshot.model_validate_json(prediction_json.read_text(encoding="utf-8"))
    realization = MarketRealization.model_validate_json(realization_json.read_text(encoding="utf-8"))
    typer.echo(json.dumps(evaluate_snapshot(snapshot, realization), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
