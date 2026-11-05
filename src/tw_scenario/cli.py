from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import TypeAdapter

from .audit import audit_repository, write_audit
from .metrics import evaluate_snapshot
from .models import MarketRealization, PredictionSnapshot
from .policy import load_scoring_policy
from .snapshot import freeze_document, freeze_snapshot, verify_document
from .split_forecast import (
    ObservedBar,
    PreopenInputs,
    SplitConfig,
    SplitForecast,
    TrainingExample,
    evaluate_split,
    forecast_split,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def freeze(input_json: Path, output_json: Path):
    """Freeze a new immutable prediction snapshot."""
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (None, "ohlc-split-v2"):
        raise ValueError("unsupported prediction schema")
    if payload.get("schema_version") == "ohlc-split-v2":
        snapshot = SplitForecast.model_validate(payload)
        path = freeze_document(snapshot.model_dump(mode="json"), output_json)
    else:
        snapshot = PredictionSnapshot.model_validate(payload)
        path = freeze_snapshot(snapshot, output_json)
    typer.echo(str(path))


@app.command()
def verify(prediction_json: Path, details: bool = False):
    """Verify raw legacy/v2 hashes with explicit normalized-legacy compatibility."""
    payload = json.loads(prediction_json.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "ohlc-split-v2":
        SplitForecast.model_validate(payload)
    result = verify_document(payload)
    typer.echo(json.dumps(result.to_dict()) if details else "OK" if result.valid else "FAILED")
    if not result.valid:
        raise typer.Exit(1)


@app.command()
def evaluate(
    prediction_json: Path,
    realization_json: Path,
    policy: Path = Path("config/targets.yaml"),
):
    """Diagnostic scores. Use audit for persistence and formal inclusion checks."""
    payload = json.loads(prediction_json.read_text(encoding="utf-8"))
    scoring = load_scoring_policy(policy)
    if payload.get("schema_version") == "ohlc-split-v2":
        snapshot = SplitForecast.model_validate(payload)
        realization = ObservedBar.model_validate_json(realization_json.read_text(encoding="utf-8"))
        scores = evaluate_split(snapshot, realization, scoring)
    else:
        snapshot = PredictionSnapshot.model_validate(payload)
        realization = MarketRealization.model_validate_json(
            realization_json.read_text(encoding="utf-8")
        )
        scores = evaluate_snapshot(snapshot, realization, scoring)
    scores.update(
        performance_scope="diagnostic_only",
        integrity=verify_document(payload).to_dict(),
        scoring_policy=scoring.model_dump(),
    )
    typer.echo(json.dumps(scores, ensure_ascii=False, indent=2))


@app.command()
def audit(root: Path, output: Path, policy: Path | None = None):
    """Write a new read-only audit ledger; never update historical records."""
    scoring = load_scoring_policy(policy or root / "config/targets.yaml")
    report = audit_repository(root, scoring)
    path = write_audit(report, root, output)
    typer.echo(
        json.dumps(
            {
                "output": str(path),
                "prediction_count": report["prediction_count"],
                "eligible_count": report["eligible_count"],
            }
        )
    )


@app.command("forecast-split")
def forecast_split_command(
    training_json: Path,
    input_json: Path,
    output_json: Path,
    config: Path | None = None,
    policy: Path = Path("config/targets.yaml"),
):
    """Fit independent gap/body research heads on strictly earlier, available bars."""
    examples = TypeAdapter(list[TrainingExample]).validate_json(
        training_json.read_text(encoding="utf-8")
    )
    inputs = PreopenInputs.model_validate_json(input_json.read_text(encoding="utf-8"))
    settings = (
        SplitConfig.model_validate_json(config.read_text(encoding="utf-8"))
        if config
        else (SplitConfig())
    )
    result = forecast_split(examples, inputs, settings, load_scoring_policy(policy))
    path = freeze_document(result.model_dump(mode="json"), output_json)
    typer.echo(str(path))


if __name__ == "__main__":
    app()
