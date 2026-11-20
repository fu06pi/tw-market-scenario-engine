"""Generate explicitly synthetic inputs for the split-forecast CLI.

Run from the repository with its installed environment:
  python examples/split_demo.py /tmp/tw-split-demo
No output here is historical market data or evidence of forecast accuracy.
"""

import sys
from datetime import date, timedelta

from tw_scenario.snapshot import write_immutable_json
from tw_scenario.split_forecast import (
    Components,
    ObservedBar,
    PreopenInputs,
    SplitConfig,
    TimedFeature,
    TrainingExample,
)


def synthetic_inputs(day, night, body):
    previous = day - timedelta(days=1)
    features = []
    for name, value, kind, observed in (
        ("night_return_bps", night, "completed_overnight", f"{day}T05:00:00+08:00"),
        ("previous_body_bps", body, "historical", f"{previous}T15:00:00+08:00"),
        ("previous_range_bps", 20, "historical", f"{previous}T15:00:00+08:00"),
    ):
        features.append(
            TimedFeature(
                name=name,
                value=value,
                kind=kind,
                observed_at=observed,
                available_at=observed,
                source="synthetic demonstration",
                availability_basis="synthetic; not historical timing evidence",
            )
        )
    return PreopenInputs(
        forecast_id=f"synthetic-{day}",
        trading_date=day,
        market="TAIEX",
        previous_close=100,
        previous_close_date=previous,
        previous_close_available_at=f"{previous}T15:00:00+08:00",
        previous_close_source="synthetic demonstration",
        previous_close_availability_basis="synthetic; not historical timing evidence",
        data_cutoff_at=f"{day}T08:30:00+08:00",
        generated_at=f"{day}T08:35:00+08:00",
        data_version="synthetic-v1",
        feature_version="synthetic-v1",
        features=tuple(features),
    )


def synthetic_actual(inputs, night, body):
    prices = Components(g=0.002 * night, b=0.004 * body, u=0.003, d=0.004).reconstruct(100)
    return ObservedBar(
        **prices.model_dump(),
        trading_date=inputs.trading_date,
        market="TAIEX",
        available_at=f"{inputs.trading_date}T15:00:00+08:00",
        source="synthetic demonstration",
        availability_basis="synthetic; not market data",
    )


def main(directory):
    from pathlib import Path

    directory = Path(directory)
    examples = []
    day = date(2026, 6, 1)
    while len(examples) < 80:
        if day.weekday() < 5:
            i = len(examples)
            night, body = (i % 11) - 5, (i % 7) - 3
            inputs = synthetic_inputs(day, night, body)
            examples.append(
                TrainingExample(inputs=inputs, actual=synthetic_actual(inputs, night, body))
            )
        day += timedelta(days=1)
    request = synthetic_inputs(date(2026, 10, 1), 1, -4)
    # write_immutable_json supports JSON arrays at runtime, but keep the public
    # document writer's dict contract by explicitly serializing the dataset here.
    import json

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.json").open("x", encoding="utf-8") as stream:
        json.dump(
            [e.model_dump(mode="json") for e in examples], stream, ensure_ascii=False, indent=2
        )
    write_immutable_json(request.model_dump(mode="json"), directory / "request.json")
    write_immutable_json(SplitConfig().model_dump(mode="json"), directory / "config.json")
    write_immutable_json(
        synthetic_actual(request, 1, -4).model_dump(mode="json"), directory / "actual.json"
    )
    print(f"Synthetic files created in {directory}; not accuracy evidence.")


if __name__ == "__main__":
    main(sys.argv[1])
