"""Synthetic fixtures only. These are not market data or accuracy evidence."""

from datetime import date, timedelta
from math import exp

import pytest

from tw_scenario.split_forecast import ObservedBar, PreopenInputs, SplitConfig, TrainingExample


def inputs_for(day: date, night=1.0, body=-4.0, market="TAIEX", contract=None):
    prior = (day - timedelta(days=1)).isoformat()
    current = day.isoformat()
    return PreopenInputs.model_validate(
        {
            "forecast_id": f"synthetic-{current}-{market}",
            "trading_date": current,
            "market": market,
            "contract": contract,
            "previous_close": 100,
            "previous_close_date": prior,
            "previous_close_available_at": f"{prior}T15:00:00+08:00",
            "previous_close_source": "synthetic fixture",
            "previous_close_availability_basis": "synthetic; not historical availability evidence",
            "previous_close_contract": contract,
            "data_cutoff_at": f"{current}T08:30:00+08:00",
            "generated_at": f"{current}T08:35:00+08:00",
            "data_version": "synthetic-v1",
            "feature_version": "synthetic-v1",
            "features": [
                {
                    "name": name,
                    "value": value,
                    "kind": kind,
                    "available_at": observed,
                    "observed_at": observed,
                    "source": "synthetic fixture",
                    "availability_basis": "synthetic, not market history",
                }
                for name, value, kind, observed in [
                    ("night_return_bps", night, "completed_overnight", f"{current}T05:00:00+08:00"),
                    ("previous_body_bps", body, "historical", f"{prior}T15:00:00+08:00"),
                    ("previous_range_bps", 20, "historical", f"{prior}T15:00:00+08:00"),
                ]
            ],
        }
    )


def bar_for(inputs, g=0.002, b=-0.016):
    opening = inputs.previous_close * exp(g)
    closing = opening * exp(b)
    return ObservedBar.model_validate(
        {
            "trading_date": inputs.trading_date,
            "market": inputs.market,
            "contract": inputs.contract,
            "open": opening,
            "close": closing,
            "high": max(opening, closing) * exp(0.003),
            "low": min(opening, closing) * exp(-0.004),
            "available_at": f"{inputs.trading_date}T15:00:00+08:00",
            "source": "synthetic fixture",
            "availability_basis": "synthetic, not market history",
        }
    )


@pytest.fixture
def split_data():
    examples = []
    for i in range(12):
        day = date(2026, 9, 1) + timedelta(days=i)
        x = i - 5
        body = (i % 4) - 2
        inputs = inputs_for(day, night=x, body=body)
        examples.append(
            TrainingExample(inputs=inputs, actual=bar_for(inputs, 0.002 * x, 0.004 * body))
        )
    request = inputs_for(date(2026, 10, 1))
    return examples, request, SplitConfig(min_training_samples=6)
