"""Night covariates: observed completed session only, explicit missing masks."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .data import reconstruct, transform


def covariates(bars, selected, context, series):
    # Transformed target i corresponds to bar i+1. Last covariate is known night
    # of that target session, not the still-unknown daytime OHLC/volume.
    return np.array(
        [
            [[series.get(str(bars[j + 1].trading_date), np.nan) for j in range(i - context, i + 1)]]
            for i in selected
        ],
        dtype=np.float32,
    )


def regression(bars, training, selected, series):
    y = transform(bars, "gbud")
    x = np.array([series.get(str(bars[i + 1].trading_date), np.nan) for i in training])
    finite = np.isfinite(x)
    if finite.sum() < 8:
        raise ValueError("insufficient observed training night returns")
    mean = float(x[finite].mean())
    std = max(float(x[finite].std()), 1e-6)

    def features(values):
        values = np.asarray(values)
        return np.stack(
            [
                (np.where(np.isfinite(values), values, mean) - mean) / std,
                (~np.isfinite(values)).astype(float),
            ],
            axis=1,
        )

    # Parameters/normalizers fit only on the predeclared train split.
    fit = Ridge(alpha=10.0).fit(features(x), y[training])
    future = [series.get(str(bars[i + 1].trading_date), np.nan) for i in selected]
    forecast = fit.predict(features(future))
    return reconstruct(forecast, np.array([bars[i].close for i in selected]), "gbud")
