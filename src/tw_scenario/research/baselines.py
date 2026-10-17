from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .data import FIELDS, reconstruct, transform

BASELINES = ("zero_return", "historical_shape_20", "arima_1_0_0")


def forecast(bars, selected: list[int], name: str) -> np.ndarray:
    if name not in BASELINES:
        raise ValueError("unknown baseline")
    transformed = transform(bars, "gbud")
    predictions = []
    for i in selected:
        previous = bars[i].close
        if name == "zero_return":
            predictions.append([previous] * 4)
            continue
        # Only completed bars through i, never target bar i+1 or later.
        history = transformed[:i]
        if name == "historical_shape_20":
            values = np.median(history[-20:], axis=0)
        else:
            result = []
            for channel in range(4):
                y = history[-60:, channel].astype(np.float64)
                if len(y) < 20:
                    raise ValueError("insufficient ARIMA history")
                if np.std(y) < 1e-10:
                    result.append(float(y[-1]))
                    continue
                mean = float(y.mean())
                std = float(y.std())
                with warnings.catch_warnings(record=True) as caught:
                    fitted = ARIMA((y - mean) / std, order=(1, 0, 0), trend="c").fit(
                        method_kwargs={"maxiter": 100}
                    )
                if not fitted.mle_retvals.get("converged", False):
                    raise ValueError("ARIMA did not converge")
                if any("converg" in str(w.message).lower() for w in caught):
                    raise ValueError("ARIMA emitted convergence warning")
                result.append(float(fitted.forecast(1)[0]) * std + mean)
            values = np.array(result)
        predictions.append(reconstruct(values[None, :], np.array([previous]), "gbud")[0].tolist())
    out = np.asarray(predictions)
    if out.shape != (len(selected), len(FIELDS)) or not np.isfinite(out).all():
        raise ValueError("invalid baseline output")
    return out
