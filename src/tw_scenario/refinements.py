"""Fixed, JSON-exportable research candidates; no network or live-job mutation."""

from __future__ import annotations

import math

import numpy as np

from .split_forecast import Components, fit_ridge_head

VERSION = "night-refinements-v1"
BASELINES = ("zero_return", "expanding_mean", "previous_return", "rolling_mean_20", "night_carry")
CANDIDATES = ("night_residual", "body_control", "body_night", "gap_calibrated")
METHODS = BASELINES + CANDIDATES
MINIMUM = 60
ALPHA = 0.1


def validate_past(past, query_date=None):
    dates = [r["date"] for r in past]
    if len(past) < MINIMUM:
        raise ValueError("insufficient past samples")
    if dates != sorted(set(dates)) or (query_date and dates[-1] >= query_date):
        raise ValueError("training must have unique, chronological, strictly past dates")
    if len({r["market"] for r in past}) != 1:
        raise ValueError("markets must be fitted independently")
    for row in past:
        if row["feature_latest_date"] >= row["date"]:
            raise ValueError("lag feature date leakage")
        vector(row, "residual")


def vector(row, kind):
    features = row["features"]
    night = row["night_return_bps"]
    if len(features) != 4 or not all(math.isfinite(x) for x in [*features, night]):
        raise ValueError("need four finite lag features and a finite night return")
    if night <= -10000:
        raise ValueError("invalid night return")
    return {
        "gap": features[:2] + [night],
        "body_control": features[2:],
        "body_night": features[2:] + [night],
        "residual": features[:2],
    }[kind]


def apply_head(parameters, values):
    size = len(values)
    if any(len(parameters[k]) != size for k in ("mean", "scale", "coef")):
        raise ValueError("checkpoint feature dimensions disagree")
    numbers = [*values, parameters["intercept"]]
    numbers += [v for k in ("mean", "scale", "coef") for v in parameters[k]]
    if not all(math.isfinite(v) for v in numbers) or any(v <= 0 for v in parameters["scale"]):
        raise ValueError("invalid checkpoint parameters")
    return parameters["intercept"] + sum(
        (x - mu) / scale * coef
        for x, mu, scale, coef in zip(
            values, parameters["mean"], parameters["scale"], parameters["coef"]
        )
    )


def fit(past, method, market):
    validate_past(past)
    if method not in METHODS or past[0]["market"] != market:
        raise ValueError("unknown method or wrong market")
    targets = [r["target"] for r in past]
    artifact = {
        "version": VERSION,
        "method": method,
        "market": market,
        "training_first": past[0]["date"],
        "training_last": past[-1]["date"],
        "training_n": len(past),
        "mean": {k: float(np.mean([getattr(t, k) for t in targets])) for k in "gbud"},
        "heads": {},
    }
    if method in BASELINES:
        subset = targets[-20:] if method == "rolling_mean_20" else targets[-1:]
        artifact["recent"] = {k: float(np.mean([getattr(t, k) for t in subset])) for k in "gb"}
        return artifact

    def head(kind, y):
        x = [vector(row, kind) for row in past]
        return fit_ridge_head(x, y, x[-1], ALPHA)[1]

    if method == "night_residual":
        # Each historical baseline body uses only labels preceding that row.
        # Warm-up rows are excluded rather than assigning an invented prior mean.
        rows, residuals = [], []
        for i in range(20, len(past)):
            row = past[i]
            baseline_body = float(np.mean([t.b for t in targets[:i]]))
            residuals.append(
                row["target"].g + row["target"].b
                - math.log1p(row["night_return_bps"] / 10000) - baseline_body
            )
            rows.append(vector(row, "residual"))
        artifact["heads"]["residual"] = fit_ridge_head(rows, residuals, rows[-1], ALPHA)[1]
    else:
        artifact["heads"]["gap"] = head("gap", [t.g for t in targets])
        body_kind = "body_control" if method == "body_control" else "body_night"
        artifact["heads"][body_kind] = head(body_kind, [t.b for t in targets])
        if method == "gap_calibrated" and market == "TAIEX":
            # Prequential calibration: each fitting target is predicted by a head
            # fitted strictly before it. Never regress fitted in-sample outputs.
            predictions, actual = [], []
            for i in range(max(MINIMUM, len(past) - 60), len(past)):
                prediction, _ = fit_ridge_head(
                    [vector(r, "gap") for r in past[:i]],
                    [t.g for t in targets[:i]], vector(past[i], "gap"), ALPHA,
                )
                predictions.append([prediction])
                actual.append(targets[i].g)
            if len(actual) < 30:
                raise ValueError("gap calibration needs 30 prior out-of-fold predictions")
            artifact["heads"]["calibration"] = fit_ridge_head(
                predictions, actual, predictions[-1], ALPHA
            )[1]
            artifact["gap_calibration_n"] = len(actual)
    return artifact


def predict(artifact, row):
    if artifact["version"] != VERSION or artifact["method"] not in METHODS:
        raise ValueError("unsupported checkpoint")
    if artifact["market"] != row["market"] or artifact["training_last"] >= row["date"]:
        raise ValueError("wrong market or non-past checkpoint")
    if row["feature_latest_date"] >= row["date"]:
        raise ValueError("lag feature date leakage")
    vector(row, "residual")
    method, mean, heads = artifact["method"], artifact["mean"], artifact["heads"]
    night = math.log1p(row["night_return_bps"] / 10000)
    g, b = mean["g"], mean["b"]
    if method == "zero_return":
        g, b = 0.0, 0.0
    elif method == "previous_return":
        # Query lags belong to the CURRENT selected contract, including roll day.
        g = math.log1p(row["features"][0] / 10000)
        b = math.log1p(row["features"][1] / 10000) - g
    elif method == "rolling_mean_20":
        g, b = artifact["recent"]["g"], artifact["recent"]["b"]
    elif method in ("night_carry", "night_residual"):
        g = night
        if method == "night_residual":
            b += apply_head(heads["residual"], vector(row, "residual"))
    elif method in CANDIDATES:
        g = apply_head(heads["gap"], vector(row, "gap"))
        kind = "body_control" if method == "body_control" else "body_night"
        b = apply_head(heads[kind], vector(row, kind))
        if "calibration" in heads:
            calibrated = apply_head(heads["calibration"], [g])
            # Opening magnitude is calibrated independently of total close return.
            b += g - calibrated
            g = calibrated
    components = Components(g=g, b=b, u=mean["u"], d=mean["d"])
    return components, components.reconstruct(row["previous_close"])
