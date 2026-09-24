# Quantitative Research Design

## 1. Objective

Test whether the V1 scenario engine adds out-of-sample predictive information over simple baselines while remaining calibrated, auditable, and faithful to the project's original text-first hypothesis workflow.

## 2. Unit of analysis

One Taiwan trading day.

Each day has exactly one pre-open frozen prediction snapshot and one post-close market realization.

The prediction snapshot contains both:

1. **Narrative hypotheses** — written expectations and scenario scripts.
2. **Quantitative forecasts** — directions, regimes, probabilities, and magnitudes.

Both are frozen at the same cutoff.

## 3. Separate targets

Do not collapse all market behavior into one `up/down` label.

- Open direction
- Close direction
- Intraday regime
- Breadth direction
- Magnitude (bps)

The written narrative should explicitly describe the expected behavior for the same target horizons.

## 4. V0 as reference evidence

V0 remains a separate text-first research record and may be used as read-only reference material.

For reproducible experiments, V0 content must be captured through a frozen export with provenance. Direct V0 consultation is suitable for qualitative comparison and hypothesis design, but V0 post-close text cannot be backfilled into a same-day pre-open feature set.

## 5. Baselines

At minimum compare with:

1. Always-up prior
2. Previous-day continuation
3. TAIFEX night-session sign
4. Weighted overnight baseline: TAIFEX + SOX + TSMC ADR
5. Simple logistic model without LLM reasoning

A complex model is not useful unless it improves out-of-sample metrics relative to these baselines.

## 6. Evaluation metrics

### Classification

[
Accuracy = \frac{\sum_{t=1}^{T} I(\hat y_t = y_t)}{T}
]

### Magnitude MAE

[
MAE = \frac{1}{T}\sum_{t=1}^{T} |\hat r_t-r_t|
]

### Brier score

For scenarios (k \in \{Bull,Base,Bear\}):

[
BS = \frac{1}{TK}\sum_{t=1}^{T}\sum_{k=1}^{K}(p_{t,k}-o_{t,k})^2
]

### Baseline lift

[
Lift = Metric_{V1} - Metric_{Baseline}
]

For loss metrics such as MAE or Brier score, report improvement with the sign reversed or as percentage reduction.

## 7. Narrative evaluation

The narrative itself should not be scored by vague semantic similarity alone.

Instead, written hypotheses are mapped to explicit coded claims before the open, such as:

- expected gap sign;
- expected trend / gap-fade / range path;
- expected close sign;
- expected breadth condition;
- expected dominant driver;
- invalidation trigger.

The coded claims are then evaluated against realized data. Human-readable text remains the explanation layer; quantitative labels provide the scoring layer.

## 8. Validation protocol

Use walk-forward testing only.

Example:

```text
Train: days 1-60 -> test day 61
Train: days 1-61 -> test day 62
...
```

No random train/test shuffle for time-series forecasting.

## 9. Anti-leakage rules

- Feature timestamp must be <= pre-open cutoff.
- Same-day open/high/low/close cannot enter the morning feature set.
- News published after cutoff cannot be backfilled into the morning record.
- Prediction narrative and numeric targets cannot be edited after creation.
- Model, feature, prompt, and data versions must be persisted.
- Same-day V0 post-close review cannot be used to explain or retrain the same-day morning prediction.

## 10. Model-change gate

Do **not** change a rule because of one bad day.

A rule change should require one or more of:

- repeated error pattern;
- statistically meaningful degradation;
- clear data-quality defect;
- documented regime shift;
- improvement in walk-forward validation.

This protects the system from hindsight overfitting.
