# KPI Framework

## Primary KPIs

| KPI | Target | Why it matters |
|---|---|---|
| Open Direction Accuracy | open direction | tests overnight-information usefulness |
| Close Direction Accuracy | close direction | tests full-day predictive value |
| Brier Score | Bull/Base/Bear probability | tests probability quality, not just hit rate |
| Calibration Error | scenario probabilities | detects false confidence |
| Baseline Lift | all primary targets | prevents complex-model theater |

## Driver metrics

- Data freshness pass rate
- Snapshot integrity pass rate
- Coverage of required features
- Error type frequency
- Heavyweight concentration error
- Breadth prediction accuracy

## Guardrails

- Zero post-cutoff feature leakage
- Zero mutation of frozen prediction snapshots
- Every published performance number must specify sample period and baseline
