# TimesFM 3 output-head model search phase 2

Status at declaration: **NOT STARTED**. The existing eight completed candidates all fail the complete development gate. The interrupted wick-loss run has no checkpoint or fold result and therefore has no accuracy conclusion.

This phase responds to the user's explicit instruction to finish model training before any forward-day, report, Sheet, or scheduling work. It does not modify or refund the exhausted phase-1 ledger at `reports/generated/quant/experiments/output-head-search-budget.json`.

## Frozen experiment

- Dataset: `cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce`
- Config: `config/quant/timesfm_output_head_range_loss_v2.json`
- Output root: `reports/generated/quant/model-search-v2`
- Order: TAIEX full three-fold walk-forward, then TX full three-fold walk-forward; never in parallel.
- Single change: add training-only, train-standardized log wick-span SmoothL1 loss with `wick_weight=0.25` to the completed context-64 `train_std` candidate.
- Unchanged: development dates, 240 validation sessions, folds, features, seed, optimizer, learning rate, weight decay, epochs, patience, decoder, calibration, baselines, bootstrap and promotion gates.
- TimesFM code: `e51928e27119cb17bebc005be2696b75e0a9e688`
- Weights: `google/timesfm-3.0-pytorch@43046b85ec22d584a13f8098c2ed39c889e129c2`

## Budget and stop rule

- New phase total wall-clock allowance: 7,200 seconds.
- Per instrument operation allowance: 3,600 seconds.
- Stop if the phase budget is exhausted, a fail-closed resource check rejects the run, or the candidate completes for both independent markets.
- A partial fold is not a completed result. No final-test claim is allowed. If this candidate has no validation gain, the result remains FAIL and the next model change must be separately declared under the existing three-no-gain rule.

## Required comparison

For every market, retain all 240 intended validation dates and compare on identical dates against the same locked baseline and prior `train_std` candidate. Report OHLC MAE/RMSE/bp, close improvement, per-fold results, block-bootstrap confidence interval, gap/body/range errors, direction confusion matrix, marginal interval coverage/score/width, and all failure dates. A checkpoint is evidence of completed training, not evidence that the gate passed.

## Evidence-driven second candidate

The completed wick candidate did not fix TAIEX fold-1 instability. The next declared training candidate changes optimizer sampling only: use the latest 504 eligible sessions at stride 1, including the unchanged final 40-session early-stop suffix. The context, features, wick loss, seed, optimizer, decoder and gates remain unchanged. Its config is `config/quant/timesfm_output_head_recent.json`; it continues the same phase-2 ledger and starts with TAIEX only.
