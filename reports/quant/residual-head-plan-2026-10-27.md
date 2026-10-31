# Night-regression residual head plan — 2026-10-27

## Decision

The direct recent JOINT output head fails the fixed, equally calibrated
night-regression comparator: TAIEX Close MAE is 13.71% worse and TX is 5.93%
worse. A causal 60-session convex blend recovers only +0.72% for TAIEX and
-0.59% for TX; neither confidence interval excludes zero and neither reaches
the fixed 7.5% development buffer.

There is still a weak structural signal that differs from another absolute-head
hyperparameter search. The existing head correction has positive Close-residual
correlation in all three folds for both markets, and the 27-feature nested Ridge
diagnostic previously reached +4.98% for TAIEX and +3.78% for TX. This supports
one bounded residual-target experiment, but does not support a PASS claim.

## Frozen experiment

- Config: `config/quant/timesfm_output_head_residual.json`.
- Dataset and three 80-session development folds are unchanged.
- Instrument: one JOINT eight-target run; no independent or parameter sweep.
- Anchor: causal 252-session `night_regression`; fixed same-date `range`
  fallback only when the anchor cannot produce coherent positive OHLC.
- Target: actual GBUD minus anchor GBUD. Every historical context residual uses
  the anchor generated from information available before that historical day.
- Inputs: unchanged 64-session residual history and the same 54 pre-open causal
  JOINT features.
- Optimizer: unchanged recent 504-session, stride-1 training policy, learning
  rate, seed, early stopping and frozen TimesFM 3 backbone.
- Wick loss is disabled because a GBUD log-wick residual is not an absolute
  wick level; point, Close, quantile and crossing losses remain fixed.
- Fair evaluation: fixed night-regression comparator, range fallback, equal
  60-session/40-label causal interval calibration, all 240 dates retained.

This is custom output-head training over frozen TimesFM 3 representations. It
is not the official TimesFM fine-tuning API.

## Budget and stop rule

The existing phase-2 ledger remains authoritative: 3,052.35 of 7,200 seconds
are charged and 4,147.65 seconds remain before this run. The operation keeps the
existing 3,600-second per-run ceiling. It does not reset prior charges.

After this one structural experiment, stop model search if either market misses
the full fair development gate. Do not tune the same development dates again,
weaken the comparator, remove failed days, or open the old final as a new test.
Even a development PASS requires a newly frozen future holdout.
