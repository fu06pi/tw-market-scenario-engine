# Phase-3 causal price-residual Ridge plan — 2026-10-29

The frozen TimesFM shared output head has too many trainable parameters for roughly
464 samples per fold and failed the fixed fair comparator. Phase 3 therefore begins
with a low-parameter diagnostic that tests whether the verified pre-open information
contains a stable correction signal before another TimesFM training run.

The candidate is fitted separately for TAIEX and TX. Its anchor is the predeclared
causal `night_regression`, with `range` used only when that anchor fails on the same
date. The 27-feature input contract is unchanged and uses only information available
by 08:30. Targets are actual-minus-anchor O/H/L/C errors in basis points relative to
the same-contract previous close. Reconstruction clips corrections at a predeclared
500 bp and enforces positive coherent OHLC.

Three 80-session outer folds remain fixed. For each fold, alpha and rolling training
window are selected only on an 80-session inner period before the outer fold. An
eligible inner candidate must keep O/H/L MAE within 5% of the anchor; among eligible
candidates the lowest Close MAE wins. All outer dates and failures are retained.
Candidate and comparator receive the same 60-session, 40-observation causal interval
calibration and the existing 7.5% development gate.

This is a new development route, not a final test. Data through 2026-10-08 remain
development-only. A locked checkpoint may be evaluated on observations beginning
2026-10-09, but the future holdout cannot be used to reselect features, windows,
alpha, clipping, or calibration. A development failure remains a failure; it does
not justify another unbounded search on the same dates.

FinMind free-level probes on 2026-10-29 rejected option, institutional-investor and
exchange-rate datasets. Those features are not represented as available. Official
TAIFEX public features may be added in a separately frozen data contract after this
low-parameter information test.
