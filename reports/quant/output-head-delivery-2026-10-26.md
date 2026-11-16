# Output-head model delivery — 2026-10-26

All results are retrospective development evidence. The complete gate remains **FAIL** and production keeps the locked baselines.

| Market | Candidate | Context | Loss scale | Close MAE | Improvement | CI95 | Gate |
|---|---|---:|---|---:|---:|---|---|
| TAIEX | dd4ef236644360b9e734 | 64 | none | 420.06 | 7.98% | [-2.74%, 15.57%] | FAIL |
| TAIEX | 6197e7f4be27ac0c8f6e | 96 | none | 467.53 | -2.42% | [-21.26%, 9.48%] | FAIL |
| TAIEX | 7ed6c8055e40e5d8fb4e | 128 | none | 495.91 | -8.64% | [-30.79%, 5.74%] | FAIL |
| TAIEX | 88b26bf3d30c9ded6d0d | 64 | train_std | 415.00 | 9.09% | [-1.11%, 15.20%] | FAIL |
| TX | d635e40df84cdfad4cde | 64 | none | 392.75 | 12.59% | [1.47%, 21.79%] | FAIL |
| TX | 601d01aa4ed209962486 | 96 | none | 431.70 | 3.92% | [-12.90%, 16.45%] | FAIL |
| TX | e48eafb8d283c39acc1d | 128 | none | 407.74 | 9.25% | [-4.74%, 19.34%] | FAIL |
| TX | 340db2b412224657c0e7 | 64 | train_std | 393.12 | 12.50% | [0.97%, 22.42%] | FAIL |

## Causal interval calibration

- TAIEX: O/H/L/C coverage 82.92%/76.25%/77.08%/75.83%; 240 days retained; gate FAIL.
- TX: O/H/L/C coverage 74.58%/76.67%/72.92%/77.08%; 240 days retained; gate FAIL.

## Daily checkpoint delivery

The dual-market snapshot for 2026-10-12 was loaded, calibrated, sealed and read back with hash `62e90965ab347c55f1335d7f34ec688382126a4586bd5cf952b8c8cb9253db77`. It is a late retrospective run, not genuine forward evidence.

The wick candidate was interrupted before a complete fold and has no accuracy result. The earlier claim that its loss could not affect Close was rejected by the shared-parameter audit; this does not establish improvement.

Research ledger: 21600.00/21600 charged seconds. The last unclean reservation is not measured compute time.
