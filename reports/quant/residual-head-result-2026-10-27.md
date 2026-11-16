# Night-regression residual head result — 2026-10-27

The fixed JOINT residual-head experiment completed three folds and 240 intended dates, but the fair development gate is **FAIL** for both markets.

| Market | Candidate Close MAE | Fair comparator MAE | Improvement | Folds | CI95 | Gate |
|---|---:|---:|---:|---|---|---|
| TAIEX | 362.88 | 353.06 | -2.78% | 0.19% / -2.11% / -4.55% | [-6.34%, 0.39%] | `FAIL` |
| TX | 379.18 | 377.84 | -0.35% | 5.24% / -3.01% / -0.22% | [-3.33%, 2.23%] | `FAIL` |

## Causal shrinkage diagnostic

- **TAIEX**: Close improvement -1.39%; folds -0.19% / -2.13% / -1.21%; CI95 [-2.77%, 0.01%]; gate `FAIL`.
- **TX**: Close improvement -0.27%; folds 1.92% / -1.12% / -0.36%; CI95 [-1.55%, 1.08%]; gate `FAIL`.

The shrinkage calculation uses only labels published before each 08:30 cutoff. For the single failed JOINT date it optimistically emits the same-date comparator; the frozen failure row is unchanged. This still does not improve either market.

## Training and stop decision

The run used 1356.85 seconds and peaked at 2852.88 MiB. Three checkpoint hashes are recorded in the JSON aggregate. The phase-2 ledger has 2789.68 seconds left, but direct, blended and residual-head routes all fail the fixed comparator. The predeclared no-gain rule therefore stops further tuning on these development dates.

The production default remains the fixed night-regression comparator with range fallback. `production_switch=false`; no final or profitability claim is made.

## Reproduce

```bash
.venv/bin/tw-scenario quant timesfm-head-walk-forward \
  data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce \
  --instrument JOINT --config config/quant/timesfm_output_head_residual.json \
  --output-root reports/generated/quant/model-search-v2 --cache-dir data/raw/timesfm3
```

The generated artifact is immutable and will be reused rather than retrained when the identity matches.
