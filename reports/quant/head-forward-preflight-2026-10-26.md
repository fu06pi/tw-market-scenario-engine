# TimesFM head forward preflight — 2026-10-26

Status: **READY_FOR_MANUAL_FORWARD_WINDOW**. This is a preflight result, not a genuine forward forecast.

- Next verified exchange date: 2026-10-27.
- Start window: 08:20–08:29 Asia/Taipei; cutoff 08:30; persistence deadline 08:40.
- Certified base dataset: `cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce`, through 2026-10-12.
- Fixed bundle SHA-256: `997a29d11dcb97d541e35bbc7f24161037bd74c08fd640bd56e05f4d18223889`.
- Both checkpoints and both causal calibration artifacts resolved and passed provenance checks.
- Runtime: CPU, `READY_TO_LOAD`, 5,195 MiB available RAM, 1,034 MiB free swap.
- Validation: 112 tests, Ruff and diff check passed.
- FinMind token is absent and optional; TEJ is available through the private env file but is not needed for this daily capture. No credential value was exposed.

Run once inside the start window:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra timesfm3 tw-scenario quant \
  timesfm-head-live-preopen --date 2026-10-27 \
  --base-dataset-dir data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce \
  --bundle reports/generated/quant/experiments/head-delivery-2026-10-26/bundle.json
```

The command preflights the fixed artifacts, captures and officially reconciles the input increment before 08:30, waits for the cutoff, runs both checkpoint heads, and freezes a write-once snapshot. It fails closed and preserves an attempt record on a wrong window, late or inconsistent data, artifact mismatch, resource failure, model error, or missed 08:40 deadline.

The selected candidates still have `development_gate=FAIL`. A successful run supplies one fixed-version forward observation only; it does not change the production baseline, Sheet, morning report, schedule, or `production_switch=false` policy.
