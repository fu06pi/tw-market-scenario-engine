# Codex project instructions

## Current user decision
The user reversed the workflow on 2026-10-07 and requested a Codex web handoff on 2026-10-08.
Read CODEX_HANDOFF.md and docs/architecture/TW_OHLC_QUANT_ARCHITECTURE.md first. Older text-first scenario docs describe the legacy system.
V1 forecasts daily official TAIEX and contract-specific TX daytime OHLC numerically; the future V0 text layer reads frozen V1 output and adds news/world/financial context. In phase one, narrative context cannot change model numbers.
TEJ access is deferred. Use official TWSE/TAIFEX and CSV adapters first. Never invent data, credentials, downloads, or measured accuracy.

## Data and methodology
- Timezone Asia/Taipei. Proposed configurable cutoff 08:30 and persistence before 08:40 on trading day D; TX daytime opens 08:45. Do not reuse the legacy 08:50 cutoff for TX-open prediction.
- TAIEX is the official index, not an ETF. TX identifies contract month and session; C is last traded price, not settlement.
- Separate night/day sessions and exchange trading-date attribution. Compute returns relative to the same contract. Fix rollover rules before final tests.
- Compare naive, ARIMA and ARIMAX before selecting TimesFM 3; identical dates/information sets and walk-forward tests. No random temporal split.
- Fit transforms, hyperparameters and calibration on past data only. Preserve valid OHLC geometry. Marginal intervals are not joint intervals.
- Report all eight target errors separately, baseline improvement, missing data, failures and forecast coverage. Never silently exclude hard days.
- Daily OHLC cannot establish the order of high/low or intraday regime. Do not invent path narratives.
- Historical pretrained-model tests may overlap pretraining; distinguish retrospective comparisons from new forward evidence.
- Verify current official TimesFM documentation and weight licensing before use/deployment.
- Freeze numerical snapshots before rendering narratives; retain data/code/model/config versions and trustworthy publication-time evidence.

## Migration and engineering
During development do not alter live ChatGPT automations, V0 Google Sheet, or historical snapshots/evaluations. Implement migration support separately.
Keep credentials out of git; avoid committing full datasets/model weights. Label synthetic fixtures.
Use existing Python package/pyproject.toml. Run existing tests before changes, then meaningful tests for sessions/dates, same-contract returns/rollovers, cutoff leakage, OHLC coherence, walk-forward isolation and snapshot consistency.
Deliver reproducible commands, real data quality/test results and a reviewable PR. Describe blockers honestly; no accuracy claims without experiments.
Use clear Traditional Chinese in user-facing updates.
