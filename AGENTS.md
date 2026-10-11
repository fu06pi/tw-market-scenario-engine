# Codex project instructions

## Current decision and scope — 2026-10-11
Work in fu06pi/tw-market-scenario-engine. Rebuild its existing V1 into a TEJ-backed TAIEX and contract-specific TX daytime OHLC forecasting engine, researching TimesFM 3.0 historical fine-tuning and zero-shot variants against naive, night-session regression and ARIMA/ARIMAX baselines. Frozen V1 forecasts feed the original V0 morning report and post-close review. “V01” is the user's name for that existing report flow; inspect the real interface before mapping it to a module.

The user confirms possessing TEJ API access. This supersedes “TEJ deferred” in older documents. Actual credentials, table permissions, coverage and quotas still require verification. Do not invent connections, downloads, training or accuracy.

Read README.md, CODEX_HANDOFF.md and PROJECT_STATUS.md first. The 2026-10-07 architecture remains design background; older isolation/scenario documents do not prohibit the now-authorized V1-to-V0 integration. Resolve conflicts using this current user decision. Preserve old records and live operation during development.

## Data and prediction contract
- Timezone Asia/Taipei; trading-day cutoff 08:30, numerical persistence before 08:40. Never reuse legacy 08:50 for a TX pre-open forecast. Late output is a failure/late sample, not an on-time prediction.
- Forecast same-day daytime O/H/L/C for official TAIEX and an explicitly selected TX contract. Separate day/night sessions, last trade/settlement, contract month and exchange trading date.
- Read TEJ_API_KEY from environment. Never print credentials or include them in Git, logs, snapshots or reports. When absent, continue independent engineering and explain safe environment setup; do not request a key in chat.
- Inspect available TEJ tables, fields, quotas, publication timing and history. Aim for 2018 onward, but use actual common coverage. Use verified official TWSE/TAIFEX or CSV where necessary; log differences.
- Preserve provenance, raw cache, fetched_at, available_at and availability_basis. Current download time is not historical availability evidence.
- Reject/report missing, duplicate, invalid-scale and incoherent OHLC data. Do not fill missing days with zero returns or fabricate holiday bars.
- Calculate returns within the same TX contract and use its own previous daytime close. Fix rollover rules before final testing; avoid future-adjusted continuous series leakage.
- Align completed night-session and latest completed overseas sessions to forecast day D using actual as-of availability. Preserve stale-data age and holidays. Known prediction-period features may include already completed night/US sessions only when available before cutoff.
- No D daytime prices, future actual releases or later revisions in pre-open features.
- Compare direct relative OHLC plus constraints with g=ln(O/P), b=ln(C/O), u=ln(H/max(O,C)), d=ln(min(O,C)/L). Validate reversible nonnegative transforms for u,d and positive coherent reconstruction.

## Model implementation
- Verify official TimesFM 3.0 code, weights, license, interfaces and training support before use. Pin package versions, code commit and weight revision. Do not substitute 2.5 tutorials or silently use another version.
- Establish zero-shot inference before fine-tuning. Fine-tuning requires actual parameter updates through verified forward/loss/gradient paths; tuning windows or residual calibration is not fine-tuning.
- If no official fine-tuning interface exists, validate any custom training path with a small gradient/update/checkpoint test. Report unavailable capabilities honestly.
- Compare naive/volatility, night regression, ARIMA/ARIMAX, TimesFM historical zero-shot, multivariate zero-shot and historical fine-tuning. First independent markets, then joint models. Match dates, targets, cutoff and information sets.
- Measure hardware memory/runtime and record an affordable experiment budget before expensive runs. Do not purchase cloud compute. Support resumable runs and checkpoints.
- Record data/config/code/model versions, seeds, checkpoints, duration and failures. Add model dependencies reproducibly to the existing pyproject workflow.

## Validation and acceptance
- Chronological train/validation/final-test separation; no random temporal split. With enough data, reserve roughly the latest year as final test. Fix split dates before selection.
- Fit transforms, features, hyperparameters, calibration and ensemble weights using allowed past data only. Walk-forward retraining uses a predeclared update rule.
- Iterate on training/validation until target or budget; never repeatedly tune on final test. Lock candidate and strongest per-market baseline on validation before final evaluation.
- Target: both markets' final-test close MAE at least 5% lower than their locked strongest baseline. Each O/H/L MAE must not deteriorate by more than 5% against the same baseline.
- Report eight individual point/bp MAEs, RMSE, gap/close-return/range errors, forecast coverage and all failures. Direction uses a predeclared neutral threshold, class counts, confusion matrix and trivial baseline.
- Target 80% marginal interval coverage 75–85%; also compare interval score and width. Marginal quantiles cannot be directly combined into OHLC joint quantiles. Validate reconstruction, dependence and calibration.
- Use time-block uncertainty estimates and annual/volatility/rollover/event breakdowns. Pretraining overlap makes historical results retrospective; add fixed-version genuine pre-open forward evidence.
- Early stop and retain validation-best checkpoint. Stop after three successive rounds without validation improvement or when budget is exhausted; report best candidate, failure to meet targets and bottlenecks. Keep the strongest baseline as default candidate when no model clears the gate.
- Accuracy is a research target, never a guaranteed outcome or a claim of trading profitability.

## Narrative, snapshots and migration
- Freeze numerical output first. Narrative cites forecast_id and exact OHLC/intervals/versions, distinguishing numerical inputs from news background.
- Phase-one news cannot change numbers or probabilities. Do not invent rise probabilities from quantiles.
- Daily OHLC cannot establish high/low order, intraday regime or breadth. Label any extra narrative paths as unvalidated scenarios; quantitative path studies require intraday data.
- Preserve the original morning-report format and Sheet meanings: pre-open hypothesis versus actual open/intraday/post-close. Avoid repeating a column label inside cells.
- Do not rewrite historical predictions/evaluations. Add versioned OHLC contracts and adapters while preserving legacy schemas/records; separate legacy and new-model metrics.
- V1-to-V0 integration is authorized. Implement and verify it end to end before switching live schedules or Sheet flows. Keep existing tasks running during development and retain rollback configuration; do not silently pause or redirect them.
- Verify numerical persistence, narrative agreement, Sheet readback, realized-data updates and failure reporting. A hash proves consistency, not publication time; retain trustworthy persistence evidence.
- No standalone-model-only completion: the full goal includes generation of the original morning report and post-close comparison.

## Engineering and delivery
Use the existing Python package and pyproject.toml. Run existing tests before code changes and meaningful checks for cutoff, night-session dates, same-contract returns, rollovers, OHLC coherence, chronological isolation, snapshot/narrative consistency and failure handling. For documentation-only changes, review links/contracts and avoid adding implementation-mirroring tests.

Keep credentials, full licensed datasets and large weights out of Git. Label fixtures synthetic. Respect verified code/weight/data licenses and keep the work academic; do not deploy a trading service automatically.

Deliver a reviewable PR, reproducible commands, real data-quality/model-comparison reports, checkpoints where feasible and clear status distinguishing implemented, executed, target met and pending. Preserve originals before architecture-document rewrites. Use clear Traditional Chinese in user-facing updates.
