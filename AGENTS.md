# Codex project instructions

## 訂閱額度中斷續接 — 2026-10-24

每次開始或恢復工作先讀 [RESUME_HANDOFF.md](RESUME_HANDOFF.md)。依使用者指示，在可用額度內持續推進；每個重要步驟及啟動長工作前更新交接，不等額度耗盡才保存。訂閱重置後由使用者說「繼續」再接手，不建立自動喚醒。ChatGPT額度與模型運算預算分開記錄；中斷後先查進程、checkpoint及帳本，避免重跑、重置預算或虛報背景工作。

## 2026-10-18 模型優先執行目標

以 [CURRENT_GOAL.md](CURRENT_GOAL.md) 為本階段可重播目標。先完成真實訓練 checkpoint、三折驗證與每日 OHLC 推論；模型交付前暫緩卡片、晨報排版、Sheet及排程。派送任務與 active 狀態不得視為訓練進度；需以日誌、參數更新、checkpoint與fold結果佐證。先獨立市場再比較聯合；已執行中的有效實驗先完成，不重啟。原 final 永久 FAIL，後續 V0 整合仍保留，但不得搶占本階段模型工作。

## Current decision and scope — 2026-10-13
Work in fu06pi/tw-market-scenario-engine. Rebuild its existing V1 into a TAIEX and contract-specific TX daytime OHLC forecasting engine, researching TimesFM 3.0 zero-shot variants and a separately labelled custom output-head-only training path against stable naive, night-session regression and ARIMA/ARIMAX baselines. The custom path freezes the TimesFM backbone and is never described as the official fine-tuning API. Frozen V1 forecasts feed the original V0 morning report and post-close review. “V01” is the user's name for that existing report flow; the actual interface is the external ChatGPT tasks `財經事件晨報` and `台股收盤預測檢討`, plus their existing Google Sheet.

The verified source order is now FinMind as the primary resumable API, official TWSE/TAIFEX downloads as authoritative reconciliation evidence, and TEJ as a 2025 overlap cross-check. The user authorized this change after TEJ trial coverage proved insufficient for the requested 2018+ study. Do not discard TEJ evidence or relabel FinMind/official data as TEJ.

Verified constraints as of 2026-10-13:
- `TEJ_API_KEY` works, but the account returns `TRAIL/TAPRCD` TAIEX and explicit-contract `TRAIL/TAFUTR` rows only for 2025. The observed account limits are 500 requests/day and 50,000 rows/day.
- FinMind live probes return `TaiwanStockPrice`/`TAIEX` and `TaiwanFuturesDaily`/`TX` in both 2018 and 2025. TX uses `position` for day and `after_market` for night and contains explicit contract month, OHLC, volume, settlement and open interest. A token is optional; never print or persist `FINMIND_TOKEN`.
- FinMind Free is non-commercial, requires attribution, forbids raw-data redistribution, and documents 300 requests/hour without a token or 600/hour with a registered token. The repository's Apache license covers its code, not broader data-service rights.
- TimesFM 3.0 code is pinned to `e51928e27119cb17bebc005be2696b75e0a9e688`; weights are pinned to `google/timesfm-3.0-pytorch@43046b85ec22d584a13f8098c2ed39c889e129c2`. Self-hosted weights use the TimesFM non-commercial license. Keep the project academic and `production_switch=false`.

Read README.md, CODEX_HANDOFF.md and PROJECT_STATUS.md first. The 2026-10-07 architecture remains design background; older isolation/scenario documents do not prohibit the now-authorized V1-to-V0 integration. Resolve conflicts using this current user decision. Preserve old records and live operation during development.

## Data and prediction contract
- Timezone Asia/Taipei; trading-day cutoff 08:30, numerical persistence before 08:40. Never reuse legacy 08:50 for a TX pre-open forecast. Late output is a failure/late sample, not an on-time prediction.
- Forecast same-day daytime O/H/L/C for official TAIEX and an explicitly selected TX contract. Separate day/night sessions, last trade/settlement, contract month and exchange trading date.
- Read `FINMIND_TOKEN` and `TEJ_API_KEY` from the environment or a private ignored `.env`. Never print credentials or include them in Git, logs, snapshots or reports. Both providers must support credential-free engineering and cached/offline verification.
- Use bounded, immutable, incrementally cached FinMind requests and record API status/schema. Use official TWSE/TAIFEX records to reconcile values and source definitions; use TEJ only over its observed authorized overlap. Aim for 2018 onward, but use actual common coverage and report source differences.
- Preserve provenance, raw cache, fetched_at, available_at and availability_basis. Current download time is not historical availability evidence.
- Reject/report missing, duplicate, invalid-scale and incoherent OHLC data. Do not fill missing days with zero returns or fabricate holiday bars.
- Calculate returns within the same TX contract and use its own previous daytime close. Fix rollover rules before final testing; avoid future-adjusted continuous series leakage.
- Align completed night-session and latest completed overseas sessions to forecast day D using actual as-of availability. Preserve stale-data age and holidays. Known prediction-period features may include already completed night/US sessions only when available before cutoff.
- No D daytime prices, future actual releases or later revisions in pre-open features.
- Compare direct relative OHLC plus constraints with g=ln(O/P), b=ln(C/O), u=ln(H/max(O,C)), d=ln(min(O,C)/L). Validate reversible nonnegative transforms for u,d and positive coherent reconstruction.

## Model implementation
- Verify official TimesFM 3.0 code, weights, license, interfaces and training support before use. Pin package versions, code commit and weight revision. Do not substitute 2.5 tutorials or silently use another version. The verified 3.0 high-level model is inference-only; a custom output-head gradient smoke is not an official fine-tuning API and must not be reported as one.
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
- The failed 2025-10-09 through 2026-10-08 final remains permanently closed. A redesign informed by it may treat all data through 2026-10-08 as development-only, but must use expanding folds, a buffered 7.5% close-MAE promotion gate and a new future holdout before any new acceptance claim.
- Model work has priority over cards, report styling and schedule changes. Do not resume shadow scheduling until the redesigned model is locked and a manual genuine live staging day passes.
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

The first official-data collection produced dataset `fa3b3c0e5ddecb386b51446f0758efdff715e5b09d66ddcbe3c384b85e5be19e`, but it is rejected for research: a parser treated official `settlement=0` as a positive price validation error, dropping TX for 2018-2025 and most of 2026. Its DQ result is `FAIL` with 4,226 missing expected rows. Do not use that dataset for model claims. Also fix the weekend night-session end time (Friday 15:00 to Saturday 05:00, attributed to Monday trading date), empty/observed-only calendar false passes, calibration-label availability, low-coverage model selection, and dirty/untracked code identity before running selection.

Keep credentials, full licensed datasets and large weights out of Git. Label fixtures synthetic. Respect verified code/weight/data licenses and keep the work academic; do not deploy a trading service automatically.

Deliver a reviewable PR, reproducible commands, real data-quality/model-comparison reports, checkpoints where feasible and clear status distinguishing implemented, executed, target met and pending. Preserve originals before architecture-document rewrites. Use clear Traditional Chinese in user-facing updates.
