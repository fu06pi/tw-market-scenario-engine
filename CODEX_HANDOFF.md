# Codex 交接：FinMind／官方／TEJ → V1 量化 → V0 晨報

更新：2026-10-16。Repository：`fu06pi/tw-market-scenario-engine`。分支：`feat/tej-ohlc-research`，起點 commit `b2984f5`。

## 不可改變的研究邊界

1. FinMind 是主要 bounded incremental API；TWSE／TAIFEX 是權威 value reconciliation；TEJ 只保留帳戶實際可得的 2025 overlap。
2. 固定 split：train 2018-01-01 起、validation 2024-10-09 起、final test 2025-10-09 至 2026-10-08。
3. Baseline與TimesFM final皆已各開啟一次並永久關閉；總 gate為 `FAIL`。不得看 final 後換模型、重調參數或把同期間稱為新 holdout。
4. `production_switch=false`。正式 Sheet、歷史紀錄及排程保持不變，直到 staging readback與 3–5 日 shadow 通過。
5. 不得提交憑證、FinMind raw data、完整逐日 actual rows、TimesFM 權重或 cache。

## 已完成證據

- Certified dataset：`cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce`。
- DQ：`PASS_RETROSPECTIVE`；2018-01-01 至 2026-10-08 共 2,133 個交易日，selected series missing／duplicate／scale／calendar／manifest issues 均為 0。
- Reconciliation：6,396 paired keys OHLC mismatch 0；2,133 settlement pair mismatch 0。
- Baseline full artifact：`reports/quant/experiments/f82462a281bc8db89146/baseline_results.json`（本機、Git ignored）。
- Independent TimesFM validation：`reports/quant/experiments/0d411a35a522a3c37ccb/timesfm_validation.json`（本機、Git ignored）。
- Joint TimesFM validation：`reports/generated/quant/experiments/5e8f5532503734ccf89b/timesfm_joint_validation.json`（本機、Git ignored）。
- 可公開彙總：`reports/quant/research-summary-validation-lock-2026-10-14.json`。
- TimesFM code `e51928e27119cb17bebc005be2696b75e0a9e688`；weights `google/timesfm-3.0-pytorch@43046b85ec22d584a13f8098c2ed39c889e129c2`。
- CPU load peak RSS約 2.80 GiB；batch 1；獨立 validation 8/8 checkpoints完成；joint 244/244 日完成。
- Joint lock：TAIEX = GBUD joint multivariate，validation close MAE 169.19；TX = GBUD independent multivariate，156.73。
- validation 對相同基準 paired dates：TAIEX改善 3.69%（未達 5%），TX改善 6.34%。
- Staging Sheet retrospective E2E：2026-10-08兩市場列完成 A:F、O盤前寫入、G:N、P收盤回填與逐欄 readback；A:F、O未被改寫。摘要：`reports/quant/staging-sheet-validation-2026-10-15.json`。
- Custom output-head gradient smoke：單一 train-period樣本、output head only、1 SGD step；五項必要證據皆通過。摘要：`reports/quant/timesfm3-output-head-gradient-smoke-2026-10-15.json`。不是官方 fine-tuning或候選模型。
- Forward heartbeat：`ohlc-shadow` 目前為 `PAUSED`。它曾在 live daily orchestration 完成前過早啟用；正式 Sheet與正式排程未修改。
- 驗證：51 passed；Ruff及 `git diff --check`通過。

## Final 已關閉

- 完整本機 artifact：`reports/generated/quant/experiments/5e8f5532503734ccf89b/timesfm_final.json`。
- 公開 aggregate summary：`reports/quant/research-summary-final-2026-10-14.json`。
- 總 gate：`FAIL`。TAIEX close MAE 371.53，較 paired baseline惡化 4.64%；TX close MAE 377.46，只改善 0.45%。
- TAIEX close interval coverage 73.64%；TX各項 73.14%–74.38%，未達 75%下限。O/H/L degradation gate皆通過。
- Bootstrap 95% CI均包含零。242個 TimesFM final days兩市場皆成功；TAIEX baseline的3個失敗日仍使 paired coverage為98.76%。
- 不得重跑、調參或把 2025-10-09 至 2026-10-08稱為新 holdout。失敗時預設政策已啟動：每日輸出使用各市場鎖定最強 baseline。

## 接續順序

1. 現有 CLI可從 certified retrospective資料建立 locked-baseline body，但尚未完成 live daily orchestration；不得稱為每日引擎完成。
2. 建立單一 staging-only入口，串接08:30增量收集、live availability/cutoff認證、DQ、body、seal/freeze、deterministic V0晨報與Sheet write/readback；任何步驟失敗都要保存失敗紀錄並停止。
3. 建立收盤 realization/evaluation與G:N、P回填，先驗證A:F、O列指紋未改寫。
4. 手動完成至少一個 genuine 08:40前 live staging E2E，才可重新啟用 `ohlc-shadow`；再累積3–5個交易日 genuine forward shadow。
5. 自訂 output-head gradient smoke已完成且另名；不能改變已關閉 final的結論。
6. 只有端到端證據通過後，另行評估正式切換與 rollback；目前不修改正式排程。

可重播目標見 [CURRENT_GOAL.md](CURRENT_GOAL.md)，逐項狀態見 [PROJECT_STATUS.md](PROJECT_STATUS.md)。
