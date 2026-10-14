# Codex 交接：FinMind／官方／TEJ → V1 量化 → V0 晨報

更新：2026-10-14。Repository：`fu06pi/tw-market-scenario-engine`。分支：`feat/tej-ohlc-research`，起點 commit `b2984f5`。

## 不可改變的研究邊界

1. FinMind 是主要 bounded incremental API；TWSE／TAIFEX 是權威 value reconciliation；TEJ 只保留帳戶實際可得的 2025 overlap。
2. 固定 split：train 2018-01-01 起、validation 2024-10-09 起、final test 2025-10-09 至 2026-10-08。
3. Baseline final 已開啟；TimesFM final 尚未開啟。不得看 final 後換模型、重調參數或把同期間稱為新 holdout。
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
- 測試：48 passed；Ruff及 diff check通過。

## 下一個不可逆研究步驟

使用目前 joint candidate lock執行唯一一次 TimesFM final：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra timesfm3 \
  tw-scenario quant timesfm-final \
  reports/generated/quant/experiments/5e8f5532503734ccf89b/timesfm_joint_validation.json \
  --baseline reports/quant/experiments/f82462a281bc8db89146/baseline_results.json \
  --dataset-dir data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce \
  --config config/quant/research.json \
  --cache-dir data/raw/timesfm3
```

必須同時檢查：兩市場 close MAE改善至少 5%；O/H/L MAE惡化不超過 5%；四項區間 coverage皆為 75%–85%；paired coverage至少 95%；block bootstrap及所有失敗日保留。任一市場失敗即總 gate `FAIL`，並以最強 baseline作預設，不能再調這個 final。

## Final 之後

1. 建立不含逐日授權行情的 final summary，保存完整本機 artifact hash。
2. 完成最佳可接受模型到真實每日 `OHLCForecastSnapshotBodyV1` 的 orchestration。
3. 以凍結 snapshot生成 deterministic V0 晨報；新聞只能提供 cutoff前背景，不能改數字。
4. 在 staging Sheet做 A:P header fingerprint、morning/post-close write/readback及 3–5 個交易日 shadow。
5. 只有端到端證據通過後，另行評估正式切換與 rollback；目前不修改排程。

可重播目標見 [CURRENT_GOAL.md](CURRENT_GOAL.md)，逐項狀態見 [PROJECT_STATUS.md](PROJECT_STATUS.md)。
