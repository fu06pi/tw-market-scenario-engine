# Output-head 校準與每日重載驗證

全部為 development-only 學術研究；舊 final 不重新驗收，production_switch=false。

## 因果區間校準

固定設定 `config/quant/head_interval_recalibration.json`。保留全部日期，40 日暖機沿用原區間；只使用當天 08:30 已公布的較早殘差，跨折延續同一預先指定預測管線。輸出是邊際區間，不能解讀成 OHLC 聯合分位數。此操作不更新模型參數。

```bash
uv run --frozen tw-scenario quant timesfm-head-calibrate \
 data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce \
 reports/generated/quant/experiments/timesfm3-output-head-walk-forward/dd4ef236644360b9e734 \
 reports/generated/quant/experiments/head-calibration-v1/TAIEX
```

TX 改用實驗 `d635e40df84cdfad4cde` 及獨立輸出目錄。輸出不可覆寫。`result.json` 是可分享彙總；`predictions.json` 含授權資料，不得加入 Git。完整樣本 gate 為主，calibrated subset 僅為診斷。

## Fold checkpoint 每日推論

bundle 每市場須有 `result`、`checkpoint` 路徑，可加整數 `fold`。路徑相對 bundle；有 fold 時 result 必須是原始 walk-forward result，旁邊保留對應 fold-N.json。來源、訓練邊界及 head hash 都必須一致。只訓練至 fold train_end（含 early-stop）的 checkpoint 才能預測更晚日期。

本機已準備 `reports/generated/quant/experiments/head-daily-replay-v1/bundle.json`，指定 64 日兩市場第三折。以下重播以完整 pinned model 載入，應在訓練資源鎖釋出後執行。會占用同一 6 小時累積預算，不買算力、不重設預算。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --frozen tw-scenario quant timesfm-head-replay \
 data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce \
 reports/generated/quant/experiments/head-daily-replay-v1/bundle.json \
 2026-10-08 reports/generated/quant/experiments/head-daily-replay-v1
```

這只是已保存預測的重載一致性驗證，時間會誠實標示 LATE，不是 genuine forward 或新 final。快照預測仍為未校準原模型；校準研究失敗，尚未接為每日預設。正式晨報、Sheet 與排程不切換。
