# Project Status

更新：2026-10-14。

## 已完成並有執行證據

- [x] 在既有 repo／V1 改造，保留原 V0 晨報、收盤檢討、Sheet A:P 介面及正式排程。
- [x] TEJ 安全連線與帳戶實測；TAIEX／TX 可用歷史主要限於 2025，保留為 overlap 證據。
- [x] FinMind bounded／immutable／可續跑 provider；支援無 token、schema/status/hash fail-closed。
- [x] TWSE／TAIFEX 官方下載、parser、夜盤歸屬、同合約前收、固定換月及 reconciliation。
- [x] Certified dataset `cce0208a...`：2018-01-01 至 2026-10-08，2,133 個交易日，6,396 個 selected-series expected observations，DQ 為 `PASS_RETROSPECTIVE`。
- [x] FinMind／官方 6,396 個 key OHLC 完全一致；2,133 個 settlement pair 完全一致。這是回溯 value agreement，不是歷史 point-in-time 證明。
- [x] 固定 split、08:30 cutoff、08:40 persistence、5-session roll、5bp neutral threshold、2-hour CPU budget及 coverage gate。
- [x] Baseline validation 選模與一次性 final test。TAIEX 鎖定 GBUD+ARIMAX；TX 鎖定 GBUD+night regression。
- [x] TimesFM 3 code／weight revision、非商業授權、CPU batch=1 load probe與真實資料 smoke。
- [x] TimesFM 獨立市場 8 個候選 validation；244/244 日全部成功，final 尚未開啟。
- [x] 三類 validation-only 改善：模型 blend、因果 residual bias、126/252/512 視窗；未達雙市場門檻後停止盲搜。
- [x] TAIEX+TX 八序列 joint GBUD validation；244/244 日成功。TAIEX 鎖定 joint 候選，TX 保留 independent 候選。
- [x] CLI：config／inspect／collect／quality／reconcile／backtest／TimesFM validation/final／daily predict/freeze/review/narrative/sheet dry-run。
- [x] Write-once 雙標的 snapshot、hash/tamper check、append-only realization/evaluation、新聞 cutoff、待驗證盤中情境及 Sheet header fingerprint dry-run。
- [x] 48 個 pytest、Ruff及 `git diff --check` 通過。

## 已鎖定的 validation 結果

| 標的 | 鎖定 baseline close MAE | 鎖定 TimesFM candidate | candidate close MAE | paired 改善 | validation gate |
|---|---:|---|---:|---:|---|
| TAIEX | 175.67 | GBUD joint multivariate | 169.19 | 3.69% | `FAIL`：未達 5% |
| TX | 167.34（相同 244 日口徑） | GBUD independent multivariate | 156.73 | 6.34% | `PASS` |

TAIEX joint 相較原 independent TimesFM 的 close MAE 由 170.43 改善至 169.19；TX joint 未優於 independent，因此未入選。所有四項 80% 邊際區間 validation coverage 皆在 75%–85%。歷史結果可能和 TimesFM 預訓練資料重疊，只能標示回溯比較。

## Baseline final 已開啟的事實

- TAIEX：239/242 成功，close MAE 355.61 點；三個失敗日完整保留（兩次 ARIMA nonconvergence、一次 low 還原失敗）。
- TX：242/242 成功，close MAE 379.18 點。
- 同一 final 期間 2025-10-09 至 2026-10-08 不得重新包裝成新的獨立 holdout。
- TimesFM candidate final 尚未開啟；下一步只能依目前 candidate lock 執行一次。

## 尚未完成

- [ ] 執行鎖定候選的一次性 TimesFM final test並記錄總 gate `PASS`／`FAIL`。
- [ ] 自訂 output-head gradient smoke；官方 TimesFM 3 高階 API仍為 inference-only，不能把 window/calibration 稱為微調。
- [ ] 由最佳可接受模型直接建立真實每日 snapshot body 的 orchestration。
- [ ] Staging Google Sheet morning/post-close write/readback及 3–5 個交易日 shadow。
- [ ] 正式排程切換與 rollback 演練。

## 運作與宣稱

`production_switch=false`。正式 Google Sheet、歷史預測／評估及排程未改。現階段沒有證據可宣稱雙市場 final 驗收達標或交易獲利。完整逐日研究 artifacts 因 FinMind raw-data redistribution 限制留在本機 ignored path；repo 保存可重現程式、設定、hash與不含逐日價格的彙總證據。
