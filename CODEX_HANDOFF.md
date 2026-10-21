# Codex 交接：FinMind／官方／TEJ → V1 量化 → V0 晨報

## 2026-10-21 後續訓練與替代路線

64、96 日視窗已完成兩市場共 12 個真實 fold checkpoint。64 日收盤 MAE 改善為 TAIEX 7.98%、TX 12.59%；96 日分別為 -2.42%、3.92%，未改善最佳候選。這些是 development 結果，完整 gate 均 FAIL；比較集合限於 zero/range/night_regression，不包含本輪 ARIMA/ARIMAX。

已執行固定 126 日、至少 40 個已公布殘差的因果邊際區間校準，兩市場各保留全部 240 日（40 暖機、200 校準），點預測完全不變。全樣本 O/H/L/C 覆蓋：TAIEX 83.75%/77.08%/75.83%/73.75%；TX 74.58%/76.67%/72.92%/77.08%。完整 gate 仍 FAIL，TAIEX bootstrap 下界亦未過。部分區間 score 比原模型惡化，不能只報 H/L 改善。詳見 reports/quant/output-head-calibration-{taiex,tx}-2026-10-21.json 與 output-head-context-comparison-2026-10-21.json。

128 日候選 TAIEX 已啟動、TX 等待同一運算鎖；尚無完成結果，不能把啟動視為訓練完成。此前四次訓練實用 10,839.78 秒／總上限 21,600 秒；目前 reservation 不等於實際完成用量，權威紀錄為 reports/generated/quant/experiments/output-head-search-budget.json。

每日推論已增加原生 walk-forward fold bundle，驗證來源 identity、fold summary、時間邊界與 checkpoint hash；不把 fold 假裝成 standalone refit。新增 timesfm-head-replay 可用同一預算將真實 checkpoint 經每日快照路徑重算，再對保存 fold 預測以 0.001 點容差比對。真實大型模型 replay 尚待資源鎖釋出；合成測試不算真實執行證據。production_switch=false，正式排程未變。


## 2026-10-21 真實 output-head 訓練進度

64交易日候選已完成 TAIEX、TX 各三折 expanding validation，兩市場各240/240日成功，六個fold checkpoint均已保存並核對SHA256。這是自訂 frozen-backbone/output-head 訓練，非官方微調API；不是新的獨立final驗收。

| 市場 | 基準 close MAE | head close MAE | 改善 | 完整開發gate |
|---|---:|---:|---:|---|
| TAIEX | 456.47 | 420.06 | 7.98% | FAIL：區間覆蓋、bootstrap下界 |
| TX | 449.30 | 392.75 | 12.59% | FAIL：區間覆蓋 |

TAIEX改善95% block-bootstrap CI為[-2.74%,15.57%]，TX為[1.47%,21.79%]。High覆蓋兩市場均92.5%，Low為TAIEX66.25%、TX65%，不符合75%～85%。八項點預測誤差、方向、區間score/width、全部fold訓練紀錄及版本見 [完整彙總](reports/quant/output-head-context64-development-2026-10-20.json)。此輪基準集合依凍結設定為zero/range/night_regression，不能宣稱已勝過所有ARIMA/ARIMAX。

已用累積運算5,141.95秒（85.70分鐘），6小時上限剩16,458.05秒。下一輪只改為預定candidate 1／96日context，沿用相同fold、特徵、optimizer及gate，先TAIEX再TX。沒有模型通過完整gate，正式預設仍是鎖定基準，production_switch=false。單機checkpoint真實重載／每日推論驗證與區間校準仍待完成；不以合成測試代替真實模型交付。


## 2026-10-18 模型優先執行目標

以 [CURRENT_GOAL.md](CURRENT_GOAL.md) 為本階段可重播目標。先完成真實訓練 checkpoint、三折驗證與每日 OHLC 推論；模型交付前暫緩卡片、晨報排版、Sheet及排程。派送任務與 active 狀態不得視為訓練進度；需以日誌、參數更新、checkpoint與fold結果佐證。先獨立市場再比較聯合；已執行中的有效實驗先完成，不重啟。原 final 永久 FAIL，後續 V0 整合仍保留，但不得搶占本階段模型工作。

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
- Live CLI：`quant daily live-preopen`與`live-postclose`已串接cutoff capture、官方核對、快照、晨報、append-only覆盤及staging write/readback；`config/quant/staging.json`鎖定staging並拒絕production ID。
- Legacy task audit：`財經事件晨報`、`V1 台股劇本`、`V1 台股收盤驗證`仍採舊介面；遷移映射與切換門檻已寫入`docs/V1_V0_MIGRATION_CONTRACT.md`。payload已硬性拒絕08:50 cutoff與額外機率欄位。
- External task cutover packet：`docs/V0_EXTERNAL_TASK_CUTOVER_PACKET.md`保存三份待審查prompt、receipt依賴、失敗處理與rollback；狀態`NOT_AUTHORIZED`，不得在forward gates前套用。
- Final stability：`reports/quant/final-stability-diagnostics-2026-10-16.json`只作關閉final的事後描述。TAIEX在中／高波動組惡化；TX非換月日近零改善。重大事件因缺少凍結事件日曆未評估。
- 驗證：62 passed；Ruff及 `git diff --check`通過，包含盤前／盤後完整orchestration、legacy contract拒絕、staging唯讀preflight與穩定性分組測試。`sheet-preflight-staging`可在live capture前驗證ADC、表頭與空白列；Google ADC在本機尚未設定，因此新writer尚無genuine live執行證據。

## Final 已關閉

- 完整本機 artifact：`reports/generated/quant/experiments/5e8f5532503734ccf89b/timesfm_final.json`。
- 公開 aggregate summary：`reports/quant/research-summary-final-2026-10-14.json`。
- 總 gate：`FAIL`。TAIEX close MAE 371.53，較 paired baseline惡化 4.64%；TX close MAE 377.46，只改善 0.45%。
- TAIEX close interval coverage 73.64%；TX各項 73.14%–74.38%，未達 75%下限。O/H/L degradation gate皆通過。
- Bootstrap 95% CI均包含零。242個 TimesFM final days兩市場皆成功；TAIEX baseline的3個失敗日仍使 paired coverage為98.76%。
- 不得重跑、調參或把 2025-10-09 至 2026-10-08稱為新 holdout。失敗時預設政策已啟動：每日輸出使用各市場鎖定最強 baseline。

## 接續順序

1. 優先執行新版TimesFM output-head-only研究。TAIEX、TX及JOINT八目標程式、三段expanding walk-forward與7.5%雙市場開發gate已完成；本機preflight因可用RAM低於3,072 MiB且free swap為0而阻擋，尚無新checkpoint或成績。2026-10-17已建立ChatGPT雲端訓練工作，雲端結果仍須回寫repo可重現artifact並通過相同gate。
2. 依CURRENT_GOAL先完成64日獨立市場三折實驗，再比較JOINT；已有有效實驗執行中先完成保存。所有2026-10-08以前結果只作development，不得改稱新final。未通過雙市場gate時保留baseline預設並停止晨報呈現工作。
3. 模型鎖定後，才在安全位置設定具有staging Sheet最小權限的Google ADC；不得把JSON內容貼入對話或提交Git，且不得分享正式Sheet給該服務帳號。
4. 交易日08:20–08:29手動執行`live-preopen`，驗證08:40前snapshot與staging readback；18:00後執行`live-postclose`並驗證同一forecast_id及A:F,O指紋。
5. 手動完成至少一個genuine live staging E2E，才可重新啟用`ohlc-shadow`；再累積3–5個交易日genuine forward shadow。
6. Shadow證據通過後，依遷移契約逐一改造三個外部任務並做rollback演練；之前不得動正式prompt、Sheet或排程。
7. 只有模型與端到端證據均通過後，另行評估正式切換與 rollback；目前不修改正式排程。

可重播目標見 [CURRENT_GOAL.md](CURRENT_GOAL.md)，逐項狀態見 [PROJECT_STATUS.md](PROJECT_STATUS.md)。
