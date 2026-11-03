# Codex 交接：FinMind／官方／TEJ → V1 量化 → V0 晨報

## 2026-10-27 residual-head 完成後接續點

唯一一條公平審計後的結構性路線已完成。JOINT run `73b3f1aadccb646372be`使用因果night-regression anchor residual、固定range fallback及凍結TimesFM 3 backbone，共完成三折checkpoint與240個預定日期。公平Close改善為TAIEX -2.78%、TX -0.35%；因果shrinkage仍為-1.39%/-0.27%，兩市場均`FAIL`。phase-2帳本累計4410.32/7200秒，剩2789.68秒；依停止規則不再用同一development日期調參。權威報告為`reports/quant/residual-head-result-2026-10-27.{json,md}`，正式預設維持night regression加range fallback，`production_switch=false`。

本機程式／結果提交為`a441dd5`與`890a6ed`；GitHub connector以遠端`58dfaf0`為parent及expected-SHA lease，非force同步等價commit `0ce472a652d11326b5e19433b954db45a17edca0`。PR #2說明已更新，GitHub回報mergeable clean且沒有status checks。`CURRENT_GOAL.md`的使用者修改持續排除。不得將原gate的+19.68%/+14.26%包裝為公平改善，也不得重開舊final。

## 2026-10-27 公平 gate 接續點

已完成Phase-2四次操作：TAIEX wick、TX wick、TAIEX recent及recent JOINT，共扣除3052.35／7200秒。原gate因區間穩定規則排除較強點baseline，JOINT表面改善不可採用。`timesfm-head-fair-gate`現固定night regression comparator、同日失敗才回退range，並對兩邊同樣作60日／40樣本因果校準。

JOINT run `75d81bef117162e5a51a`公平結果：TAIEX Close改善-13.71%，三折-12.97%／-17.21%／-10.87%，CI95 `[-21.65%,-9.09%]`；TX改善-5.93%，三折-0.04%／-4.91%／-9.20%，CI95 `[-12.24%,-0.06%]`。全部240日保留，TAIEX 2026-06-08用固定range fallback。兩市場均`FAIL`。

wick、recent、JOINT已連續三輪沒有對公平comparator產生增益，依原停止規則不再消耗剩餘4147.65秒作盲目調參。完整116項測試、Ruff及diff check已通過；本機提交`a1a8ff6`，PR #2遠端head為`31c58e3a8cac2cf247db06614ae9645dbc1e5c31`。接手先讀`reports/quant/output-head-fair-gate-2026-10-27.{json,md}`與`RESUME_HANDOFF.md`。正式預設仍是最強baseline，沒有新future holdout或production切換。

## 2026-10-26 最新接續點

量化模型的有界搜尋已停止：八個完整候選、24個fold checkpoint均已交付，但沒有候選通過雙市場完整gate。最佳研究候選為TAIEX `88b26bf3d30c9ded6d0d`與TX `d635e40df84cdfad4cde`；正式預設仍是各市場鎖定baseline。完整aggregate、checkpoint hashes、八項OHLC誤差、fold/CI/coverage/interval score在`reports/quant/output-head-delivery-2026-10-26.{json,md}`。

每日校準與推論已完成工程接線。私有可執行bundle為`reports/generated/quant/experiments/head-delivery-2026-10-26/bundle.json`；2026-10-12 late retrospective snapshot已成功readback，hash `62e90965ab347c55f1335d7f34ec688382126a4586bd5cf952b8c8cb9253db77`。不得稱作genuine forward或08:40前成功。舊checkpoint replay仍因sub-millipoint差異`FAIL`；thread count僅是假說。未來訓練已記錄numeric runtime，legacy模型只能標WARN。

研究帳本為21600/21600秒；第12次未正常完成，依既定規則全額扣除剩餘1476.9139秒。不得退回、重置或另開相同搜尋帳本。wick候選沒有完成任何fold checkpoint；`reports/quant/head-wick-parameter-coupling-2026-10-25.json`只更正共享參數判斷，不是準確率成果。

下一步是在不開新模型搜尋的前提下完成安全同步、以固定版本執行下一個真正交易日的08:40前forward shadow；沒有新授權研究預算前不重跑wick候選。正式排程與Sheet保持不變，`production_switch=false`。`CURRENT_GOAL.md`原有未提交修改需保留、不納入提交。

已新增固定checkpoint專用的`timesfm-head-live-preopen`，把artifact／資源預檢、08:30前FinMind與官方增量擷取、08:30後推論及08:40 deadline綁在同一次手動執行。下一個verified exchange day是2026-10-27；需在台北時間08:20–08:29以`cce0208...` base dataset與私有`head-delivery-2026-10-26/bundle.json`執行。2026-10-26預檢為READY、112 tests通過，但尚未到執行窗，因此不能標示genuine forward成功。預檢證據見`reports/quant/head-forward-preflight-2026-10-26.{json,md}`。

## 2026-10-24 第一折失效與推論輸入核對

已完成240日新舊TAIEX配對診斷：第一折開盤誤差優於基準，日盤body誤差較差；三折都低估振幅，正規化加劇range低估。詳見 reports/quant/head-fold-diagnostic-2026-10-24.md。TAIEX/TX各240日每日輸入與驗證輸入逐元素一致，僅排除輸入建構差異，原數值replay仍FAIL。全套101測試通過。未啟動新訓練、未改正式排程；續接見RESUME_HANDOFF.md。

## 訂閱額度中斷續接 — 2026-10-24

每次開始或恢復工作先讀 [RESUME_HANDOFF.md](RESUME_HANDOFF.md)。依使用者指示，在可用額度內持續推進；每個重要步驟及啟動長工作前更新交接，不等額度耗盡才保存。訂閱重置後由使用者說「繼續」再接手，不建立自動喚醒。ChatGPT額度與模型運算預算分開記錄；中斷後先查進程、checkpoint及帳本，避免重跑、重置預算或虛報背景工作。

## 2026-10-24 正規化損失與固定因果校準完成

TAIEX train_std/context64 真實 output-head 訓練完成三折、240/240 日，耗時 2026.76 秒，三個 checkpoint 已核對。O/H/L/C MAE 為 99.54/299.39/315.45/415.00 點；收盤比固定基準改善 9.09%，比舊64日候選改善1.21%，方向正確161/240（67.08%，中性±5bp）。這是 development 回溯比較，非新 final。

沿用事前固定126日視窗、40日暖機的因果校準，全部240日保留、點預測不變。O/H/L/C 邊際80%區間覆蓋由83.33/93.33/61.67/84.17%變成82.92/76.25/77.08/75.83%，四項均落在75–85%；interval score皆優於固定基準。但close score由原模型1895.39惡化為2067.57，不能只報coverage改善。

完整gate仍FAIL：改善95% block-bootstrap CI為[-1.11%,15.20%]；第一折收盤較基準惡化20.52%，後兩折改善15.53%、11.94%。尚不能宣稱穩定泛化或正式採用。證據與checkpoint hashes見 reports/quant/output-head-normalized-taiex-2026-10-24.json。

累積模型預算已用19469.30/21600秒，剩2130.70秒（35.51分鐘）。TX train_std未啟動：既有TX64三折耗時2728.66秒，高於剩餘額度；不重置帳本、不縮減fold冒充完整結果。現在沒有訓練進程。下一步先診斷第一折失效來源、完成既有checkpoint每日重播精度及校準推論一致性，再訂有根據的後續訓練方案。正式預設保留基準，production_switch=false；CURRENT_GOAL.md既有修改未動。

重現本輪校準：
```bash
.venv/bin/tw-scenario quant timesfm-head-calibrate data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce reports/generated/quant/experiments/timesfm3-output-head-walk-forward/88b26bf3d30c9ded6d0d reports/generated/quant/experiments/head-normalized-calibration-2026-10-24-replay --config config/quant/timesfm_output_head_loss_normalized.json
```

## 2026-10-23 保持盤前模型與損失尺度修正

使用者確認 10/22 中午試算是模擬真實盤前流程；不使用當日日盤資訊符合任務，不能列為模型缺陷。本階段不轉做盤中更新模型，保留原數字快照與失敗結果。

新增僅用第一折 optimization training labels 的尺度審計，見 reports/quant/head-training-loss-scale-audit-2026-10-23.json。g/b 與 log_u/log_d 尺度差異很大；同為一個標準差誤差時，舊 point Huber 的延伸目標輸出梯度約為 g/b 的119–143倍。這只是 point term 診斷，不等於總梯度歸因或10/22失準的唯一原因。

新增 loss_normalization=train_std 候選，按各折 training labels 計算 target 與 close-return std，early-stop 與 validation 不參與尺度擬合；僅改 loss 單位，保留64日context、特徵、folds、seed、optimizer、epochs、解碼與gate。既有none版本保持可重現，尚未有新模型改善結論。

本輪沿用總計21600秒的累積帳本，啟動前已用17442.54秒，剩4157.46秒（69.29分鐘）。先跑TAIEX完整三折；完成後才依剩餘預算決定TX，不為第二市場重置預算、不購買算力。


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
