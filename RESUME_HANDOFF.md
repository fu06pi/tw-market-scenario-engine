# 額度中斷與續接交接

## 2026-10-26 續接中的權威狀態（優先於下方舊快照）

- 最新本機提交 73696e1；遠端 feat/tej-ohlc-research 核對為 88ea85e0c5d2795e8bcb25992519970efcd47530。Shell DNS 不能連 GitHub；GitHub connector 可讀。尚未同步本輪修改。
- TX normalized 三折已完成：340db2b412224657c0e7，240/240 日，Close MAE 393.12、改善12.50%；固定校準後 O/L coverage 仍低於75%，完整 gate FAIL。
- wick 候選 d77dbc4223777b1f104e 被中止，只保存 fold-1 sample audit，沒有 checkpoint 或 fold 成績。當時「wick 不會影響 Close」的停止理由錯誤：官方 head 的參數跨 GBUD variates 共用；output 梯度為零不代表參數更新後預測不變。更正證據 research/audit_head_parameter_coupling.py 與 reports/quant/head-wick-parameter-coupling-2026-10-25.json；不能據此宣稱 wick 已成功或失敗。
- 帳本第12次嘗試保留 RESERVED_UNCLEAN_EXIT_CHARGES_FULL_ALLOWANCE：1476.9139 秒全額預留扣帳，累計21600/21600秒。這不是已實測使用滿六小時。不得刪帳、退回或新開同一搜尋帳本繞過停止規則。
- 本輪不執行大型模型訓練／重播；既有保存預測的因果校準 artifact、每日載入驗證及aggregate報告已完成。
- 未提交：CURRENT_GOAL.md（原有修改，不動、不提交）；校準/每日CLI/測試；共享參數更正審計；numeric_runtime.py 與測試（尚需接入，不能稱 replay 已修復）。
- 兩市場既有240日校準已重現並保存artifact；實際best-research bundle完成2026-10-12 late retrospective checkpoint推論、封存及readback，snapshot hash `62e90965ab347c55f1335d7f34ec688382126a4586bd5cf952b8c8cb9253db77`。不是genuine forward。
- 公開aggregate：reports/quant/output-head-delivery-2026-10-26.{json,md}；八個完整候選均FAIL，production default仍為baseline。
- 最新完整測試110 passed，Ruff/diff check及兩份研究報告重現比對通過；工程測試不代表模型達標。真實0.001點 replay仍FAIL；thread數差異只是未驗證假說。
- 下一步：提交本輪修改並安全同步；之後以固定版本執行下一個真正交易日08:40前forward shadow。正式排程維持 production_switch=false。

更新：2026-10-24。使用者指示：在可用訂閱額度內持續工作；額度用盡後由使用者在重置後說「繼續」，直接接手。不要繞過額度、不建立自動喚醒、不宣稱背景持續工作。

## 每次續接先做

1. 讀 AGENTS.md、本檔、CURRENT_GOAL.md、PROJECT_STATUS.md、CODEX_HANDOFF.md。
2. 核對 git status、最近提交、實驗帳本、日誌與 checkpoint。先確認原進程是否仍活著，不能因對話中斷就假設訓練停止，也不能重複啟動同一實驗。
3. 依下面「下一個可執行步驟」接續，保留原資料、設定、日期、seed、fold、checkpoint 和失敗紀錄。不重跑已完成候選。
4. 每個重要結果、fold完成、啟動長工作前、正常結束回合前，更新本檔。平台可能突然中斷，不能只等額度歸零才寫交接。
5. 若續跑失敗，記錄實際錯誤與可恢復位置；不得將部分完成寫成全部完成。

## 兩種限制分開處理

- **ChatGPT 訂閱額度**：由平台管理，五小時窗口不是五小時連續運算計時器。僅在工具可讀時回報實際用量與重置時間；未知就寫未知。此刻未查詢，不填猜測值。用盡後等待使用者重置後再說「繼續」。不保證能在平台切斷的最後一刻寫入文件。
- **模型研究運算預算**：現有設定總上限21600秒、單次7200秒，與ChatGPT額度無關。帳本已用19469.29697秒，剩2130.70303秒（35.51分鐘）。這是上次核對快照，續接必須重讀帳本。無訓練時不會隨牆鐘時間遞減。
- 訂閱重置不會重置研究帳本。研究額度不足某次完整訓練時，先做失效診斷、工程修復與重播驗證；不能以此停止全部工作。若證據支持新一輪訓練，先記錄新方案、測速與有界預算，再處理原實驗限制；不得悄悄清零或修改帳本。
- 不購買雲端算力、不降低記憶體安全門檻、不放寬驗收門檻。

## 目前實際狀態

- 專案：fu06pi/tw-market-scenario-engine，分支 feat/tej-ohlc-research。
- 本機位置：work/tw-market-scenario-engine（相對目前Codex工作區）。
- 最新研究成果提交：0618b12；訓練程式提交：07daa1f。本交接提交會在這兩者之後。
- 最近檢查沒有訓練進程；TX train_std 尚未啟動。這不是持續運行狀態保證，接手需再核對。
- CURRENT_GOAL.md 有原有未提交修改，保留；本次交接不覆蓋、不一併提交。
- 最新成果0618b12尚未同步GitHub。遠端先前透過Contents API更新，和本機歷史分岔；同步前檢查遠端內容，不強推、不覆蓋他人修改。

### 已執行成果

TAIEX train_std/context64 自訂凍結backbone、訓練output head完成3折、240/240日，耗時2026.76秒。保存3個真實checkpoint。

- O/H/L/C MAE：99.54 / 299.39 / 315.45 / 415.00點。
- 收盤相對固定基準改善9.09%，相對舊64日候選改善1.21%；方向161/240=67.08%，中性±5bp。
- 使用既定126日視窗、40日暖機因果校準後，O/H/L/C邊際覆蓋82.92% / 76.25% / 77.08% / 75.83%，全部240日保留，點預測不變。
- 完整開發gate仍FAIL：改善95%區塊bootstrap CI [-1.11%,15.20%]；第一折較基準惡化20.52%，後兩折改善15.53%、11.94%。
- 校準後close interval score從1895.39惡化為2067.57，但仍優於固定基準。不能只報coverage成功。
- TX既有64日候選close MAE392.75、改善12.59%，完整gate仍FAIL；不是TX正規化損失的新結果。
- 上次校準測試2 passed；先前完整測試101 passed。工程測試不等於模型達標。

### 權威證據位置

- 資料：data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce
- 新TAIEX實驗：reports/generated/quant/experiments/timesfm3-output-head-walk-forward/88b26bf3d30c9ded6d0d/（result.json、fold-*.json、sample audits、safetensors）。
- 校準實驗：reports/generated/quant/experiments/head-normalized-calibration-2026-10-24/
- 可公開彙總：reports/quant/output-head-normalized-taiex-2026-10-24.json
- 訓練設定：config/quant/timesfm_output_head_loss_normalized.json
- 固定校準：config/quant/head_interval_recalibration.json
- 累積帳本：reports/generated/quant/experiments/output-head-search-budget.json
- 每日重播：reports/generated/quant/experiments/head-daily-replay-v1/；真實模型0.001點精度gate尚FAIL，不得放寬容差假裝修好。
- 10/22失準：reports/quant/postclose-diagnosis-2026-10-22.md；原始數字不可回寫。
- 官方原始碼fork：https://github.com/fu06pi/timesfm/tree/research/tw-ohlc-source-review

## 2026-10-24 本次續接新增成果

- 已完成新舊TAIEX三折配對失效診斷，240日全保留：research/diagnose_head_folds.py、reports/quant/head-fold-diagnostic-2026-10-24.{json,md}。第一折gap MAE79.22優於基準112.56，body MAE186.67劣於基準168.59；18個實際日盤跌超過0.5%的日子close全部偏高。分類是事後診斷，禁止當作盤前可知特徵。三折range平均低估211.94/517.77/498.93點，正規化比舊模型更低估振幅。
- 已核對TAIEX、TX各240日訓練驗證sample與每日inference sample：history、past_features、forecast_features逐元素相等，reference與contract一致。證據reports/quant/head-inference-input-parity-2026-10-24.json。僅證明輸入一致，尚未修復原0.001點輸出重播FAIL。
- 全套101 tests passed；新增診斷腳本Ruff通過。沒有啟動新訓練，模型訓練帳本未重置。
- 下一步優先在原checkpoint上定位forward數值執行路徑差異，及以training-only標籤審計u/d近零變換、振幅損失；每日校準整合未完成。不要重做以上診斷。

## 2026-10-25 再續接：訓練快取完成

- 已實作凍結 TimesFM 3 backbone 表示快取：每個樣本只跑一次 backbone，後續 epoch 僅更新 output head。為避免 CPU float32 因 Linear 輸入矩陣形狀不同產生 ulp 差異，每次 head 呼叫重建原完整矩陣形狀；真實 pinned 權重輸出逐位元相同，最大差異 0。
- 真實單樣本量測：完整 forward 0.2703 秒、cache build 0.2678 秒、後續 cached head 平均 0.000660 秒（100次）；head 部分約409倍。peak RSS 2769.50 MiB。證據：reports/quant/frozen-backbone-cache-benchmark-2026-10-25.json。
- 102 tests passed，Ruff及diff check通過。這是效率修復，不是準確率提升。
- 下一步：先提交快取修復，再以相同 train_std/context64/folds/seed/loss/gate 執行尚未完成的 TX 三折；預期因快取可在剩餘2130.70秒研究帳本內完成，但仍由帳本fail-closed控制。完成後才處理 range loss 候選與 daily calibration。

## 下一個可執行步驟

1. **先診斷第一折**：讀新舊TAIEX64 fold-1.json與各自baseline_rows/candidate_rows，在完全相同日期比較close/body/gap/range誤差；檢查誤差是否集中於固定波動狀態或少數大跌日。全日期保留，輸出彙總、失敗原因及可檢驗假說；不要先為10/22單日調參。
2. 檢查每日推論與訓練重播的0.001點差異來源（數值精度、批次與執行路徑）；保留原FAIL證據，修復後用同checkpoint重驗。
3. 將已驗證的因果校準狀態接入每日推論，檢查截止時間、暖機、來源hash、不可回寫快照；校準不得使用當日實際值。
4. 根據第一步結果決定下一輪訓練只改哪一類問題；TX完整三折前次耗時2728.66秒，超過現有剩餘預算，不啟動明知無法完整驗證的搜尋、不縮fold冒充完整結果。
5. 保存報告、更新本檔及PROJECT_STATUS/CODEX_HANDOFF，檢查後提交並安全同步研究分支。若同步失敗記錄原因，不宣稱已push。

## 不可改變的研究邊界

使用08:30前可知資訊做日盤預測；中午執行是盤前流程模擬，不引入當日日盤資訊。舊final永久FAIL，新結果僅development，需另有未觸碰future holdout。baseline比較範圍目前zero/range/night_regression，不宣稱勝過全部ARIMA。正式排程與Sheet不切換，production_switch=false。模型優先，晨報整合仍是後續整體目標。不得提交金鑰、授權原始資料或完整模型權重。

## 每次更新的最小紀錄

- 更新時間、分支、最新本機與遠端提交。
- 已完成步驟與證據路徑；尚未完成及FAIL原因。
- 執行中工作：PID／session、啟動指令、日誌、輸出目錄、完成fold、是否可安全續跑（沒有則寫無）。
- 訂閱額度（如已查詢）與模型運算帳本分列。
- 下一個具體指令或要修的檔案、測試方式。
- 未提交檔案及不得覆蓋項目。

使用者下次只需說：「繼續，先讀 RESUME_HANDOFF.md，從下一個可執行步驟接手。」
