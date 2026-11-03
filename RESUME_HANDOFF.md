# 額度中斷與續接交接

## 2026-10-30 genuine forward 數值採樣（最新）

- 使用者最新指示覆蓋先前「模型完成前不做排程」的優先順序：從現在起以現行模型累積盤前 genuine forward OHLC 與盤後誤差；仍維持`production_switch=false`，不寫正式晨報或Google Sheet，也不把forward工程稱為模型達標。
- 新增`quant daily forward-preopen`／`forward-postclose`及`ops/run-forward-*`。每日固定公平comparator為`night_regression`，當日失敗才回退`range`；不使用所有gate FAIL的TimesFM／Phase-3候選。
- 盤前建立`forward-snapshots/YYYY-MM-DD.json`唯一不可回寫快照；盤後只追加realization／evaluation，並版本化重算雙市場OHLC MAE、bp、RMSE、方向混淆矩陣、區間coverage／width／score。沒有同一快照、晚到、休市或DQ失敗全部fail-closed並留attempt。
- 127 tests、Ruff、diff check及CLI help通過。證據為`reports/quant/daily-forward-engine-2026-10-30.json`，操作契約為`docs/DAILY_FORWARD_OPERATIONS.md`。
- 2026-10-30已錯過08:30，不回填；certified calendar確認下一交易日2026-11-02。
- 排程註冊尚未完成：目前computer-use沒有ChatGPT desktop app surface；`crontab -l`遭PAM拒絕，`systemctl --user`遭sandbox拒絕user bus。依OpenAI官方文件，本機專案Scheduled task需桌面App保持運行。入口已就緒，接手後若桌面surface可用，建立週一至週五08:20及18:10 Asia/Taipei兩個local-project task；不可宣稱目前已排程。
- `CURRENT_GOAL.md`仍是使用者既有修改，不覆蓋、不提交。Phase-3、Put/Call與daily forward已由本機commit `729e297`保存；shell DNS無法push後，GitHub connector以遠端`2e2a7c0`為parent及expected-SHA lease，非force同步等價commit `b8127c5e52c06da434c5fe7063d8190c25e9ac26`。PR #2 head、說明及mergeable clean已核對；工作樹只應保留`CURRENT_GOAL.md`。

## 2026-10-29 Phase-3 續接（最新）

- 使用者要求持續推進至模型目標。已核對本機 HEAD `83a684a`、工作樹僅有使用者既有的 `CURRENT_GOAL.md` 修改，沒有訓練進程；不得覆蓋或提交該檔。
- 最新權威結果仍是 residual-head run `73b3f1aadccb646372be`：公平 comparator 下 TAIEX/TX Close 改善 -2.78%/-0.35%，兩市場均 `FAIL`。舊 development 日期不得再用於無界 TimesFM shared-head 調參。
- FinMind 無 token 的實際探查顯示 `TaiwanOptionDaily`、選擇權／期貨法人、股票法人與匯率資料均回覆 free-level 拒絕；不能把未取得資料寫成已接入。TEJ 私密金鑰仍只存在 `.env`，未輸出值。
- 新 Phase-3 事前路線：先以既有 27 個已核實 08:30 前特徵，建立各市場獨立、價格／bp 空間的 causal residual Ridge。每個 outer fold 只用更早資料做 nested 選參數，保留固定 night-regression/range comparator、全部日期、同一因果區間校準與原 7.5% gate。此低參數模型若無穩定增益，不啟動 TimesFM 低秩 adapter；若有增益才把 correction route 接入凍結 TimesFM 表示。
- Phase-3 未開始產生成績；建立設定、測試與 checkpoint／daily inference 後才可報訓練完成。新 future holdout 起點維持 2026-10-09，舊資料只作 development。

## 2026-10-26 續接中的權威狀態（優先於下方舊快照）

- 2026-10-27 residual-head已完成：JOINT run `73b3f1aadccb646372be`有三個真實checkpoint、240個預定日期、實耗1356.85秒、peak RSS 2852.88 MiB。公平gate固定night regression後TAIEX Close改善-2.78%（362.88/353.06，三折+0.19%/-2.11%/-4.55%，CI95 [-6.34%,0.39%]），TX -0.35%（379.18/377.84，三折+5.24%/-3.01%/-0.22%，CI95 [-3.33%,2.23%]）；因果shrinkage仍為-1.39%/-0.27%。兩市場均FAIL。phase-2帳本4410.32/7200秒，剩2789.68秒依停止規則不再用於同一development調參。權威aggregate為`reports/quant/residual-head-result-2026-10-27.{json,md}`。118 tests、Ruff、diff check及aggregate byte-for-byte重現通過；本機提交`a441dd5`、`890a6ed`，GitHub等價commit `0ce472a652d11326b5e19433b954db45a17edca0`已用expected-SHA lease非force同步PR #2，PR為mergeable clean、沒有status checks。不得包含`CURRENT_GOAL.md`，不得再報原gate表面+19.68%/+14.26%為公平改善。

- 2026-10-27公平gate審計已完成探索重算：JOINT run `75d81bef117162e5a51a` 對固定night regression點預測baseline（TAIEX 2026-06-08 night regression產生非正價格時，以事前固定range回退）且候選與comparator同用60日／40樣本因果校準後，TAIEX close改善為-13.7113%，三折-12.9660%／-17.2069%／-10.8743%，CI95 [-21.6495%,-9.0859%]；TX為-5.9279%，三折-0.0426%／-4.9132%／-9.2019%，CI95 [-12.2426%,-0.0557%]。兩市場均FAIL。原本+10.79%／+9.97%來自區間穩定規則排除較強點baseline，不能作晉級證據。
- 公平比較已固化為`timesfm-head-fair-gate`：新增`head_fair_gate.py`、`head_fair_comparator.json`與測試。CLI已重建240日報告至`reports/generated/quant/model-search-v2/fair-gate-75d81bef117162e5a51a/`，結果與探索重算一致；公開aggregate為`reports/quant/output-head-fair-gate-2026-10-27.{json,md}`。
- wick、recent、JOINT三條模型改動都未對公平comparator產生真增益，已觸發原定連續三輪無增益停止規則。剩餘phase-2研究預算不再用於盲目調參；先完成公平gate、完整測試、結果報告與最佳baseline交付，不能放寬門檻製造PASS。
- 2026-10-27完整驗證為116 tests passed、Ruff pass、`git diff --check` pass。generated逐日結果受`.gitignore`保護；提交必須排除使用者既有`CURRENT_GOAL.md`修改。
- 本機提交`a1a8ff6`已完成；GitHub connector以遠端`322d8038`為parent、expected-SHA lease非force更新PR #2至`31c58e3a8cac2cf247db06614ae9645dbc1e5c31`，保留遠端獨有歷史。PR說明已更新公平gate結果；GitHub目前沒有回報status checks。除`CURRENT_GOAL.md`原有修改外工作樹應保持乾淨。

- 18:43後使用者糾正優先順序：模型尚未完成，不應先等待交易日。已停止推進forward／Sheet／晨報／排程，改回完整模型訓練與驗證。舊八個完整候選仍全為FAIL；checkpoint存在不代表模型完成。
- 已宣告獨立的model-search phase 2：`config/quant/timesfm_output_head_range_loss_v2.json`與`reports/quant/model-search-v2-plan-2026-10-26.md`。新帳本固定在`reports/generated/quant/model-search-v2/output-head-search-budget.json`，總額7200秒、每市場3600秒；舊21600秒帳本保持不動。只測wick loss單一改動，先TAIEX三折再TX三折，不並行。
- TAIEX phase-2 wick候選已完成三折，run `ab0491deabaa5272ecfd`，3個checkpoint、240/240日、實耗499.11秒、peak RSS 2852.71 MiB。Close MAE 414.22、改善9.26%，fold改善-20.48%／15.97%／11.86%，bootstrap CI95 [-0.82%,15.40%]；raw interval O/H/L/C coverage 83.33%／93.33%／60.83%／84.17%。完整gate仍FAIL（bootstrap下界、interval coverage），不能宣稱模型完成。
- TX phase-2 wick候選已完成三折，run `46be52591ffd4af5df82`，3個checkpoint、240/240日、實耗641.01秒。Close改善12.77%，fold改善4.68%／28.16%／-4.49%，bootstrap下界為正；完整gate仍因interval coverage FAIL。phase-2帳本目前已正式扣帳1142.62秒。
- 公平baseline審計發現各折事後選出的baseline模型不同。若先用被interval規則排除、但點預測較強的night regression做blend，再和較弱的fold baseline比較，可製造表面PASS；以相同因果選擇與校準公平比較後TAIEX僅+0.70%、TX-0.53%，均FAIL。不得使用原表面PASS。27特徵nested Ridge最佳TAIEX+4.98%、TX+3.78%，其他樹與close-only模型更差。
- 近期密集樣本TAIEX候選已完成，run `3977db39de71357a4e7b`、240/240日、實耗651.51秒。Close MAE411.11、對現行fold baseline改善9.94%，fold -20.83%／17.82%／11.65%，bootstrap CI95 [-0.32%,16.41%]，仍FAIL。與因果校準後night regression公平比較則惡化14.80%，不能利用原baseline interval排除規則宣稱接近達標。
- 22:34左右已啟動recent設定的JOINT八目標三折，session `37800`；這是獨立市場失敗後的既定joint路線。接手先poll／檢查新run，禁止重複啟動。
- 使用者最新指令是「以讓gate通過為最優先」。完成TX相同候選後，下一改動優先處理跨fold穩定性與因果baseline residual/blend；仍須只用預測日前資料，且不能以同一開發集調參結果冒充future holdout。
- 新增checkpoint專用`timesfm-head-live-preopen`單一執行入口；artifact／資源預檢、08:30前增量capture與官方核對、08:30後推論、08:40 deadline及write-once receipt已接線。112 tests、Ruff及diff check通過。2026-10-26真實bundle預檢為READY；下一verified exchange day為2026-10-27，必須08:20–08:29實際執行後才有genuine forward證據。命令與hash見`reports/quant/head-forward-preflight-2026-10-26.{json,md}`。
- checkpoint forward實作的本機提交為`d50f166`；GitHub connector在保留遠端獨有cloud evidence後建立等價提交`07ee36f2af9d7b03b2d5a70408e0374acb2f0d06`，PR #2 head已核對。同步使用expected SHA lease、非force更新；Shell DNS仍不能fetch，因此本機remote-tracking ref保持舊值，不能用`git status ahead`判斷GitHub是否已同步。
- TX normalized 三折已完成：340db2b412224657c0e7，240/240 日，Close MAE 393.12、改善12.50%；固定校準後 O/L coverage 仍低於75%，完整 gate FAIL。
- wick 候選 d77dbc4223777b1f104e 被中止，只保存 fold-1 sample audit，沒有 checkpoint 或 fold 成績。當時「wick 不會影響 Close」的停止理由錯誤：官方 head 的參數跨 GBUD variates 共用；output 梯度為零不代表參數更新後預測不變。更正證據 research/audit_head_parameter_coupling.py 與 reports/quant/head-wick-parameter-coupling-2026-10-25.json；不能據此宣稱 wick 已成功或失敗。
- 帳本第12次嘗試保留 RESERVED_UNCLEAN_EXIT_CHARGES_FULL_ALLOWANCE：1476.9139 秒全額預留扣帳，累計21600/21600秒。這不是已實測使用滿六小時。不得刪帳、退回或新開同一搜尋帳本繞過停止規則。
- 本輪不執行大型模型訓練／重播；既有保存預測的因果校準 artifact、每日載入驗證及aggregate報告已完成。
- 未提交僅有`CURRENT_GOAL.md`的使用者原有修改，持續保留、不提交。校準、每日checkpoint CLI、共享參數更正審計與numeric runtime接線均已提交；legacy checkpoint的0.001點replay仍FAIL，不能稱已修復。
- 兩市場既有240日校準已重現並保存artifact；實際best-research bundle完成2026-10-12 late retrospective checkpoint推論、封存及readback，snapshot hash `62e90965ab347c55f1335d7f34ec688382126a4586bd5cf952b8c8cb9253db77`。不是genuine forward。
- 公開aggregate：reports/quant/output-head-delivery-2026-10-26.{json,md}；八個完整候選均FAIL，production default仍為baseline。
- 最新完整測試112 passed，Ruff/diff check及兩份研究報告重現比對通過；工程測試不代表模型達標。真實0.001點 replay仍FAIL；thread數差異只是未驗證假說。
- 下一步：2026-10-27台北時間08:20–08:29執行預檢報告中的`timesfm-head-live-preopen`命令；成功也只記為固定FAIL候選的一次forward observation。正式排程維持 production_switch=false。

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
