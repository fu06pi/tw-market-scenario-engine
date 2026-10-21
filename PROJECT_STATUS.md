# Project Status

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


更新：2026-10-16。

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
- [x] 自訂 output-head gradient smoke：只解凍 `output_head.weight/bias`，單一 train-period樣本、單一步驟；梯度、參數差、預測差皆非零，checkpoint重載差為0。明確不是官方 TimesFM 3 fine-tuning或 production candidate。
- [x] TimesFM 獨立市場 8 個候選 validation；244/244 日全部成功，final 尚未開啟。
- [x] 三類 validation-only 改善：模型 blend、因果 residual bias、126/252/512 視窗；未達雙市場門檻後停止盲搜。
- [x] TAIEX+TX 八序列 joint GBUD validation；244/244 日成功。TAIEX 鎖定 joint 候選，TX 保留 independent 候選。
- [x] TimesFM 候選的一次性 final test；兩市場均 242/242 成功，但總驗收為 `FAIL`，禁止再用該期間調參。
- [x] CLI：config／inspect／collect／quality／reconcile／backtest／TimesFM validation/final／daily predict/freeze/review/narrative/morning及post-close Sheet dry-run。
- [x] Write-once 雙標的 snapshot、hash/tamper check、append-only realization/evaluation、新聞 cutoff、待驗證盤中情境及 Sheet header fingerprint dry-run。
- [x] 鎖定 baseline直接建立每日 body 的 CLI；2026-10-08真實資料 retrospective shadow在約83秒完成 body→freeze→narrative→A:P dry-run，同一 forecast_id/hash鏈驗證通過。
- [x] 獨立 staging Sheet第132–133列完成盤前 A:F、O與收盤 G:N、P寫入及逐欄 readback；收盤更新後盤前欄位完全不變。證據：`reports/quant/staging-sheet-validation-2026-10-15.json`。這是 retrospective shadow，不是 genuine forward。
- [x] 發現 heartbeat 在 live daily orchestration 完成前過早啟用；`ohlc-shadow` 已改為 `PAUSED`，正式 V0 排程未動。
- [x] Staging-only `live-preopen`：08:20–08:29 capture、FinMind／TAIFEX夜盤核對、08:30後forecast→freeze→narrative→A:F,O write/readback；失敗attempt不可變留痕。
- [x] Staging-only `live-postclose`：FinMind／TWSE／TAIFEX日盤核對、append realization/evaluation、G:N,P write/readback，並保護A:F,O指紋。
- [x] Repo內Google Sheets writer只接受設定的staging ID並明確拒絕production ID；使用ADC，憑證不進repo／參數／日誌。
- [x] 2026-10-16 connector唯讀preflight確認staging表頭fingerprint一致、最後已用列133、134–135為下一組空白列；未寫入任何儲存格。證據：`reports/quant/staging-sheet-preflight-2026-10-16.json`。
- [x] 已核對外部 legacy 任務介面並建立 `docs/V1_V0_MIGRATION_CONTRACT.md`；V0 payload程式層拒絕08:50 cutoff與額外 probability欄位，覆盤仍綁定同一forecast_id/hash。
- [x] 已依實際外部任務輸出建立`V0_EXTERNAL_TASK_CUTOVER_PACKET.md`，含三份待審查prompt、receipt依賴、fail-closed條件與rollback；狀態保持`NOT_AUTHORIZED`，未修改任何任務。
- [x] Staging唯讀preflight CLI可在live capture前驗證ADC、最小讀取權、A:P表頭與下一組空白列；回報只含Sheet ID hash且不寫入儲存格。
- [x] 已對關閉 final 做年度、固定實際振幅狀態及TX換月日的事後穩定性診斷；保留全部失敗日。重大事件因沒有凍結的point-in-time事件日曆而明確未評估。
- [x] 新增TimesFM 3 output-head-only訓練與expanding walk-forward CLI；支援TAIEX、TX及八目標JOINT head，強制凍結backbone、train-only scaling、early stopping、write-once fold checkpoint及7.5%開發晉級gate。JOINT會分市場選baseline及驗收，任一市場失敗即整體失敗。這是自訂研究訓練，不是官方fine-tuning。
- [x] 新增27項盤前因果特徵：夜盤結構、前日OI、基差、換月／到期、日曆及5／20／60日量價狀態；保留舊五特徵介面供封存實驗重現。
- [x] 基準選擇可套用失敗率、OHLC bp誤差及區間寬度穩定門檻；每日模型metadata已分開selection reason、runtime fallback chain與fallback used，移除把失敗TimesFM誤寫成runtime fallback的語意。
- [x] 76 個 pytest、Ruff及 `git diff --check` 通過；新增output-head凍結、聯合八目標loss／遮罩／對齊／分市場還原、時間fold、固定候選grid、三輪無增益停止、雙市場開發gate、基準穩定性及fallback語意測試。

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
- TimesFM final：TAIEX close MAE 371.53，較 paired baseline **惡化 4.64%**；TX close MAE 377.46，只改善 **0.45%**。兩者皆未達 5%。
- TAIEX close interval coverage 73.64%；TX四項 coverage 73.14%–74.38%，低於 75%下限。O/H/L degradation gate則全部通過。
- Block bootstrap 95% CI：TAIEX close改善 `[-10.94%, 0.78%]`；TX `[-2.70%, 3.77%]`，均包含零且不支持穩定改善。
- 事後穩定性：TAIEX低／中／高實際振幅組close改善分別為`+5.39%`／`-4.21%`／`-6.30%`；TX為`+2.15%`／`+1.94%`／`-1.05%`。TX換月日`+8.14%`，非換月日`-0.06%`；這不是預先selection gate。
- 完整結果摘要：`reports/quant/research-summary-final-2026-10-14.json`。該期間不得重新包裝成新的獨立 holdout。

## 尚未完成

- [ ] 在實際Google staging Sheet執行新writer；本機目前無Application Default Credentials，只有fake transport與既有connector retrospective readback證據。
- [ ] 至少一個 genuine 08:40 前手動 live staging E2E；目前只有2026-10-08 retrospective shadow。
- [ ] 3–5 個交易日 genuine forward shadow；單次 retrospective staging write/readback已完成。
- [ ] 經歷史時間戳核實的重大事件日曆與海外市場特徵；目前不把新聞加入數值訓練，未平倉量亦尚未進入模型特徵。
- [ ] 實際執行新版TimesFM獨立與JOINT output-head訓練及三段walk-forward。2026-10-16本機preflight為`BLOCKED_INSUFFICIENT_MEMORY`：可用RAM 2,762 MiB、free swap 0 MiB，低於3,072 MiB門檻；尚無新checkpoint或準確度結果。2026-10-17已建立ChatGPT雲端訓練工作，結果仍須回寫可重現artifact並通過同一gate才能採信。
- [ ] 將外部 `財經事件晨報`、`V1 台股劇本`、`V1 台股收盤驗證` 從 legacy 08:50／情境機率介面遷移到凍結 OHLC 契約；正式切換前不得修改現行任務。
- [ ] 正式排程切換與 rollback 演練。

## 運作與宣稱

`production_switch=false`。正式 Google Sheet、歷史預測／評估及排程未改。雙市場 final 驗收已確定為 `FAIL`，預設輸出政策保留各市場鎖定最強 baseline；這不代表交易獲利。完整逐日研究 artifacts 因 FinMind raw-data redistribution 限制留在本機 ignored path；repo 保存可重現程式、設定、hash與不含逐日價格的彙總證據。
