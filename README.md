# 台股日盤 OHLC 量化預測與晨報

## 2026-10-27 residual-head 最終停止結論

公平審計後唯一新增的結構性候選已完成：run `73b3f1aadccb646372be`以固定night-regression作anchor、range作失敗回退，TimesFM 3凍結backbone的自訂共享output head只預測因果GBUD residual。JOINT三折、三個真實checkpoint與240個預定日期均已執行，耗時1356.85秒、peak RSS 2852.88 MiB。

對固定且同樣校準的公平comparator，TAIEX Close MAE為362.88對353.06點（改善-2.78%，三折+0.19%/-2.11%/-4.55%）；TX為379.18對377.84點（改善-0.35%，三折+5.24%/-3.01%/-0.22%）。因果shrinkage後仍為-1.39%/-0.27%。兩市場完整gate均`FAIL`，因此停止在同一development期間繼續調參；正式預設維持night regression加range fallback，`production_switch=false`。完整結果見[`reports/quant/residual-head-result-2026-10-27.md`](reports/quant/residual-head-result-2026-10-27.md)。

## 2026-10-27 公平 gate 更正

Phase-2已完成wick、recent及recent JOINT三條路線。原gate把原始區間寬度和點預測baseline選擇綁在一起，會排除Close較強的night regression，造成對較弱range baseline的表面改善。新增的`timesfm-head-fair-gate`固定night regression為點預測comparator、當日失敗才回退range，並對候選與comparator套用相同因果區間校準。

Recent JOINT候選在公平比較下，TAIEX Close MAE 407.22點對358.12點，改善-13.71%；TX 404.50點對381.87點，改善-5.93%。兩市場三折皆沒有正改善，完整gate均`FAIL`。wick、recent與JOINT連續三路線無真實增益，依事前停止規則停止盲目搜尋；正式預設保留最強baseline，`production_switch=false`。完整結果與重現命令見[`reports/quant/output-head-fair-gate-2026-10-27.md`](reports/quant/output-head-fair-gate-2026-10-27.md)。

## 2026-10-26 output-head 研究交付

TAIEX／TX 各完成四個三折 expanding walk-forward 候選，共24個真實 fold checkpoint；八個候選皆保留240/240日，完整開發 gate 全部 `FAIL`。目前最低 Close MAE 為 TAIEX train-std 64日模型415.00點（改善9.09%，CI95 `[-1.11%,15.20%]`）及 TX 原64日模型392.75點（改善12.59%，CI95 `[1.47%,21.79%]`）。TAIEX欠缺穩定改善證據；TX固定因果校準後 Open／Low coverage為74.58%／72.92%，未達75%下限。公開彙總見 [`reports/quant/output-head-delivery-2026-10-26.md`](reports/quant/output-head-delivery-2026-10-26.md)。

`timesfm-head-calibrate` 現在會輸出綁定 result、完整 fold hash及逐筆殘差的因果校準artifact；每日推論逐一核對來源殘差與certified公布時間，暖機不足即失敗。2026-10-12已完成一次雙市場真實checkpoint late retrospective推論、校準、封存及readback，hash為`62e90965ab347c55f1335d7f34ec688382126a4586bd5cf952b8c8cb9253db77`。這不是genuine forward證據；舊checkpoint的0.001點replay仍FAIL，未來訓練會保存CPU數值執行環境供fail-closed重播。

wick-loss候選在第一折完成前中止，沒有checkpoint或準確率結果。後續共享參數審計推翻「wick output梯度不直接作用於g/b，所以Close不可能改變」的判斷：官方head的參數跨GBUD variates共用；此更正只說候選仍需實測，不表示候選有效。累積研究帳本依既定unclean-exit規則已扣滿21600秒；最後1476.91秒是全額reservation，並非實測運算時間。正式輸出仍使用鎖定baseline，`production_switch=false`。

2026-10-17 失敗樣本處理修正：`timesfm-head-train` 與
`timesfm-head-walk-forward` 會在 optimizer 執行前保存不可變的 sample audit，
逐日記錄 training／early-stop 的成功與失敗原因，只使用具完整連續資訊集的樣本。
不補值、不跨缺口拼接、不移動 split 或縮短 context；驗證日期不刪除，失敗仍計入
coverage。training 少於60個可用樣本或 early-stop 集合無可用樣本時，先保存 audit 再停止。
續跑時 audit 必須完全一致；舊 cloud NO_GO 證據保留在
[`reports/quant/cloud-2026-10-17`](reports/quant/cloud-2026-10-17/README.md)。
這項工程修正本身不代表模型訓練完成或 development gate 通過。

Repository：https://github.com/fu06pi/tw-market-scenario-engine  
架構與實作狀態更新：2026-10-16。

本專題將既有 V1 改造成台灣加權指數（TAIEX）與指定月份台指期（TX）的日盤 OHLC 量化預測引擎。經 live 資料查核後，FinMind 成為主要增量 API，TWSE／TAIFEX 官方資料負責權威核對，TEJ 實際可得的 2025 重疊期保留作第三方交叉驗證。研究 TimesFM 3.0 零樣本推論，並與簡單基準、ARIMA／ARIMAX 比較；任何自訂梯度測試會和官方微調明確分開。盤前先保存數字，再由原本 V0 晨報流程加入截止前新聞、國際局勢及金融數據，生成文字假說；收盤後比對同一份盤前結果。

**目前已完成資料品質、基準回測、TimesFM validation lock、唯一一次 final test，以及一次 retrospective Sheet staging端到端驗證。** Certified FinMind／官方資料在 2018-01-01 至 2026-10-08 通過回溯 DQ。TimesFM final總驗收為 `FAIL`：TAIEX close MAE較 paired baseline惡化 4.64%，TX只改善 0.45%，區間 coverage亦未全部達到 75%下限；因此不再用這個 final調參，預設保留鎖定最強 baseline。2026-10-08 shadow已在獨立 staging Sheet驗證 A:F、O盤前寫入、G:N、P收盤回填與盤前欄位不變；仍須 3–5 個交易日 genuine forward shadow，正式晨報切換尚未進行，`production_switch=false`。

## 文件與版本角色

- [AGENTS.md](AGENTS.md)：工程與研究執行規則。
- [CODEX_HANDOFF.md](CODEX_HANDOFF.md)：下一階段實作順序與交接。
- [PROJECT_STATUS.md](PROJECT_STATUS.md)：已完成與待完成項目。
- [V1→V0／V01 遷移契約](docs/V1_V0_MIGRATION_CONTRACT.md)：已核對的外部任務、legacy 欄位處理、切換門檻與 rollback 邊界。
- [外部任務切換套件](docs/V0_EXTERNAL_TASK_CUTOVER_PACKET.md)：三個既有任務的待審查 prompt、依賴順序、失敗處理與 rollback；目前為 `NOT_AUTHORIZED`。
- [原 OHLC 架構方案](docs/architecture/TW_OHLC_QUANT_ARCHITECTURE.md)：2026-10-07 的設計背景；其中「TEJ 暫緩」、實作順序與未固定門檻，以本文件及 AGENTS.md 的新決策為準。
- 其他舊情境、V0 isolation、KPI 與研究文件保留作歷史參考；與新流程衝突時，以本次新決策為準。

V1 負責數值、訓練、回測與預測快照；V0 負責依快照產生晨報與文字復盤。使用者所稱 V01 視為原本晨報流程的稱呼，實作前需確認實際介面，不能假設 repo 已存在名為 V01 的模組。

## 新流程

```mermaid
flowchart TD
    A["FinMind 主來源"] --> B["資料品質、時段與截止檢查"]
    O["TWSE／TAIFEX 官方核對"] --> B
    T["TEJ 2025 重疊核對"] --> B
    B --> C["V1 基準與量化模型"]
    C --> D["凍結 OHLC 數字快照"]
    D --> E["V0 晨報文字假說"]
    N["截止前新聞與金融背景"] --> E
    D --> F["收盤驗證"]
    E --> F
    R["實際日盤 OHLC"] --> F
    F --> G["誤差與後續模型研究"]
```

模型改版只影響後續預測。新聞第一階段只補充解釋，不修改數字；歷史新聞具備可驗證時間戳後，才能另做數值特徵增益實驗。

## 預測任務

| 項目 | 固定要求 |
|---|---|
| 標的 | 官方 TAIEX；依事前固定換月規則選定的 TX 月份合約 |
| 目標 | 當日日盤開盤、最高、最低、收盤，共八個價位 |
| 時區 | Asia/Taipei |
| 資料截止 | 交易日 08:30 |
| 數字保存 | 08:40 前完成持久化；遲到另標失敗，不補成正常盤前樣本 |
| 預測長度 | 一個交易日 |
| TX 口徑 | 日盤與夜盤分開；最後成交價與結算價分開 |

加權指數不能以 ETF 替代；日 OHLC 無法判斷高低點先後、盤中趨勢型態或市場廣度。文字可以列待驗證情境，但不能宣稱它們是 OHLC 模型已預測的結果。

## FinMind、官方資料與 TEJ

1. FinMind 使用 `TaiwanStockPrice`／`TAIEX` 與 `TaiwanFuturesDaily`／`TX`。`position` 是日盤、`after_market` 是夜盤；保存明確 `contract_date`，不使用連續合約代替同合約參考價。
2. `FINMIND_TOKEN` 可選；`TEJ_API_KEY` 只供 TEJ overlap。兩種憑證都不得進入 Git、日誌、快照或報告。
3. FinMind 請求按年或更小區間增量下載，保留 raw response、request、API status、擷取時間與內容 hash；schema 或 status 異常立即失敗。
4. 以 TWSE／TAIFEX 官方資料逐日核對 OHLC、時段、最後成交價、結算價及 open interest。FinMind Free 資料須標示來源、僅作非商業研究且不得散布 raw data。
5. TEJ 已驗證可連線，但本帳戶的 TAIEX／TX 查詢只在 2025 有實際 rows，因此只作重疊核對，不補造 2018-2024 資料。
6. 候選歷史起點為 2018 年，終點為最新完整交易日；保存來源、交易日、實體時段、合約、擷取時間、歷史可用時間與其依據。今天下載舊資料的時間不能代替歷史可用時間。
7. 檢查缺漏、重複、休市、異常尺度及 OHLC 關係；缺漏不填成零報酬，休市不建立虛構 K 棒。官方／FinMind 夜盤的 `settlement=0` 表示不適用，不能當作正價格或整批資料錯誤。
8. TX 報酬先於同合約內計算，再依固定換月規則拼接；參考前收使用被選合約自己的前一日盤收盤。

已結束夜盤及最近已收盤美股，依台灣當天早上真正可取得的時間對齊到當日特徵，保留海外休市與資料距今時間。禁止用相同日曆日期直接合併跨市場日線，或輸入當日日盤、未公布實際值與事後修訂資訊。

## 目標表示與模型

比較兩種 OHLC 表示法：相對前收的 OHLC 變化加價格約束；以及以下四個目標。令 P 為同標的／同合約前一日盤收盤：

| 目標 | 定義 |
|---|---|
| g | ln(O/P) |
| b | ln(C/O) |
| u | ln(H/max(O,C)) |
| d | ln(min(O,C)/L) |

對 u、d 使用經驗證的可逆非負轉換，還原保持非負。還原為 O=P×exp(g)、C=O×exp(b)、H=max(O,C)×exp(u)、L=min(O,C)×exp(-d)。所有輸出須為正，且 H≥max(O,C)、L≤min(O,C)。

對照至少包含：零報酬與歷史振幅、夜盤簡單回歸、ARIMA／ARIMAX、TimesFM 3 零樣本歷史版、多變量版與歷史微調版；先測市場獨立，再測聯合預測。共享外部特徵的模型使用相同資訊集。

TimesFM 3 是本專題指定研究目標。已固定 code commit `e51928e27119cb17bebc005be2696b75e0a9e688` 與 weight revision `43046b85ec22d584a13f8098c2ed39c889e129c2`。官方 3.0 高階介面為 inference-only；完整 LoRA 範例仍屬 2.5。自訂 `custom_output_head_gradient_smoke` 已在單一 train-period樣本執行一個 SGD step，證實 loss有限、output-head梯度／參數／預測改變及 checkpoint重載等價；這不是官方 fine-tuning、不證明改善，也不是 production candidate。公開摘要見 [`reports/quant/timesfm3-output-head-gradient-smoke-2026-10-15.json`](reports/quant/timesfm3-output-head-gradient-smoke-2026-10-15.json)。自架權重屬非商業授權，本專案維持學術研究與 shadow forward test。

新版模型優先研究另提供 `timesfm-head-train` 與 `timesfm-head-walk-forward`。`--instrument` 可指定 `TAIEX`、`TX` 或 `JOINT`；`JOINT` 以同一個head聯合預測兩市場八個GBUD目標，並以各市場自己的前收還原。它們凍結完整 TimesFM 3 backbone，只允許 `output_head` 更新，使用 GBUD Huber／close-return／pinball／quantile-crossing loss、train-only feature scaling、early stopping、write-once fold結果與可續跑checkpoint。設定固定八個有編號候選，比較256／512 context、`1e-4`／`3e-4` learning rate及0／`1e-4` weight decay，不能在執行後匿名改參數。新增的27項盤前特徵包含夜盤結構、前日OI、基差、換月／到期、日曆及5／20／60日量價狀態；聯合模式保留兩市場各自的27項特徵，舊五特徵介面保持不變，以保存已關閉實驗的程式口徑。開發gate對TAIEX與TX分別選穩定baseline並分別要求close MAE至少改善7.5%、fold多數改善、block-bootstrap下界非負、O/H/L不劣化、區間及99%成功率同時通過；聯合候選只有兩市場都通過才通過，候選排序使用兩市場中較低的close改善率。所有2026-10-08以前資料只作開發，不能重標為新final。

歷史阻擋記錄（截至2026-10-16）：當時新訓練尚未產生checkpoint或成績。2026-10-20已完成64日候選兩市場各三折，保存6個checkpoint，但完整開發gate仍為FAIL；見[最新結果](reports/quant/output-head-context64-development-2026-10-20.json)。2026-10-16實際preflight為`BLOCKED_INSUFFICIENT_MEMORY`：可用RAM 2,762 MiB、free swap 0 MiB，低於鎖定的3,072 MiB門檻；套件、commit、權重cache與磁碟皆已就緒。證據見 [`reports/quant/timesfm3-output-head-preflight-2026-10-16.json`](reports/quant/timesfm3-output-head-preflight-2026-10-16.json)。不得降低門檻後把OOM風險包裝成有效實驗。

Checkpoint 每日推論入口為 `timesfm-head-daily`。它接受 standalone `timesfm-head-train` 的 `result.json` 及 head safetensors；三折結果本身不是可部署 checkpoint。先鎖定開發候選，再依固定訓練截止日產生 standalone head。每個預測日必須晚於 training 與 early-stop 最後標籤日。模型尚未達標時，此命令僅供顯式研究，原預設 baseline 不變。

建立私有 bundle JSON（路徑相對於 bundle 所在目錄；不可提交大型 checkpoint）：

```json
{
  "TAIEX": {"result": "taiex/result.json", "checkpoint": "taiex/output_head.safetensors"},
  "TX": {"result": "tx/result.json", "checkpoint": "tx/output_head.safetensors"}
}
```

聯合模型改用單一 `JOINT` entry。以已包含指定交易日曆、同合約前收及已結束夜盤的 immutable dataset 執行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra timesfm3 tw-scenario quant timesfm-head-daily \
  data/processed/<dataset-id> --bundle reports/generated/quant/head-bundle.json \
  --date YYYY-MM-DD --output reports/generated/quant/snapshots/YYYY-MM-DD-head.json
```

命令不接受回填生成時間；08:30 前拒絕，08:40 後如實標示 LATE。先核對 checkpoint hash、模型 revision、feature 順序、scaler及標籤截止日，再使用不含當日日盤答案的輸入推論。輸出沿用 `ohlc-forecast-v1`，包含雙市場 OHLC、指定 TX 合約、同合約前收、區間、來源與模型雜湊；檔案不可覆寫並執行 readback。區間目前是未校準的轉換分位數包絡，名義80%並不代表已達到實際覆蓋率，快照固定 WARN 並揭露限制；不會修改正式排程。

真正盤前 shadow 使用單一 `timesfm-head-live-preopen` 入口。它只允許交易日台北時間08:20–08:29啟動，先驗證bundle、checkpoint、校準artifact與本機資源，再擷取並核對FinMind／官方增量；08:30後載入固定head並在08:40前封存。成功receipt仍明確保留`development_gate=FAIL`、`research_only=true`與`production_switch=false`，不寫Google Sheet或正式晨報：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra timesfm3 tw-scenario quant \
  timesfm-head-live-preopen --date YYYY-MM-DD \
  --base-dataset-dir data/processed/<certified-dataset-id> \
  --bundle reports/generated/quant/experiments/<fixed-bundle>/bundle.json
```

錯誤時段、資料逾時、官方核對差異、artifact竄改、資源不足、模型錯誤及08:40逾時都會fail-closed並保存不可變失敗attempt。此入口用來取得固定版本的前向觀察，不能把一日準時執行改稱模型達標。

2026-10-18 工程驗證：獨立與聯合 synthetic 小模型各完成 optimizer → safetensors → 重載 → CLI 推論 → 凍結快照及 readback，並驗證竄改拒絕、不可覆寫、盤前不讀實際值。全套92項測試通過。這不是 pinned TimesFM 真實訓練、三折成績或 genuine forward 完成證據。


## 訓練、驗證與停止

- 時間順序切分訓練、驗證及最終測試；資料充足時保留最近約一年作最終測試，選模前固定日期。
- 逐日 walk-forward：每次只使用當時已知資料。轉換、標準化、選模與校準不得偷看測試集。
- 「訓練直到一定準確率」指在訓練與驗證資料上迭代；鎖定候選後才跑最終測試。測試失敗不能反覆調同一測試集再當獨立證據。
- 先量測硬體、記憶體與耗時，記錄運算預算；不自行購買雲端運算。
- 使用 early stopping，保存驗證最佳 checkpoint。連續三輪無驗證增益或用完預算，停止搜尋並交付結果與瓶頸。
- 保存設定、資料版本、種子、權重版本、checkpoint、耗時與失敗，支援續跑。
- 預訓練資料重疊無法排除時，歷史結果標為回溯比較，另以固定模型的盤前前向測試補強。

## 驗收門檻

以下是研究目標，不是已達成績效或保證：

1. 在驗證集選定每個標的最強基準，再固定進入最終測試。兩標的收盤 MAE 均降低至少 5%。
2. 任一 O/H/L 的 MAE 不得比相同基準惡化超過 5%；八項價位分開列出。
3. 報告點數／bp MAE、RMSE、跳空、收盤報酬與振幅誤差；方向用事前中性門檻、混淆矩陣及簡單方向基準評估。
4. 名義 80% 邊際區間目標覆蓋率 75%～85%，同時比較 interval score 與寬度。四個目標的分位數不能直接拼成 OHLC 聯合分位數。
5. 時間區塊抽樣估計改善不確定性，按年度、波動、換月與重大事件檢查。報告全部應預測日、失敗日及覆蓋率。
6. 未達標仍交付最佳候選與完整結果，以最佳基準作預設候選輸出。預測達標不等於交易獲利。

已關閉 final 的事後穩定性診斷見 [`reports/quant/final-stability-diagnostics-2026-10-16.json`](reports/quant/final-stability-diagnostics-2026-10-16.json)。TAIEX 候選在低波動組改善 5.39%，但在中／高波動組分別惡化 4.21%／6.30%；TX 在2025改善7.54%，2026則惡化0.54%，非換月日近乎沒有改善。這些分組是 final 關閉後的描述性分析，不能拿來重選模型。重大事件分組因缺少凍結的歷史時間戳事件日曆而標示未評估。逐項完成／失敗／阻擋矩陣見 [`reports/quant/acceptance-audit-2026-10-16.json`](reports/quant/acceptance-audit-2026-10-16.json)。

## 晨報與每日作業整合

數字快照保存 forecast_id、交易日、合約、時段、截止／發布／持久化時間、前收、OHLC 與區間、資料品質、資料／程式／模型／設定版本及 hash。

原本 V0 晨報讀取同一 forecast_id 的數字，保留開盤前假說、實際開盤、實際盤中、實際盤後的用途與格式。模型輸入與新聞背景分開描述，不編造概率或另一套點數。若表頭已是「實際盤中」，儲存格不重複加同名標籤。

已核對的外部 `V1 台股劇本` 仍採 legacy 08:50 cutoff、Bull／Base／Bear 機率與盤中 regime；`財經事件晨報`、`V1 台股收盤驗證` 也尚未改讀新 OHLC 契約。這些欄位不會被當成新 V1 輸出。程式現在直接拒絕 08:50 narrative cutoff 與額外 probability 欄位；完整遷移表見 [`docs/V1_V0_MIGRATION_CONTRACT.md`](docs/V1_V0_MIGRATION_CONTRACT.md)。

`quant daily sheet-postclose-dry-run` 只產生 G:N、P 更新內容，並保存 A:F、O 的列指紋。寫入端必須先比對盤前列，再只更新保留的收盤欄位；任何 forecast_id、snapshot hash、標的／合約或 realization/evaluation 關聯不符都會停止。2026-10-15 staging readback摘要見 [`reports/quant/staging-sheet-validation-2026-10-15.json`](reports/quant/staging-sheet-validation-2026-10-15.json)。

目前 repo 另提供真正的 staging-only live 入口：`quant daily live-preopen` 只允許在08:20–08:29開始，先建立不可變FinMind capture並以TAIFEX當日夜盤核對，08:30後才預測、凍結、生成晨報及寫入staging A:F、O；`quant daily live-postclose` 在18:00後重新抓取FinMind實際日盤並以TWSE／TAIFEX核對，append realization/evaluation後只更新G:N、P。兩者皆逐欄readback、保存receipt，且 [`config/quant/staging.json`](config/quant/staging.json) 會拒絕正式Sheet ID。若資料晚於截止、官方核對不一致、Google憑證不存在、列非空或讀回不同，流程fail-closed並保存失敗attempt。

既有排程、Google Sheet 與歷史快照在開發期間保持運作。新資料、模型、數字保存、晨報及收盤更新通過端到端驗證後，才切換資料流；保留恢復版本。舊紀錄不回寫成新模型績效，新舊成績分開統計。

Codex heartbeat `ohlc-shadow` 目前為 `PAUSED`。先前在只有 retrospective staging 證據、尚未完成 live daily orchestration 時就啟用 heartbeat，順序不正確，因此已停用。重新啟用前必須先由單一 staging-only 執行入口完成 08:30 增量資料與 availability/cutoff 認證、DQ、body、seal/freeze、V0 narrative、Sheet write/readback，以及收盤 realization/evaluation；並至少手動完成一次 genuine live 交易日端到端驗證。失敗日必須保留，不能事後補成 forward PASS。正式 V0 排程仍獨立運作。

## 現有程式與實作路線

`src/tw_scenario/` 原有情境資料結構、固定權重 baseline、快照及評分仍保留。`src/tw_scenario/quant/` 已提供 contracts、providers、collection、quality、forecast、evaluation、backtest、TimesFM validation/final 及 daily integration CLI。研究結果仍是回溯證據，不能當作交易獲利或正式服務。

第一次官方收集的 dataset `fa3b3c0e5ddecb386b51446f0758efdff715e5b09d66ddcbe3c384b85e5be19e` 因 `settlement=0` parser 問題只留下 2,210 rows，缺 4,226 個 expected rows，資料品質結果為 `FAIL`。該資料集保留作失敗證據，禁止進入績效比較。最新執行狀態見 [PROJECT_STATUS.md](PROJECT_STATUS.md)，可重播工作目標見 [CURRENT_GOAL.md](CURRENT_GOAL.md)。

安裝與驗證：

```bash
python -m venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra research
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra dev pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra dev ruff check .
python -m tw_scenario.cli evaluate examples/sample_prediction.json examples/sample_realization.json
```

Staging Sheet自主寫入使用Google Application Default Credentials，不把憑證放入repo或命令列：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra research --extra sheets
export GOOGLE_APPLICATION_CREDENTIALS=/secure/path/staging-writer-service-account.json
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra sheets tw-scenario quant daily sheet-preflight-staging
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra research --extra sheets tw-scenario quant daily live-preopen \
  --date YYYY-MM-DD \
  --base-dataset-dir data/processed/CERTIFIED_DATASET_ID \
  --baseline reports/quant/experiments/EXPERIMENT_ID/baseline_results.json \
  --final-summary reports/quant/research-summary-final-2026-10-14.json
```

服務帳號只需分享獨立staging Sheet；不得授權正式Sheet。目前本機ADC尚未設定，因此writer程式已驗證但尚未完成genuine live Sheet寫入。

本階段可公開的實作／未執行邊界見 [`reports/quant/daily-engine-implementation-2026-10-16.json`](reports/quant/daily-engine-implementation-2026-10-16.json)。

鎖定最強 baseline已可直接建立真實資料 snapshot body；2026-10-08 retrospective shadow在約83秒完成 body→freeze→V0 narrative→A:P Sheet dry-run，且保持同一 forecast_id/hash。獨立 staging Sheet亦已完成晨報與收盤復盤逐欄 readback，確認收盤更新沒有改寫盤前 A:F、O；這仍是回溯演練，不是 08:40 前向發布證據。單一 live pre-open/post-close orchestration、自主 staging writer、cutoff/DQ與官方核對已實作並以fake transport測試；尚未執行genuine forward日，且本機ADC未設定。完成手動live staging E2E後，才可重新啟用heartbeat並累積3–5日forward shadow。TimesFM只保留研究結果，不因final失敗反覆調整同一測試集。

## 本次文件備份

改造前 README.md、AGENTS.md、CODEX_HANDOFF.md、PROJECT_STATUS.md 已連同來源 commit 與 SHA-256 清單備份：
[Google Drive 原始文件備份](https://drive.google.com/file/d/1UPYML3IeAbG40iw5FREPCORo-RP8f-98/view?usp=drivesdk)。

舊文件備份仍有效。自備份後已新增研究原型與 raw cache，但尚未切換正式排程。


### Output-head 運算預算與續跑

新一輪研究先凍結總上限 `search_budget_seconds=21600`（6小時），每個 candidate／market／train或walk-forward操作累積最多7200秒。優先完成64日 TAIEX、TX，再考慮JOINT；96／128日僅在剩餘預算容許時執行。這是已宣告的新輪次上限，不表示已花費6小時，也不抹除舊實驗成本。

兩個訓練 CLI 共用 `--output-root/output-head-search-budget.json`。相同候選重啟沿用已花時間，程式版本變更不會替相同設定重置額度；所有設定與市場共用總額。不得刪除ledger或換output-root規避上限。記帳採elapsed wall time（舊設定欄位名`budget_cpu_seconds`保留相容），包含載入、樣本、訓練及評估；單一運算步驟可能超過截止，實際超時仍全額記錄。程序持有獨佔鎖，執行前先預留剩餘額度；正常退出按實際時間結算，非正常退出保守地把整筆預留額度計為已花費，不自動重設。資源preflight失敗不算啟動訓練。未完成epoch不算完整epoch，因預算中止的驗證日期仍留在validation audit。此機制不建立任何排程或購買算力。
