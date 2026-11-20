# 影子收盤與晨報轉接驗證

2026-11-05，Asia/Taipei。此次補上自動收盤資料取得、跨日監控與晨報轉接；沒有重新訓練／選模，也沒有產生或補填任何盤前預測。**真實新模型前向樣本仍為 0。**

## 已實作

- `postclose.py`：14:00 後取得官方 TAIEX／指定月份 TX 日盤 OHLC。現貨日期必須完全一致；期貨使用官方 CSV 明確篩選日期、合約、一般時段及有效成交，使用最後成交價而非結算價。保存原始 bytes、實際取得時間、HTTP method／表單、來源及指紋；缺一個市場時保存部分結果與缺漏，不替代或補零。
- `finish_shadow_day.py`：從當日盤前 Actions run 下載不可覆寫 artifact，核對 workflow 身分、ZIP 指紋、內容及遠端保存截止，才取得同日實際行情並執行監控。缺少盤前 run 就保留缺漏，不能事後產生預測；重跑、缺 artifact、下載／來源／評分失敗都保留。
- 獨立收盤 workflow 設工作日 **15:30** 執行，保留 actual、來源 receipts、評分、覆盤 patch 與所有失敗。它與原晨報／收盤任務分開，沒有更改既有任務的時間或寫入目的地。
- 跨日診斷從最多 50 個既有收盤 run／最近 60 個日曆日累積，按 checkpoint／市場／方法隔離。相同盤前 run 被不同收盤任務重複存檔時，按 run ID 與原始快照／receipt bytes 去重；不同預測不靠選擇較好結果去重。actual 若出現不同修訂則停止累計；過期、缺漏及查詢截斷明列。保留來源 artifact，不遞迴複製歷史資料；不聲稱全歷史或正式準確率。
- `morning_adapter.py`：唯讀核對原本可見的晨報表頭後，建立 16 欄轉接。只引用凍結預設模型的 OHLC／區間／checkpoint hash／forecast_id；方向中性門檻採快照政策。區間不充當方向勝率，不推斷高低點先後、盤中型態或消息因果。
- 盤前 wrapper 在數字保存後輸出 `morning-export.json` 與真實時間 receipt，超過 08:40 則整次標失敗。收盤 patch 只允許 G:N 與 P，讀回核對 A:F、O 原封不動，以及全部九個覆盤欄位；拒絕敘事數字與快照不符、日期／合約不符或已被改過的晨間資料。

## 實際取得的來源證據

[來源與執行彙總](2026-11-05-shadow-close-sources.json) 保留擷取時間、指紋及真實失敗，不公開完整行情。

| 實測 | 結果 | 意義 |
|---|---|---|
| 11/5 現貨日盤 | 成功取得 | 日期、OHLC、取得時間已保留 |
| 11/5 TX202611 HTML | 只回夜盤，拒絕 | 即使要求一般時段，也不能忽略實際回傳時段 |
| 11/5 TX202611 官方 CSV | 只有夜盤，日盤仍缺漏 | 僅供本次擷取時點判斷；不代填或聲稱永遠不可用 |
| 11/4 TAIEX／TX202611 官方日盤 | 兩市場完整成功 | 驗證日盤 collector 的真實來源／解析；不是新增預測準確率 |
| 當日收盤 job API probe | HTTP 404，保存失敗 | 新 workflow 尚未合併啟用，沒有虛構成功 run 或 artifact |

所有 actual 的 `available_at` 都採此次真正取得的時間，沒有回填成歷史發布／盤前可用時間。較晚執行的 15:30 任務仍須重新核對實際回傳日期、時段與合約；不保證官方資料一定已更新。

## 已執行的工程驗證

原有 128 項先通過；新增後 **136 項 pytest 通過**，Ruff、`pip check` 與兩份 workflow YAML 解析通過。新增測試含同日／同合約 outcome、結算排除、禁止最近日期替代、尚未收盤拒絕、ZIP 指紋及路徑、敘事數字竄改、盤前不變／覆盤讀回、兩市場本地全循環，以及跨日 artifact 累積。

[合成整合範例](../../examples/synthetic-shadow-bridge.json) 包含數字快照、晨報轉接、合成 actual、覆盤 patch 與本地讀回列。檔案明示 `SYNTHETIC_INTEGRATION_ONLY`；這只證明轉接契約和數字一致性，不是 Google Sheet 真實寫入或市場成效。

```bash
python -m pytest -q
python -m scripts.collect_day_actuals \
  --date 2026-11-04 --contract 202611 \
  --output reports/generated/new-official-actual-check
python -m scripts.finish_shadow_day \
  --repository fu06pi/tw-market-scenario-engine \
  --output reports/generated/new-close-job
```

每次使用新目錄，禁止覆寫先前結果。GitHub artifact 下載在 Actions 使用 read-only `GH_TOKEN`；憑證只傳 api.github.com，重新向簽名下載 URL 發送不含憑證的請求。原始行情與逐日累計資料留在 ignored 路徑／artifact，不提交公開 Git。

## 尚待真實驗證

PR 仍未合併，所以新盤前與收盤 workflow **尚未排程啟用，也沒有 Actions 真實端到端證據**。沒有盤前數字就不能從下午的 actual 補作預測。目前輸出的晨報是轉接 payload，未寫入正式或 staging Sheet；本地讀回不能宣稱 Google Sheet 寫入成功。

下一個有效驗證需有真實 08:30 前來源、08:40 前遠端 artifact、同日兩市場 actual、完整晨報／Sheet 讀回及其後逐日累積。原 11/4 凍結 checkpoint、研究報告、歷史 prediction／actual／evaluation／manifest 均未回寫。盤前 workflow 此次只換成數字後產生晨報轉接的 wrapper；11/4 計畫中的舊 workflow 指紋仍作當時版本紀錄，本次流程指紋另列於新 manifest，沒有為更新流程重選或改寫舊模型。
