# TAIEX 公開歷史資料封存

本次範圍是加權股價指數 **日線 OHLC**，不是五分鐘逐筆資料、TX 期貨或總報酬指數。未修改晨報、Sheets、模型與正式排程，也未執行訓練。研究用途；各來源條款仍適用，公開可查詢不等於任意商用授權。

## 來源與限制

- `twse/taiex/monthly/`：使用者指定 TWSE 頁面的官方月查詢原始 JSON，含 SHA256、擷取時間、來源 URL。頁面註明最早 1999-01-05；已取得 1999-01、02。1999-03 及獨立檢查的 04 回覆 HTTP 307 安全拒絕頁，沒有 Location；未繞過限制。`manifest.json` 明列未取得月份，不能視為完整官方歷史。
- `twse/taiex/openapi/`：官方 OpenAPI `https://openapi.twse.com.tw/v1/indicesReport/MI_5MINS_HIST` 的當月資料，另外封存，不能補稱歷史月查詢已成功。
- `finmind/taiex/`：經使用者同意，使用專案既有 FinMind API 合約 `TaiwanStockPrice` / `TAIEX` 分檔補充。原始回應在 `yearly_raw/`；`daily.csv` 與 `yearly/` 僅輸出日期及 OHLC，單位指數點。未混入 ETF、期貨或猜測成交量單位。
- FinMind 匿名查詢發生 HTTP 402，實際訊息 `Requests reach the upper limit`，因此停止下載。這是配額阻擋，不能推論該年份無資料或必須付費。1990 年曾成功回覆空資料，1994 年查詢則受配額阻擋；1990–1998 仍未完成確認。最新年度亦未完成保存。
- `reconciliation.json`：官方與 FinMind 重疊日期逐欄十進位比較，任何差異完整揭露，資料保持分檔。

資料截止 2026-10-21（Asia/Taipei），排除抓取當天未完成日線。保存歷史週六交易，不用現代週一到週五規則刪除。檢查日期、重複、有限正數、OHLC 關係；不補零、不捏造休市日。缺日是否為休市仍需交易日曆驗證。`fetched_at`/`retrieved_at` 是本次擷取時間，**不是歷史當時可取得的證據**；不得直接用來宣稱無資料洩漏的盤前回測。

## 重跑

在 repository 根目錄，使用已安裝專案的 Python：

```bash
PYTHONPATH=src python -m tw_scenario.twse_history --as-of 2026-10-21
PYTHONPATH=src python -m tw_scenario.finmind_history --as-of 2026-10-21 --start-year 1990
PYTHONPATH=src python -m tw_scenario.finmind_history --as-of 2026-10-21 --offline
python -m pytest -q
```

月資料遇首個錯誤停止；FinMind 遇 402/403/429 停止，不立即重試配額拒絕；暫時伺服器錯誤有限重試。完整有效快取會先驗證雜湊再重用。`--offline` 僅禁止 FinMind 網路下載，仍會向官方 OpenAPI 取得核對資料。部分完成退出碼 **2**，完整請求區間成功為 **0**；不要把退出 2 當成功。

配額恢復後再續跑；不需把金鑰貼到聊天。此工具目前使用匿名公開存取，不會把 TEJ 或其他憑證寫入 URL/檔案。若日後改用授權 API，需另行安全接線及確認資料保存權限。原始與衍生資料有重複內容，這是可稽核封存；模型權重、環境及 TEJ 原始資料不在此目錄。
