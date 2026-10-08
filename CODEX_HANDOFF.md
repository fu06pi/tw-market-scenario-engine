# Codex 網頁版交接
日期：2026-10-08。Repository：fu06pi/tw-market-scenario-engine；分支：main。

## 已確認需求
- 台灣加權指數與台指期每日 OHLC。
- 新流程：量化模型預測 → 凍結數字 → 加入消息解釋 → 文字報告。
- V1 負責數字與回測；V0 未來讀取 V1 做文字敘述。
- 消息先解釋，之後才測試加入模型的增益。
- TEJ 暫緩確認，先做官方資料及 CSV 替代路線。
- 開發完成前維持舊每日作業與歷史紀錄。

## 現況
完成了 OHLC 架構方案，尚未完成新模型、完整歷史取數或準確率驗證。
舊程式提供情境／快照／評分骨架。models.py 是資料結構；固定權重夜盤 baseline 不是 ARIMA 或 TimesFM。
先讀 AGENTS.md 及 docs/architecture/TW_OHLC_QUANT_ARCHITECTURE.md。README 下方 legacy 內容與舊 isolation 文件供原系統參考；新模型依此次數字主導需求實作。

## 第一個實作任務
1. 檢查環境、執行原有測試。
2. 在既有 Python package 內建立 OHLC 統一資料格式、交易日／時段／合約處理及 CSV adapter。
3. 實作 TWSE 指數 OHLC、TAIFEX TX 合約日／夜盤 adapters，驗證網路與歷史覆蓋。下載失敗要回報實際原因並保留 CSV 路徑。
4. 保留來源、fetched_at、歷史 available_at 或保守發布時間假設；缺漏、休市、無成交不可當成零報酬。
5. 以同合約前收建立 reference close，預測 g/b/u/d 並還原有效 OHLC。換月規則事前固定，不用當日收盤後資訊選早上合約。
6. 建立 naive 與 ARIMA 相同介面；預留 ARIMAX 的已知夜盤特徵。TimesFM 3 留待後續對照，不讓大型權重阻塞第一階段。
7. 實作 walk-forward 及訓練／驗證／最終測試分離，兩標的各 O/H/L/C 輸出點數 MAE、RMSE、bp MAE、基準增益與覆蓋率。
8. 用實際歷史資料跑可重現的小規模回測；資料完整後再正式回測。若只能用 fixture，必須標示為程式驗證，不是實測績效。
9. 提供 CLI、資料品質報告、測試結果、實作狀態與 PR。

## 驗收
- 兩標的可用相同格式匯入並執行基準／ARIMA。
- TX 月份、日夜盤、最後成交價／結算價明確區分。
- 08:30 cutoff 不讀取 D 日日盤結果，模型選擇及資料處理不偷看未來。
- H ≥ max(O,C)、L ≤ min(O,C)，所有價位為正。
- 缺漏、失败日列入報告；模型與基準比較同一批日期。
- 不以未來調整的連續合約或隨機切分取得虛假成績。
- 沒有實測前不宣稱模型已做準或 TimesFM 較好。
- 原有每日排程、Google Sheet、歷史檔未受開發影響。

## Cloud 環境
Python 3.11+；使用 pyproject.toml。先試 python -m pip install -e '.[dev]' 與 python -m pytest -q。
模型依賴由實作時加入 optional dependencies 並固定可重現版本。第一階段不需要 TEJ API key。
自動取數需開放 www.twse.com.tw、www.taifex.com.tw 及實際下載來源；若未開放網路，明確回報並支援離線 CSV。

## 新任務提示詞
請讀取 AGENTS.md、CODEX_HANDOFF.md 與 docs/architecture/TW_OHLC_QUANT_ARCHITECTURE.md，依第一個實作任務建立官方資料／CSV adapters、OHLC 基準與 ARIMA、逐日 walk-forward 及八項誤差報告。先跑原有測試，完成後提供可重現指令、真實資料品質及測試結果，建立 PR。TEJ 暫緩，消息先解釋，TimesFM 3 後續比較；維持既有排程、Google Sheet 與歷史快照，不能編造模型績效。
