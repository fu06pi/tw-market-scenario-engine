# 純量化與新聞輔助的比較契約

研究問題：在相同盤前限制及共同日期下，新聞是否改善「收盤方向判斷」？新研究不修改原量化 O/H/L/C。

| 組別 | 輸出 | 評估 |
|---|---|---|
| A 純量化 | 鎖定模型的 O/H/L/C 與相對前收方向 | MAE／RMSE、方向 |
| B1 原量化＋解釋 | 同一組價格、新聞背景及風險說明 | 價格誤差與A相同；檢查數字忠實及來源，不宣稱新價格增益 |
| B2 量化＋新聞獨立決策 | 另存漲／平／跌，不輸出新價格或未校準機率 | 與A比較同日收盤方向，不計B2的價格MAE |

`news_comparison.py` 已提供 B2 的嚴格資料契約、凍結及共同日期評分。這是評分接口，不是已執行的模型／新聞實驗，也沒有呼叫付費 AI API。

## 凍結要求

- B2 只能引用原量化 snapshot hash 與鎖定 baseline，保留日期及 TX 合約。
- 原量化先保存，再以實際執行時鐘另存 B2；不得輸入倒填 generated_at，須在同日08:40前。
- 新聞帶 news_id、HTTPS來源、發布時間、取得時間、證據內容及時間依據；發布不晚於取得，取得不晚於決策與08:30截止。
- 保存 AI model_version、prompt_sha256、理由及版本；查詢視窗與決策規則應在第一個實驗日前另行固定。修改 prompt／模型後分組，不混算。
- 所有模型、原始快照與新聞時間須另外有可核實的保存證據。本機 hash／自填來源時間不等於正式前向資格。

## 評分要求

- 收盤方向以同標的／同合約前收為參考，沿用快照 ScoringPolicy；目前中性門檻為±5bp。
- 輸入的所有量化快照市場均列為預期比較母體；缺新聞決策、actual 或錯誤須明列，不只挑能算且表現好的日期。
- 同一快照／市場出現多份有效 B2，整對拒絕，不事後挑選。
- 列 A、B2 各自方向命中、B−A差、只有A正確／只有B正確及共同樣本數；不同 checkpoint／prompt／模型／政策分組。
- 30筆以前不給改善區間結論；至少30筆才產生固定5日區塊、2000次、seed=20261116的不確定區間。仍是研究比較，不自動成為正式驗收。
- B2 沒有新價格輸出，價格 MAE／RMSE 為不適用；不能把敘事區間中點或信心值當新價格／機率。

```bash
python -m scripts.compare_news_decisions \
  --forecasts path/to/frozen-forecast.json \
  --decisions path/to/frozen-news-direction.json \
  --actuals path/to/verified-actuals.json \
  --output reports/generated/new-news-direction-comparison.json
```

輸出中 `formal_accuracy_eligible` 固定為false，另需獨立時間證據稽核。合成測試只驗證時序、缺漏、重複與不改價格，不能證明新聞改善判斷。此次沒有新的真實 B2 比較結果，不將其包裝為已完成的市場增益。
