# 可比較的簡單基準

`src/tw_scenario/baselines.py` 提供固定版本 `simple-baselines-v1`。價格預測與純方向分類分開評分，沒有用任意百分比將「猜漲」轉成價格。

## 規則與用途

| 家族 | ID | 規則 |
|---|---|---|
| 價格 | zero_return | O、C 都預測為同標的／同合約前收 |
| 價格 | expanding_mean | 用所有允許的過去樣本，分別平均 log gap、log body |
| 價格 | previous_return | 沿用前一觀測交易日的 gap／body 報酬，再用今天參考前收還原 |
| 價格 | rolling_mean_20 | 固定使用最近 20 個過去樣本的 log gap／body 平均；不搜尋視窗 |
| 純方向 | always_up／always_down／always_flat | 開盤、盤中與收盤各固定預測漲／跌／平 |
| 純方向 | historical_majority | 每個目標按已知過去標籤的多數方向預測；同票依平、漲、跌固定順序 |
| 純方向 | previous_direction | 沿用前一觀測交易日各目標方向 |
| 夜盤資訊 | overnight_gap | 原固定權重開盤方向規則；只有開盤方向，沒有收盤／價格／概率 |

所有價格基準的上下影線都採相同的過去非負 log 影線平均，保持 H≥max(O,C)、L≤min(O,C)。20 日視窗只影響該方法的 gap／body 均值；不另外更換影線估計法。

前一日報酬使用同一歷史列自己的前收計算，不照搬昨天的絕對點位。TX 的基準請求與歷史列必須是同一明示合約；不同合約或日夜盤不會自動拼接。

夜盤規則沿用 0.50×台指夜盤報酬＋0.25×SOX 報酬＋0.25×TSMC ADR 報酬，輸入單位為百分比，決策中性帶為 ±0.15%。評分實際方向仍遵守共用 `ScoringPolicy`，目前實際標籤中性帶為 ±5bp。決策門檻與評分標籤門檻是不同的預先定義。

## 盤前接口與研究接口

盤前使用 `predict_preopen_baseline(name, inputs, history, policy)`：

- `inputs` 是既有 `PreopenInputs`，會驗證 Asia/Taipei 08:30 截止及 08:40 前產生等資料契約。
- 歷史標籤日必須嚴格早於當天，收盤結果在截止前已可用，日期不得重複，市場及合約必須一致。
- 夜盤特徵仍須符合 inputs 的可用時間驗證，缺少三項任一項就失敗，不補零。
- 返回研究候選，不代表已持久化或通過正式績效資格；後續仍須有快照與保存時間證據。

`price_baseline`、`direction_baseline` 是純研究核心，只接收過去 components；它們本身沒有時間戳，不是盤前資料驗證的替代品。純方向結果的 prices／components 為 None，不能計算價格 MAE 或推論概率校準。

## 統一比較流程

```bash
python scripts/compare_simple_baselines.py \
  --reference reports/generated/historical-validation-v1 \
  --output reports/generated/simple-baseline-comparison-new
```

reference 指向原歷史研究的 plan、summary、locked-selection 與本地 daily-private。來源 SHA-256 必須吻合。若本地研究結果不存在，先依 [歷史研究報告](../../reports/research/historical-validation-v1/README.md) 使用有權限的行情快取重現。

輸出目錄必須尚不存在。比較流程先保存 plan，再計算結果，並遵守：

1. 相同市場、合約、日盤、前收、觀測日期與評分政策；所有方法使用相同 60 筆共同暖機條件。
2. 歷史基準、凍結的拆分模型與直接收盤模型使用相同前日 OHLC 資訊範圍。方向常數可忽略資料，但不擴大其評分日期範圍。
3. 原模型的預測不重訓、不調參；從原研究讀回，驗證日期完整性、OHLC 實際值及原指標一致。
4. 價格基準只按驗證期收盤 bp MAE 選出；方向基準只按驗證期收盤命中率選出。同票依固定方法順序處理，在評估期前保存 selection。
5. 不混合方向與價格成績成為一個總分。每列保留 expected／scored／coverage、失敗與混淆矩陣；未完整覆蓋的基準不能參與選擇。
6. 評估期間若候選與預選基準出現失敗或樣本分母不同，不計算改善率；基準誤差為零也不填無限改善率。
7. 夜盤屬不同資訊集。現有價格資料集沒有三項完整夜盤／海外特徵，該列明示 unavailable，不參與排名。

本次沿用的歷史評估期已被先前研究看過。因此 plan／summary 明示 `new_independent_test_evidence=false`、`formal_acceptance_pass=false`，不得將比較表當成新的未見樣本驗證。後續模型改進須另留新資料或使用事前固定的獨立驗證流程。

原資料歷史可用時間未經驗證，仍沿用回溯研究的明示假設；不回寫 available_at，也不建立虛構盤前紀錄。行情原檔與逐日實際值留在忽略追蹤的本地快取，只發佈彙總。

## 已執行結果

完整表、驗證期選擇與輸入／程式碼指紋見 [基準比較報告](../../reports/research/simple-baseline-comparison-v1/README.md)。此次未更換每日流程的預設輸出，也未宣稱模型已改善。
