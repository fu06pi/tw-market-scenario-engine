# 台股盤前 OHLC 預測與晨報

本專題研究如何用盤前可取得的資料，預測**台灣加權指數（TAIEX）**與**指定月份台指期（TX）**的當日日盤開盤、最高、最低與收盤價。量化預測先保存，晨報再補充新聞背景，收盤後用同日、同市場及同合約的官方行情逐項驗證。

[模型訓練流程圖][training] · [教授清單與成果][checklist] · [OHLC 誤差表][ohlc] · [新聞試跑紀錄][ledger]

## 模型如何訓練

TimesFM 3.0 小試驗使用 64 個歷史時間點預測下一個交易日。凍結預訓練主幹，透過損失與反向傳播更新輸出頭。8 個訓練視窗共跑 3 個 epoch，完成 24 次更新，再用 8 個驗證視窗選出 checkpoint。原完整訓練腳本仍待取得，最佳化器與資料轉換細節尚待核對。[訓練迴圈與核對依據][training]

[![TimesFM 內部結構與可訓練輸出頭][architecture]][training]

## 歷史比較結果

2026 年 1–9 月，兩市場各有 **179 個共同評估日**。下表為收盤價 MAE，以前收標準化為基點（bp），數值越小代表誤差越低。完整資料另列開、高、低、收各項點數與 bp 的 MAE、RMSE。[四價分項表][ohlc]

| 方法 | TAIEX 收盤 MAE（bp） | TX 收盤 MAE（bp） |
|---|---:|---:|
| Ridge，無夜盤 | 141.21 | 154.03 |
| Ridge，開盤加入夜盤 | 109.88 | 109.73 |
| **簡單夜盤基準** | **103.26** | **106.69** |
| TimesFM 3.0，歷史價格 | 147.71 | 158.21 |
| TimesFM 3.0，夜盤殘差 | 106.39 | 108.39 |

夜盤資訊降低 Ridge 的收盤誤差，但兩市場仍以簡單夜盤基準的收盤 MAE 最低。表中的 TimesFM 比較使用固定權重推論，共 716 次預測。各方法使用的資訊集合見研究報告，歷史標籤已用於先前研究。[TimesFM 比較與限制][timesfm]

以開盤加夜盤的 TAIEX Ridge 為例，收盤絕對誤差在 **100 點內占 16.76%**。用相對前收的 **±5 bp** 中性帶分類，收盤方向命中率為 **72.63%**。前者量距離，後者量方向，各自回答不同問題。[門檻敏感度][threshold]

## 盤前到收盤

| 台北時間 | 系統作業 |
|---|---|
| 08:30 前 | 取得先前日盤、已完成的夜盤及符合時間條件的新聞。台指期報酬使用同一月份合約的前一日盤收盤價。 |
| 08:40 前 | 凍結 OHLC、來源與模型版本。試跑另存加入新聞後的收盤方向，供同日配對。 |
| 收盤後 | 核對官方同日實際行情，計算各價位誤差與方向命中，保留來源失敗與缺漏。 |

晨報與收盤覆盤已有離線展示。正式 Google Sheet 整合仍待驗證。[系統流程與展示說明][show]

## 研究進度

- **已完成**　模型內部訓練圖、資料時間軸、兩市場八張 OHLC 比較圖、分項誤差、門檻分析、研究報告、展示素材與 SWOT。[逐項成果與證據][checklist]
- **試跑中**　截至 2026-11-20 的已完成收盤統計，新聞研究有 **2／8 筆**有效配對，每市場各 1 筆。11/18 兩組方向相同，目前樣本不足以判定新聞增益。[已驗證累計][aggregate] · [持續更新的試跑紀錄][ledger]
- **後續工作**　處理來源缺值、累積新的盤前樣本，完成正式 Sheet 讀回，再做 ARIMA 同口徑比較。[研究進度與待辦][status]

研究程式、圖源與報告目前可在 [PR #3][pr] 審閱。資料契約、驗收規則與操作指令仍可查閱。[原架構與驗收規則][original] · [開發與重現筆記][development]

[training]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/docs/presentation/MODEL_TRAINING_FLOW.md
[architecture]: https://raw.githubusercontent.com/fu06pi/tw-market-scenario-engine/d7de4eb93830616f53ad36d6a889baaffdfdf969/docs/presentation/figures/model-architecture.svg
[checklist]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/reports/academic/2026-11-23/README.md
[ohlc]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/reports/academic/2026-11-16/price-comparison.csv
[timesfm]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/reports/followup/2026-11-16/README.md
[threshold]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/reports/followup/2026-11-16/direction-sensitivity.csv
[show]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/reports/academic/2026-11-16/README.md
[ledger]: https://github.com/fu06pi/tw-market-scenario-engine/issues/4
[aggregate]: https://github.com/fu06pi/tw-market-scenario-engine/issues/4#issuecomment-7033406586
[status]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/PROJECT_STATUS.md
[pr]: https://github.com/fu06pi/tw-market-scenario-engine/pull/3
[original]: https://github.com/fu06pi/tw-market-scenario-engine/blob/c36f29e96a3c7ff90832f648b818f4ac56438275/README.md
[development]: https://github.com/fu06pi/tw-market-scenario-engine/blob/d7de4eb93830616f53ad36d6a889baaffdfdf969/README.md
