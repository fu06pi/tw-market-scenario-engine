# 教授要求驗收與整合交付｜2026-11-23

**使用者後續更正：**第一項所需是模型內部訓練流程。先前14頁整合版第2頁是系統流程，不能算完成這個新釐清的要求；本次另補[TimesFM實際輸出頭訓練流程](../../../docs/presentation/MODEL_TRAINING_FLOW.md)，保留舊報告不覆寫。

依使用者本次清單，將既有成果逐項核對並整合為14節研究報告。沒有新增訓練或改變固定新聞試跑；沒有把缺資料寫成模型錯誤，亦沒有把展示完成當成正式Sheet遷移。

| 項目 | 狀態 | 可核對證據 |
|---|---|---|
| 模型內部訓練流程（使用者更正） | 已補流程圖；原腳本細節待核對 | [TimesFM輸出頭訓練](../../../docs/presentation/MODEL_TRAINING_FLOW.md)；舊整合報告第2頁僅為系統背景 |
| 量化／新聞資料時間範圍 | 完成 | 訓練截至2025/09/30、驗證2025Q4、已見評估2026/01–09；TimesFM64點context另列；新聞72h／08:30截止／08:40前保存；第3頁 |
| OHLC各自MAE／RMSE | 完成 | [24列點數／bp表](../2026-11-16/price-comparison.csv)、[TimesFM完整比較](../../followup/2026-11-16/timesfm3-summary.json)；第5～6、11頁 |
| 四價預測與實際比較圖 | 完成 | 兩市場各四張；歷史背景為2025/12驗證資料，179日回溯標籤已見；第7～10頁 |
| 純量化與量化＋新聞 | 部分完成 | [共同方向比較契約](../../../docs/presentation/NEWS_COMPARISON.md)、[11/18真實配對](https://github.com/fu06pi/tw-market-scenario-engine/issues/4#issuecomment-6972126570)；每市場僅1筆，增益尚無法定論 |
| 評估門檻理由 | 完成說明與敏感度 | [固定網格](../../followup/2026-11-16/direction-sensitivity.csv)；100點為描述性容許值，±5bp為原方向中性帶，未宣稱經濟最佳；第4頁 |
| 研究報告 | 完成整合版 | 動機、資料、方法、誤差、負面結果、前向缺漏、結論與限制共14節 |
| 系統成果展示 | 完成展示素材 | 既有離線HTML市場切換／合成晨報覆盤、16頁可編輯簡報；正式Sheet寫入／讀回仍未完成 |
| 產品定位、使用情境、SWOT | 完成 | 學術研究與學習輔助、來源與數字可追溯、可靠性與樣本限制；第13頁 |
| ARIMA同口徑對照 | 後續選做 | 未完成，不以現有TimesFM或基準結果代替 |
| 進一步調參／策略切換 | 後續 | 先處理解析可靠性與累積前向樣本；不反覆調已見評估 |
| 美化流程／展示 | 已有基本版 | 保持可讀與離線操作，後續可再調整 |

## 最新實證邊界

- 原Ridge與簡單基準的兩市場三種方法，皆以開盤MAE最小。開盤加夜盤Ridge為TAIEX154.83點、TX183.73點；不能外推到所有模型。
- TimesFM夜盤殘差TAIEX最高價MAE74.92bp，小於開盤86.25bp；收盤仍比簡單夜盤基準差3.03%／1.60%，沒有達成原5%改善。
- 最新已完成收盤累計截止11/20：4個應執行日、8筆應配對、有效2筆、覆蓋25%；每市場1筆。11/18 A與B方向相同，TAIEX均錯、TX均對，改善差0；樣本不足，不發布穩健改善區間。[累計收據](https://github.com/fu06pi/tw-market-scenario-engine/issues/4#issuecomment-7033406586)
- 11/19、11/20的固定解析器遭遠月高低價「-」阻斷，列來源／工程失敗，不計成預測錯誤。新版本修正與試跑切換仍需分版本留證；本次未更動它。
- 11/23此次實際時鐘為13:19:57，盤前階段已逾時，新增[late_failure收據](https://github.com/fu06pi/tw-market-scenario-engine/issues/4#issuecomment-7092326803)，伺服器13:20:49，精確內容、hash與作者／ledger回讀通過。沒有補造預測，也沒有提前執行收盤；不把今天列為已完成收盤累計。

## 交付與重現

本機成果目錄為`/workspace/outputs/tw-market-professor-2026-11-23-final/`：`report.pdf`、`report.html`、`CHECKLIST.md`、`price-comparison.csv`、10張既有圖、`evidence/`公開彙總、`demo/`原離線展示及16頁簡報。舊成果包不覆寫；不含逐字稿、完整行情、原始新聞HTML或權重。

```bash
python -m scripts.build_professor_delivery \
  --comments path/to/complete-github-api-comments.json \
  --latest-failure path/to/todays-verified-late-failure.json \
  --assets path/to/original-academic-bundle \
  --output path/to/fresh-professor-delivery
```

產生器核對固定protocol／checkpoint／prompt，重放原始GitHub留言，與最新已發布累計精確相等後才產生報告。若資料、hash或配對分母有衝突則拒絕；不呼叫模型、新聞或行情服務。PDF與瀏覽器驗證環境另用Playwright／Chromium／PyMuPDF。各項實際檢查與檔案指紋見成果包`verification.json`及`package-manifest.json`。

報告產生器重現 11/23 整合版。模型訓練圖的後續補充可在 GitHub 閱讀，另有本機兩頁 PDF。[模型訓練流程與內部結構](../../../docs/presentation/MODEL_TRAINING_FLOW.md)
