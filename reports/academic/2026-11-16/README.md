# 教授要求的研究成果與展示整理

2026-11-16，Asia/Taipei。本輪固定既有結果，完成流程、指標、圖表、比較及展示素材；**模型擬合0次，沒有新增獨立測試或前向準確率主張**。

## 本輪交付

- 總流程圖、資料與新聞資訊時間軸：[架構說明](../../../docs/presentation/PROJECT_FLOW.md)。
- 兩市場各 O/H/L/C 四張曲線，附相對前收帶符號誤差；另有四價總覽與 MAE 方法比較，共14張圖。
- [逐價位MAE／RMSE彙總CSV](price-comparison.csv)：24列，2市場×3方法×4價位，各列點數／bp、樣本數及缺漏。
- [完整彙總與來源指紋](metrics.json)：含方向混淆矩陣、永遠猜漲、原發布區塊不確定性及模型清冊。
- 16頁可編輯PPTX、逐頁講稿、離線可操作報告與PDF，由下列指令產生，作為本機成果包交付。
- [量化與新聞比較接口](../../../docs/presentation/NEWS_COMPARISON.md)：獨立方向決策的凍結、相同門檻／共同日期、缺漏與重複拒絕及評分；尚缺真實新聞決策比較資料。

詳細圖／PDF／PPTX由本機成果包交付，沒有將逐日行情、完整授權資料或 TimesFM 衍生權重提交Git。原逐字稿亦不提交。

## 研究範圍與結果

主比較為 `night-ablation-v1` 的已凍結輸出。每市場各方法共同179日，2026-01-02～09-30，三種方法為無夜盤Ridge、僅開盤加夜盤Ridge及預列簡單夜盤直推基準。所有來源與既有誤差先重新核對，沒有重新調參或選模型。

相對無夜盤回歸，開盤加夜盤版的收盤MAE改善22.19%（TAIEX）與28.76%（TX）；但相對簡單夜盤基準，仍差6.41%與2.85%。TAIEX開盤MAE相對無夜盤版反而惡化3.65%。完整負面結果保留，不能只挑方向命中率或只看兩條價格線相近。

最初訓練至2025/09，驗證2025Q4，已見評估2026/01–09；按事前規則每日擴展擬合。圖左側20個12月點是先前驗證期背景，不冒充新訓練。TX換月不連線，但全部點保留；各報酬仍用當次選定合約自己的前收。

歷史發布時點未核實，評估價格先前已使用；這些圖不是新未見或真實盤前結果。新排程與Sheet真實端到端仍待證據，工程測試與合成展示也不當成市場準確率。

## TimesFM／ARIMA／AI 的實際角色

本次另找到 TimesFM `google/timesfm-3.0-pytorch` 的輸出頭小樣本試驗紀錄：8個訓練視窗、8個驗證視窗，有24次 optimizer step 與非零梯度／參數更新；本次只核對紀錄及checkpoint指紋，沒有重跑該模型。它不是完整179日同口徑比較，也不是全模型微調完成。權重不散布。

ARIMA尚無此共同期間正式結果，依教授方向列後續選做。AI目前解釋同一組凍結數字，不能宣稱價格MAE因文字改善；新增方向比較接口先分清這個問題，待獨立盤前新聞決策與actual齊全後再驗證。

## 指標

- 每價位MAE：平均絕對誤差；RMSE：平均平方誤差的平方根。
- 點數及相對同標的／合約前收的bp並列。四價分別評估，不必因此改成四個獨立模型。
- 漲／平／跌採同一快照政策：相對前收±5bp內為平。100點內命中是另外的容許誤差指標，本輪沒有事後挑該門檻。
- 對完整179個共同日期比較；缺漏／失敗清單及分母保留。大誤差日期僅供診斷，不宣稱已找到消息因果。

## 重現

需要本機保留的原始來源及 `reports/generated/night-ablation-v1-verified/daily-private.json`；公開Git僅含彙總與重現程式。原始資料取得及權限依原研究規則。

```bash
python -m pip install -e '.[dev,presentation]'
python -m scripts.prepare_academic_results \
  --output reports/generated/new-academic-evidence \
  --public reports/academic/new-version
python -m scripts.build_academic_bundle \
  --evidence reports/generated/new-academic-evidence/evidence-private.json \
  --output path/to/new-academic-bundle
```

選用PDF／操作檢查另需Playwright、Chromium及PyMuPDF：

```bash
python -m scripts.check_academic_bundle --output path/to/new-academic-bundle
```

每次採新目錄，保留原研究與舊快照。產生報告不需要TEJ金鑰、新聞API或模型推論；當原凍結輸入、摘要、日期／合約或誤差不一致時直接拒絕。

## 驗收與展示順序

1. 動機／目的 → 總流程 → 資訊時間軸 → 資料前處理。
2. 實際模型清冊 → 指標 → 四價曲線與相同日期比較。
3. 量化／AI比較的成立條件 → 離線合成操作示範。
4. 使用者價值、SWOT、負面結果、限制與下一步。

真實前向數字保存、獨立新聞決策、同日官方actual與Sheet讀回仍須執行，沒有因完成簡報就視為完成實證驗收。

本輪147項pytest與Ruff通過；桌面1440px／手機390px、兩市場切換、合成示範凍結數字不變與零外部服務請求已檢查。PDF15頁已讀回，PPTX16頁與逐頁講稿、物件邊界已讀回；沒有宣稱原生Office排版引擎渲染驗證。詳見 [驗證與檔案指紋](verification.json)及[新聞比較尚待資料狀態](news-comparison-status.json)。
