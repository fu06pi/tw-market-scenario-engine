# TimesFM 3 實際成果與未達標紀錄

更新：2026-10-17，Asia/Taipei。**研究已可重現；模型未達驗收門檻，尚未接管晨報。**

## 這次做了什麼

先前研究程式未提交、checkpoint 不在目前環境；不能只靠對話宣稱可用。本次在 `research/timesfm3-recovery-20261017` 分支重建 `src/tw_scenario/research/`，重新下載官方指定權重、重新取真實 TEJ 試用資料、重新訓練並保存可核對的摘要。未合併 main，未改寫舊預測或排程。

官方 TimesFM 3.0.2 / torch 2.8.0+cpu，權重 `google/timesfm-3.0-pytorch` SHA `43046b85ec22d584a13f8098c2ed39c889e129c2`；官方程式查核 commit `e51928e27119cb17bebc005be2696b75e0a9e688`。官方 model 檔約1.323GB；LICENSE 為非商業、非 production，研究衍生 checkpoint 不散布。

TEJ 官方 SDK 連線可用，不印 token；正式 `TWN/AFUTR`、`TWN/AFUTRSTD`、`TWN/AFUT_T1` 均回 PDB003。`TRAIL/TAPRCD/Y9999` 實取2025年243筆，`TRAIL/TAFUTR/TX202603`196筆中9筆缺OHLC；保留原始與排除紀錄，不填補缺日，選2025/07/04起124筆連續有效區段。現有資料只到2025/12/31，**不可當成目前行情**。日曆與歷史發布時間未獨立驗證，整輪只稱回溯研究。

## 比先前 pilot 更嚴格的比較

- context=32、horizon=1、batch=8，兩市場一致；僅用同合約歷史。
- 訓練截止2025/09/30；完整2025/10～11月40個驗證交易日。12月 final-test 先隔離，未送入特徵、loss 或評分。
- TAIEX148訓練樣本，TX29；小樣本尤其限制TX，不能宣稱完成長期研究。
- 三組預先固定試驗：g/b/u/d + lr1e-5、g/b/u/d + lr1e-4、相對OHLC + lr1e-4。每組最多3 epochs、2次epoch無改善早停；總預算與設定見config。並未調到達標為止。
- TAIEX133次、TX28次真實 optimizer updates；737,856個輸出頭參數可訓練，backbone固定。這是實際 head fine-tuning，**不是全模型微調**。
- 使用相同官方 `decode` 路徑，驗證 autograd版本與官方版本輸出完全相同；無夜盤時另與公開 evaluator 比對誤差0。避免訓練、推論各用一套不一致前處理。
- checkpoint載回後，40天驗證預測逐值一致；不同程序再讀回亦重現相同MAE。
- 零報酬、歷史形態、ARIMA(1,0,0)全部完成40天比較，無缺失/不收斂日期。ARIMA以每個時點之前的資料擬合，不用未來值。

## 結果：仍未達標

下表是40個驗證交易日的**收盤MAE（點）**，不是 final-test 準確度：

| 市場 | 零報酬 | 歷史形態 | ARIMA | 符合O/H/L限制的最佳TimesFM |
|---|---:|---:|---:|---:|
| TAIEX | 279.50 | 292.11 | 286.34 | 284.03 |
| TX | 297.30 | 324.37 | 303.80 | 314.68 |

TAIEX TimesFM 比最佳基準差1.62%，TX差5.85%；要求是兩者至少改善5%，因此兩者均不過。TAIEX選模最後保留epoch0（原始權重）：**本輪微調沒有帶來可採用的驗證增益**。TX使用相對OHLC表示的epoch3，但仍敗給基準；g/b/u/d的較低close錯誤候選則不符合O/H/L約束，不偷偷忽略護欄。

以5觀測日移動區塊、2000次 bootstrap 計算模型/基準close MAE比值的95%區間：TAIEX [0.9275, 1.1214]，TX [0.9857, 1.2043]。沒有證明穩健優勢。八個OHLC的點/bp MAE、RMSE、模型更新、各試驗與基準詳細數字，見 `reports/research/2026-10-17-timesfm3-validation.json`。分位數未校準成OHLC聯合區間或漲跌概率。

TAIEX完整試驗約81.1秒、TX約40.0秒；程序peak RSS約2803.5MiB（2.74GiB），CPU可負擔本輪。這不代表全模型訓練資源已足夠。

## 夜盤增益：已實作，未完成訓練

核對TAIFEX官方報表 `futDailyMarketReport`，marketCode=1為盤後；確認合約月份、最後成交價、成交量及官方交易日歸屬說明。covariate使用同合約「已完成夜盤close/上一日盤close」對數報酬，不使用未知日盤價量。

已實作官方擷取、日期/時段/合約驗證、缺報價mask、官方core decode past-future covariate、僅train split擬合的night ridge regression基準，以及獨立設定 `timesfm3-night.json`。

但官方回HTTP429，已保留51日快取、改單線2秒間隔、按Retry-After有限等待後仍429；官方批次下載頁也逾時。**未繞過代理或限流，未把不完整夜盤資料用來訓練，尚無夜盤增益結果**。後續須等服務可用後從快取續跑，仍不得碰final-test。歷史publication delay仍未驗證。

## 重跑與保存

```bash
cd /workspace/tw-market-scenario-engine
bash scripts/research/install.sh
# 只在缺少快取時用此命令下載指定revision
DOWNLOAD_TIMESFM3=1 bash scripts/research/install.sh
.venv/bin/python scripts/research/prepare_data.py
.venv/bin/python scripts/research/run_experiment.py
.venv/bin/python scripts/research/verify_experiment.py RUN_DIRECTORY
.venv/bin/python -m pytest -q
```

夜盤資料完整後才執行：

```bash
.venv/bin/python scripts/research/prepare_night.py
.venv/bin/python scripts/research/run_experiment.py --config config/research/timesfm3-night.json
```

程式、固定設定、CPU lock及不含原始行情的摘要保存在Git分支。raw/processed、權重、checkpoint、完整run留在ignored路徑；不是已確認的跨環境備份。研究衍生權重不得公開或提交。

本輪run：`reports/generated/timesfm-research/20261017T104415-cf07864e/`。保存config、data checksum、所有試驗、checkpoint及重現/不確定性結果。必要的資料歷史版本和權重仍需確認允許的持久保存位置；不能宣稱一次環境快照會永久保存一切。

## 下一階段的確切阻塞與驗收

1. **資料**：正式TEJ日盤/夜盤/基本資料權限與最新、較長歷史資料；官方夜盤限流解除或取得允許的匯出檔。不是缺token，勿把金鑰貼聊天。
2. **研究**：可靠交易日曆、歷史available_at、TX換月政策、ARIMAX/夜盤完整基準、跨年度與換月樣本、預訓練重疊風險。
3. **驗收**：目前點預測已未達標；未做final-test，未校準80% OHLC intervals，沒有75–85% coverage證據。
4. **整合**：V0任務prompt/實際排程介面、Sheets寫入與讀回、新聞來源、外部持久保存仍待確認。只有Sheet網址不能證明整合完成。
5. **運作**：不以2025舊行情生成今日盤前快照，不改08:30/08:40門檻，不使用開盤後資料補檔，不切換正式排程。

只靠重訓同一小批日線，不能保證出現不存在的可預測訊號。已採用預先限定預算、完整驗證與失敗揭露；有新資訊來源時繼續研究，仍以獨立最終測試與真實盤前證據決定是否採用。無法保證一定達標，更不保證交易獲利。
