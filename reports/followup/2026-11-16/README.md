# 四項研究缺口：第二輪推進

本輪實際完成TimesFM比較、門檻敏感度與大誤差查核，並準備獨立、可保存伺服器時間證據的新聞研究試跑。沒有將未來資料或下午補建當成有效盤前樣本。

## 1. TimesFM完整分項比較：已實際執行

固定官方TimesFM3權重revision `43046b85ec22d584a13f8098c2ed39c889e129c2`，context64、horizon1、CPU4執行緒、官方past-only z-normalization；先凍結[計畫](timesfm3-plan.json)，再執行。兩市場、兩變體各179日，全部完整，參數更新0次。這是zero-shot比較，先前8訓練／8驗證的輸出頭小試驗沒有擴大或混算。

| 市場／方法 | O MAE bp | H MAE bp | L MAE bp | C MAE bp |
|---|---:|---:|---:|---:|
| TAIEX 歷史價格TimesFM | 38.271 | 111.297 | 113.079 | 147.709 |
| TAIEX 夜盤殘差TimesFM | 86.247 | 74.922 | 79.200 | 106.390 |
| TX 歷史價格TimesFM | 128.135 | 132.669 | 144.909 | 158.210 |
| TX 夜盤殘差TimesFM | 44.142 | 73.713 | 85.981 | 108.389 |

夜盤殘差版收盤仍比預列簡單夜盤基準差3.03%／1.60%，TAIEX開盤也高。沒有依結果改視窗或再挑參數；不能宣稱通過原5%價格改善門檻。不同資訊集合分開標示，不是只改演算法的受控比較。

執行220.7秒、尖峰RSS約2813MiB；未購買運算。計畫／結果及O/H/L/C點數MAE、RMSE、bp、方向、投影次數與失敗見[summary](timesfm3-summary.json)。歷史評估已見、預訓練重疊未知、歷史發布未核實，仍為回溯研究。只輸出研究結果，不散布權重或衍生checkpoint，不作生產／商業部署。

## 2. 門檻依據與敏感度：已補齊研究說明

5bp是沿用政策的中性帶，用來避免把極小變動直接當方向，且相對前收正規化。它不是自然常數、已證明的經濟最適值或交易成本模型。

先凍結[診斷計畫](diagnostic-plan.json)的0／2／5／10／20bp網格，驗證與已見評估分開，不選新最佳門檻，`config/targets.yaml`原指紋保持不變。原開盤加夜盤Ridge在已見評估的5bp收盤方向命中為72.63%／72.07%，而「誤差≤100點」僅16.76%／18.44%；兩者不是同一種正確率。

完整網格及實際平盤比例見[CSV](direction-sensitivity.csv)，其他價格容許誤差見[診斷JSON](threshold-and-error-audit.json)。若將來研究交易成本或經濟最佳門檻，需另用訓練／驗證設計，不能重挑這次已見結果。

## 3. 十個大誤差案例：已完成可核對的工程查核

各市場固定取原夜盤Ridge收盤絕對bp誤差最大的五日。官方日盤OHLC、自己的前收／同合約、日期lag、原alpha0.1模型重現及組件價格重建均通過。10次原預測重現共20個Ridge head擬合核對，沒有新增超參數搜尋。

九案body誤差較大；gap log誤差＋body log誤差精確等於收盤log誤差。這只是誤差所在位置，不是市場／新聞因果。未檢查問題與歷史可用時間不因此被全面排除。完整查核布林、合約、換月／資料年齡及代數分解保留於診斷JSON；逐日實際行情保持ignored。

## 4. 新聞配對與真實流程：有限期獨立試跑

已實作[研究協議與執行程式](../../../docs/presentation/FORWARD_TRIAL.md)：固定量化checkpoint、新聞指令與資料時間窗，B只另存方向；以專用[Issue #4](https://github.com/fu06pi/tw-market-scenario-engine/issues/4)的不可改寫留言及GitHub伺服器時間回讀保存。實際留言寫入／讀回測試成功，但那是下午preflight，新增盤前樣本0。

研究排程已建立、啟用並由服務讀回：2026-11-17～11-30共10個工作日，盤前08:15／收盤15:30。方案最多5個啟用任務，因此本研究使用一個任務、兩個時段；原四個已啟用任務及正式Sheet保持不變。固定執行程式commit `398d3b7b35f8d007178037ccbc8e06b6ee5d19a6`；後續文件提交不改此研究版本。完整排程、下次執行時間與限制見[實際啟用狀態](trial-status.json)。尚未首次執行，預約成功不代表執行成功。

真正的新聞增益仍需要後續同日配對；10日只驗證流程與描述性結果，30筆前不給穩健改善區間。Sheet實際寫入／讀回及TimesFM全模型微調沒有在本輪完成。

## 重現

現有學術環境使用timesfm3.0.2、torch2.8.0+cpu、官方本機快取權重；依模型授權僅研究使用。選用套件於pyproject的timesfm-research extra，原shadow runtime不引入Torch。

```bash
python -m scripts.run_timesfm_comparison \
  --weights path/to/43046b85ec22d584a13f8098c2ed39c889e129c2 \
  --output reports/generated/new-timesfm-comparison
python -m scripts.diagnose_thresholds_and_errors \
  --public reports/followup/new-version \
  --private reports/generated/new-error-audit
python -m scripts.build_followup_report \
  --timesfm reports/generated/new-timesfm-comparison \
  --diagnostics reports/followup/new-version/threshold-and-error-audit.json \
  --output path/to/new-followup-report
```

每次新目錄，不覆寫舊快照與研究。詳細圖、PDF及8頁補充簡報作為本機成果包交付，Git只保留方法、彙總與指紋。
