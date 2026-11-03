# TAIEX 三折失效診斷

2026-10-24；僅 development 事後診斷，不是新 holdout、不選新模型。保留每折80日、共240日，新舊模型與固定基準日期、actual、前收和基準預測逐一核對；失敗0日。

| fold | 新 close MAE | 舊 close MAE | 基準 close MAE | 新 gap MAE | 基準 gap MAE | 新 body MAE | 基準 body MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 |225.62|228.47|187.21|79.22|112.56|186.67|168.59|
| 2 |510.93|493.74|604.86|123.80|168.22|446.31|486.57|
| 3 |508.45|537.98|577.36|95.60|141.43|471.63|496.65|

全部單位為點。gap誤差是預測開盤減實際開盤；body是C−O；range是H−L。絕對誤差不可相加來分解close MAE。

第一折18個實際C/O跌超過0.5%的日子，新模型close MAE364.52（基準246.10），全部偏高；24個漲超過0.5%的日子close MAE272.72（基準228.66），全部偏低。新模型相對基準的總絕對誤差多3072.57點，其中最大的五個超額誤差合計1640.20點；仍保留這些樣本，不刪除極端日。這支持進一步檢查日盤body對大波動反應不足的假說，但不是因果證明，條件分組本身也會造成方向偏差。

新模型三折range平均偏差為−211.94、−517.77、−498.93點；原模型為−193.48、−445.44、−389.92點。損失正規化雖改善全樣本close，卻加劇振幅低估，不能稱為全面改善。第二折close也比舊模型變差。

分組使用當日實際C/O的固定±0.5%門檻，只是事後診斷，不能用於盤前選模。不得根據這些標籤直接挑模型或宣稱可辨識當天大跌。可檢驗的下一步：只用各折訓練資料檢查GBUD的u/d近零值及log尺度分布、訓練標籤振幅與早停損失各項貢獻，再決定是否另開價格尺度/range損失候選。尚未執行該候選，尚未提升驗收狀態。

重現（輸出路徑需不存在，避免覆寫證據）：
```bash
.venv/bin/python research/diagnose_head_folds.py reports/generated/quant/experiments/timesfm3-output-head-walk-forward/88b26bf3d30c9ded6d0d reports/generated/quant/experiments/timesfm3-output-head-walk-forward/dd4ef236644360b9e734 /tmp/head-fold-diagnostic-replay.json
```

數值與來源hash見同名JSON。原逐日價格不對外發布。
