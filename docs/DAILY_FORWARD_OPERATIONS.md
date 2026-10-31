# 每日 OHLC genuine forward 採樣

## 固定研究契約

- 交易日台北時間 08:20 啟動；08:30 截止後推論，08:40 前凍結。
- 模型為 `night_regression`；當日主要模型失敗才使用 `range`。
- 每日只允許一份 `forward-snapshots/YYYY-MM-DD.json`，不得補寫錯過截止的日期。
- 18:00 後以同一 `forecast_id`／`snapshot_hash` 追加實際值及評分，不修改盤前檔案。
- 累積報告包含雙市場 OHLC MAE、bp MAE、RMSE、跳空／收盤報酬／振幅誤差、方向混淆矩陣、80% 區間覆蓋、區間寬度與 interval score。
- 這條路徑只累積研究證據，不寫 Google Sheet、晨報或正式排程，`production_switch=false`。

所有已訓練 TimesFM／residual 候選仍未通過公平 gate，因此不得在每日輸出中替代固定基準。新增模型只有在預先固定的 future holdout 通過後，才能另行版本化切換。

## 執行命令

盤前：

```bash
./ops/run-forward-preopen
```

盤後：

```bash
./ops/run-forward-postclose
```

命令會依 `Asia/Taipei` 取得當日日期。休市、08:20 前、08:30 後、資料品質失敗、缺少夜盤、盤後沒有同日盤前快照或盤後資料未齊全時都會 fail-closed，並在 `reports/generated/quant/daily-forward/attempts/` 保存失敗原因。

成功產物：

- 盤前快照：`reports/generated/quant/daily-forward/forward-snapshots/`
- 盤前／盤後 receipt：`reports/generated/quant/daily-forward/forward-receipts/`
- 實際值：`reports/generated/quant/daily-forward/realizations-v1/`
- 單日誤差：`reports/generated/quant/daily-forward/evaluations-v1/`
- 累積評分：`reports/generated/quant/daily-forward/forward-metrics/`

## 排程設定

本機排程應直接使用目前 checkout，不建立隔離 worktree，因為增量資料、不可回寫快照及累積評分必須共用同一條 append-only 記錄鏈。

建議排程：

- 週一至週五 08:20 Asia/Taipei：執行 `./ops/run-forward-preopen`。
- 週一至週五 18:10 Asia/Taipei：執行 `./ops/run-forward-postclose`。

若使用 ChatGPT desktop Scheduled task，專案選本 repo、執行環境選 local project，任務 prompt 只需分別寫明執行上述命令、保留完整 stdout/stderr、不得改檔或補寫失敗日。OpenAI 官方文件指出，本機專案排程需要電腦保持開機且桌面 App 持續運行。

目前執行環境沒有可操作的 ChatGPT desktop surface；`crontab` 被 PAM 拒絕，user systemd bus 也受 sandbox 阻擋。因此程式與排程入口已完成，但本回合無法把 OS／desktop 排程實際註冊。註冊後第一個有效樣本應從 2026-11-02 開始；2026-10-30 已錯過截止，不可回填為 genuine forward。
