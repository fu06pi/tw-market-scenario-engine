# V1 OHLC 到 V0／V01 的遷移契約

更新：2026-10-16。這份文件記錄已核對的外部任務介面，以及正式切換前必須遵守的資料契約。它不代表已切換正式排程。

## 已核對的現況

既有晨報與收盤驗證不是本 repository 內的 Python 排程，而是外部 ChatGPT 任務與既有 Google Sheet：

- `財經事件晨報`：產生晨報並寫入正式 Sheet。
- `V1 台股劇本`：仍使用舊情境快照，包含 08:50 截止、Bull／Base／Bear 機率、gap／close bps、盤中 regime 與 breadth。
- `V1 台股收盤驗證`：驗證舊情境欄位，尚未讀取新的雙市場 OHLC 快照。

近期舊任務曾於 08:50 後、甚至開盤後重建快照。這些紀錄仍屬 legacy 流程，不得回標成新 V1 的準時盤前預測。正式任務與正式 Sheet 在 staging 驗證完成前維持原狀。

## 唯一可接受的新資料鏈

```text
FinMind／官方核對／TEJ overlap
  -> V1 OHLCForecastSnapshotV1（08:30 截止，08:40 前 write-once）
  -> V0NarrativePayload（逐字引用同一快照）
  -> V0 A:F,O 盤前列
  -> 同 forecast_id 的 realization／evaluation
  -> V0 G:N,P 收盤覆盤
```

V0 不得直接呼叫模型、重新計算 OHLC、改寫區間或另建方向機率。新聞只可使用 08:30 前已發布內容，且只作背景說明。

## Legacy 欄位遷移

| Legacy 概念 | 新契約處理 |
|---|---|
| 08:50 截止 | 拒絕；固定為 Asia/Taipei 08:30，08:40 後保存視為遲到 |
| Bull／Base／Bear 機率 | 不映射；不得宣稱為新模型輸出 |
| 上漲／下跌機率 | 不產生；信心欄固定說明沒有建立方向機率 |
| gap bps | 只能由凍結 open 與同標的／同合約前收推導，不另編點數 |
| close bps | 只能由凍結 close 與前收推導，不另編點數 |
| intraday regime／高低先後 | 只能標示為「待驗證情境假說」；日 OHLC 無法證明路徑 |
| breadth | 只能列作有截止時間證據的背景，不得冒充 OHLC 模型輸出 |
| legacy scenario id | 不沿用；新流程以 `forecast_id` 與 `snapshot_hash` 綁定 |
| 收盤檢討 | 必須引用同一 `forecast_id`、`snapshot_hash`、標的與 TX 合約 |

## 程式層硬限制

- `OHLCForecastSnapshotV1` 強制 TAIEX、TX 各一筆、TX 明確月份合約、08:30 cutoff、OHLC 一致性、來源 availability 與內容 hash。
- `V0NarrativePayload` 強制 08:30 cutoff，拒絕額外 probability 欄位；其 deterministic verifier 會拒絕任何文字數字修改。
- 晨報 Sheet batch 只填 A:F、O，且 O 保存 `forecast_id`、`snapshot_hash`、cutoff、persisted time 與 DQ 狀態。
- 收盤 batch 必須比對同一快照、realization、evaluation 與盤前列指紋，只能更新 G:N、P。
- staging writer 明確拒絕正式 Sheet ID；`production_switch=false` 時不得切換正式資料流。

## 切換門檻與回復點

1. 在真實交易日 08:20–08:29 手動啟動一次 `live-preopen`，於 08:40 前完成快照和 staging readback。
2. 同日 18:00 後完成 `live-postclose`，證明 A:F、O 未變且 G:N、P 綁定同一快照。
3. 通過後才恢復 staging-only shadow，累積 3–5 個完整交易日，包含所有失敗日。
4. 另行審查外部三個任務的 prompt、正式 Sheet 寫入權與 rollback，再切換正式排程。
5. 回復點是目前仍在運作的 legacy 任務與正式 Sheet；切換不得刪除或重寫其歷史資料。

目前第 1–4 項均未完成，因此新流程不得宣稱 production ready，也不得恢復 `ohlc-shadow`。

三個外部任務的逐項 prompt 草案、依賴順序與 rollback 步驟已整理於 [`V0_EXTERNAL_TASK_CUTOVER_PACKET.md`](V0_EXTERNAL_TASK_CUTOVER_PACKET.md)。該套件狀態為 `NOT_AUTHORIZED`，只能用於審查與 staging 演練。
