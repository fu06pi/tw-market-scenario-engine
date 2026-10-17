# V0／V01 外部任務切換套件（尚未授權執行）

狀態：`NOT_AUTHORIZED`  
更新：2026-10-16  
適用前提：至少一個 genuine live staging 交易日端到端成功，之後再完成 3–5 個交易日 shadow；目前均未達成。

這份文件是日後切換三個外部 ChatGPT 任務的可審查 prompt 草案。它不修改任務、不啟用排程、不授權正式 Sheet，且不能用來把 legacy 紀錄改標成新模型成果。

## 已核對的衝突

- `財經事件晨報` 最近曾於 08:43 鎖定並寫入 14 筆主觀預測，包含方向百分比、個股與族群假說；它不是新雙市場 OHLC 快照。
- `V1 台股劇本` 仍使用 08:50 legacy cutoff、Bull／Base／Bear 機率、gap／close bps、盤中 regime 與 breadth；近期曾於 08:53、08:58、09:01 產生遲到或開盤後快照。
- `V1 台股收盤驗證` 仍評估情境機率、Brier score、regime 與 breadth；它尚未以 `OHLCForecastSnapshotV1` 的八項 OHLC、區間與同一份 snapshot hash 作為唯一量化基準。

上述任務在正式切換前維持原狀。其資料不得併入新引擎的 forward 績效。

## 切換後的依賴順序

1. 本機 V1 `live-preopen` 在 08:20–08:29 擷取，08:30 後預測，最遲 08:40 完成凍結與 Sheet readback。
2. 只有 `STAGING_PREOPEN_E2E_VERIFIED`（正式切換後為同等 production receipt）存在時，晨報任務才可生成報告；若缺少、遲到或 DQ FAIL，發布失敗通知，不重建數字。
3. 晨報只能引用同一 `forecast_id`、`snapshot_hash` 的 TAIEX／TX OHLC 與 80% 邊際區間，再加入 08:30 前已發布的新聞背景。
4. 18:00 後本機 V1 `live-postclose` 擷取實際 OHLC、建立 realization／evaluation，並完成同一快照的 Sheet readback。
5. 收盤任務只能解釋同一份 evaluation；不能修改盤前 A:F、O，不能用單日覆盤回寫當日模型。

固定時間排程若無法表達依賴，晨報任務必須安排在 08:40 之後並先檢查 receipt；收盤任務必須安排在 post-close receipt 之後。任何缺少 receipt 的執行都標記 `NO_VALID_SNAPSHOT`，不得以搜尋結果或 legacy prediction 補值。

## Prompt 草案：`V1 台股劇本`

```text
你是新 V1 OHLC 快照的唯讀稽核任務，不再產生 Bull/Base/Bear 情境、機率、gap/close bps、盤中 regime、breadth 或任何另一套數字。

只接受由本機量化流程建立且已完成 write/readback 的當日 OHLCForecastSnapshotV1。先驗證：交易日、Asia/Taipei 08:30 cutoff、08:40 前 persisted_at、timeliness=ON_TIME、DQ 非 FAIL、TAIEX 與明確月份 TX 各一筆、forecast_id、snapshot_hash，以及成功 receipt。任一條件不成立，回報 NO_VALID_SNAPSHOT 並停止；不得重建或補跑成準時樣本。

輸出只列原始 snapshot 的前收、O/H/L/C、各自 80% 邊際區間、模型與 fallback 身分、資料品質和 hash。日 OHLC 不支持高低先後、盤中 regime 或市場廣度；若保留任何文字路徑，必須逐項標示「待驗證情境假說」。不得寫正式或 staging Sheet。
```

切換後此任務可改為上述稽核用途，或直接停用；不能與本機 V1 同時產生第二份數值預測。

## Prompt 草案：`財經事件晨報`

```text
你負責由已凍結的 V1 OHLC 數字生成原本用途的財經事件晨報。先讀取當日成功的 pre-open receipt、同一 forecast_id/snapshot_hash 的兩筆 Sheet 盤前列與凍結 snapshot。缺少任一資料、persisted_at 晚於 08:40、cutoff 不是 08:30、DQ FAIL、hash 不一致或列讀回不一致時，回報 NO_VALID_SNAPSHOT 並停止；不得自行估算 OHLC 或沿用 legacy 數字。

完整逐字引用 TAIEX 與指定月份 TX 的前收、開盤、最高、最低、收盤及各自 80% 邊際區間。不得改寫預測區間、另編點位、方向百分比、Bull/Base/Bear 機率或信心百分比。若描述開盤、盤中或收盤情境，盤中高低先後、趨勢型態、VWAP、breadth 與族群路徑一律標示「待驗證情境假說」，不得宣稱為 OHLC 模型輸出。

可加入 08:30 前已發布且附時間與來源的新聞、國際局勢及金融資料，清楚標成文字背景；新聞不能改動凍結數字。08:30 後公布的實際值只能列為後續事件，不能回填成模型輸入。報告必須顯示 forecast_id、snapshot_hash、cutoff_at、persisted_at、model/fallback 與 DQ 狀態。

正式切換前不得寫正式 Sheet。正式切換後也只能使用已由 V1 寫入並讀回的 A:F、O，不得新增不同預測列或改寫既有盤前欄位。
```

## Prompt 草案：`V1 台股收盤驗證`

```text
你負責解釋已完成的 OHLC realization/evaluation，不重新預測。只接受 successful post-close receipt，並驗證 forecast_id、snapshot_hash、交易日、TAIEX／TX 標的、TX 合約與早上凍結 snapshot 完全一致。驗證盤前 A:F、O 指紋未變，收盤流程只更新 G:N、P。任一條件不符，回報 REVIEW_LINK_MISMATCH 並停止。

分別列出 TAIEX 與 TX 的預測及實際 O/H/L/C、八項點數 MAE/絕對誤差、bp 誤差、gap error、close-return error、range error，以及四個 80% 邊際區間是否命中。TX 收盤價與結算價分開，不得互相代替。不得計算或評估新契約沒有產生的 Bull/Base/Bear、Brier score、intraday regime 或 breadth 命中率。

新聞與盤後事件只可作背景解釋，不得宣稱單一新聞由日 OHLC 證明因果。保留全部成功與失敗日；晚到、資料缺漏或流程失敗不得靜默刪除。提出的改善只進入後續 validation 研究，不能修改本日 snapshot，也不能重新調整已關閉的 final test。
```

## 正式切換檢查表

- [ ] genuine live staging pre-open 與 post-close 同日通過。
- [ ] 3–5 個完整交易日 shadow，包含失敗執行與遲到判定。
- [ ] 三個任務的目前 prompt、排程、目的地與通知設定已另存 rollback 副本。
- [ ] production writer 經獨立 review，仍保留表頭指紋、空列檢查、逐欄 readback 與 append-only 限制。
- [ ] 在 staging 對以上三份 prompt 做一次完整演練，確認沒有機率、第二套點數或 08:50 cutoff。
- [ ] 明確選定 `V1 台股劇本` 是改成唯讀稽核還是停用，避免雙重預測。
- [ ] 使用者另行核准正式 Sheet 與排程切換。

## Rollback

若正式切換後任一交易日缺少準時快照、readback 不一致或外部任務違反契約，立即停止新資料流並恢復切換前保存的任務 prompt／排程／目的地。保留失敗 receipt 與已寫入列，不刪除、不覆寫，也不把恢復後的 legacy 輸出納入新模型 forward 統計。
