# 有限期盤前新聞研究試跑

此為使用者要求推進新聞比較與真實使用流程的獨立學術試跑，不改原V0／V1任務、正式Sheet或量化checkpoint。

## 固定協議

- `config/forward_news_trial_v1.json`：2026-11-17～11-30，共10個工作日，Asia/Taipei；08:15盤前、15:30收盤。
- A沿用驗證鎖定的night_carry，不依這次TimesFM已見結果重選。數值checkpoint與新聞指令以SHA-256固定。
- B只另存收盤方向；最多5則、最近72小時、可核實發布／取得時間的新聞。無合格來源就保留B缺漏，不虛構或從下午倒填。
- 原數字與B決策皆須08:40前完成遠端保存；新聞與量化來源須符合08:30截止。延遲執行列失敗。
- AI精確後端revision若未公開要如實標示，不把模型家族／固定prompt冒充完全固定後端。
- 10日是流程可行性與描述性試跑，不足保證增益；每市場30筆前不產生穩健改善區間。

## 免等待主PR合併的研究保存

專用 [Issue #4](https://github.com/fu06pi/tw-market-scenario-engine/issues/4) 作研究ledger。每份完整盤前payload及收盤結果以**新的獨立留言**保存，不改寫舊留言。原始完整授權資料、模型衍生權重與逐字稿不放入留言。

`forward_trial.py` 用GitHub API回讀留言的created_at、updated_at、作者、issue_url及精確payload：

1. 以程式／協議／checkpoint固定版本核對，payload本身也有指紋。
2. 盤前留言伺服器建立與修改時間都須早於當日08:40；若事後修改則拒絕。
3. 本機時間與hash單獨不足以證明當時已保存；GitHub時鐘只證明內容保存，不證明模型準確或發布者歷史時間已由第三方驗證。
4. 收盤只讀今天對應的有效盤前留言，找不到就保留缺漏，不能回退昨天。
5. 累計按固定協議／模型／prompt／政策分組；多個盤前留言拒絕整日；採第一個完整官方收盤收據，其他失敗保留。不得事後挑較好的預測。

盤前數字用既有`run_shadow_with_report`；只新增B的決策和研究payload，不改量化價格。收盤用`forward_trial_cli close`取得官方actual、評分、核對晨報覆盤的本地往返。這不是正式Google Sheet寫入證據；Sheet實際讀回仍需另完成。

## 執行

研究任務必須使用另行記錄的**精確commit**，以獨立worktree執行，保留互動中的工作目錄與舊任務。即使主PR未合併也可讀該commit；新研究以留言證據保存，不需要寫回程式分支。

```bash
python -m scripts.run_shadow_with_report \
  --checkpoint reports/research/prospective-night-v1/checkpoint.json \
  --output reports/generated/forward-new-date
python -m scripts.forward_trial_cli preopen \
  --run reports/generated/forward-new-date \
  --decisions path/to/news-taiex.json path/to/news-tx.json \
  --output reports/generated/forward-new-date/preopen.json
```

JSON與`.comment.txt`產生後，以連線GitHub工具新增ledger留言，立刻回讀實際伺服器時間與內容。不把自行填寫的metadata視為遠端證據。若缺新聞，可不傳`--decisions`，A保存但B仍為缺漏。

```bash
python -m scripts.forward_trial_cli close \
  --metadata path/to/github-comment-api-readback.json \
  --source-commit EXACT_LOCKED_COMMIT \
  --issue-number 4 \
  --output reports/generated/new-close-date
```

收盤另貼新的close留言，再用`aggregate_trial_comments`讀所有已驗證留言累計；頁面需讀完整，不忽略第二頁或失敗紀錄。

是否已建立／啟用排程、最終固定commit、伺服器回讀與首次實際執行，分別以`trial-status.json`及ledger真實記錄為準。下午的preflight留言不是盤前樣本；預約成功也不是已執行成功。

2026-11-16已由排程服務讀回確認啟用，詳見[狀態](../../reports/followup/2026-11-16/trial-status.json)。方案5個啟用任務上限，研究合併在單一任務；RRULE列10個08:15，RDATE另列10個15:30，以實際台北時鐘分流。固定執行commit `398d3b7b35f8d007178037ccbc8e06b6ee5d19a6`，原四任務不變。新排程尚未首次執行，未宣稱環境啟動、來源成功或新增有效樣本。
