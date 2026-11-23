# 盤前影子流程與模型優化第一輪

2026-11-04，時區 Asia/Taipei。**五項工程已實作，真實前向樣本仍為 0；沒有模型通過正式驗收。** 保留簡單夜盤基準為兩市場的預設研究模型，其他模型只做影子比較。

## 依序完成的內容

1. **盤前影子流程**：官方 TWSE 月份指數 JSON、TAIFEX 指定日期日／夜盤 HTML adapters。每次 HTTP 收到的實際時間、來源、原始 bytes 指紋分開保存；輸入只取前兩個已完成日盤與當次已完成夜盤。TX 在夜盤有成交的月份中選最早月，前收及 lag 都用該合約。檢查回傳日期、時段、最後成交價、缺漏與過期。標準時段不當成歷史發布時間證明。
2. **夜盤基準殘差**：夜盤直推開盤＋過去平均 body 作基準，只擬合其收盤 log-return 殘差。訓練各列的基準平均只取該列以前的資料，前 20 列不擬合殘差。
3. **盤中夜盤特徵**：`body_control` 與 `body_night` 使用相同開盤頭；差別只有 body 頭是否增加夜盤報酬。不使用當日開盤實際值。
4. **TAIEX 開盤校準**：用過去最多 60 筆逐日前推的 out-of-fold 開盤預測擬合線性幅度校準，至少 30 筆。調整 g 時反向調整 b，使收盤點估保持一致。TX 不套用現貨校準器。
5. **區間與監控**：獨立校準區塊提供 80% 邊際絕對誤差區間，保存 finite-sample rank、寬度及 interval score。監控包含嘗試／失敗、缺 actual、各 OHLC 點／bp 誤差、方向混淆矩陣、共同日期的時間區塊不確定性與區間覆蓋。少於 30 個前向觀測不判定覆蓋達標，沒有自動切換模型。

## 本輪驗證結果

先保存 [plan.json](plan.json)，再用 2025-10-01～2025-11-18 的 **32 個共同驗證日**比較 5 個價格基準與 4 個變體。2025-11-19～12-31 的另外 **30 日**只估計區間半徑，不參與本輪選模。這些歷史區間過去已使用，仍屬回溯驗證，不與之前 179 日結果混算。

| 方法 | TAIEX 開盤 MAE（bp） | TAIEX 收盤 MAE（bp） | TX 開盤 MAE（bp） | TX 收盤 MAE（bp） |
|---|---:|---:|---:|---:|
| 簡單夜盤基準（鎖定） | 45.9274 | 64.4171 | 42.6393 | 75.3448 |
| 夜盤＋收盤殘差 | 45.9274 | 64.1892 | 42.6393 | 75.8250 |
| 回歸開盤＋原盤中特徵 | 41.9679 | 68.6420 | 45.7165 | 78.4865 |
| 回歸開盤＋盤中夜盤特徵 | 41.9679 | 67.9207 | 45.7165 | 79.3943 |
| 再加 TAIEX 開盤校準 | 41.9110 | 67.9207 | 45.7165 | 79.3943 |

- 殘差版相對夜盤基準：TAIEX 收盤改善 **0.35%**，TX 惡化 **0.64%**，不足 5% 目標。
- 盤中夜盤特徵相對 control：TAIEX 收盤改善 **1.05%**，但 body 誤差增加；TX 收盤惡化 **1.16%**。不能宣稱盤中預測整體改善。
- TAIEX 開盤校準相對未校準版只改善 **0.14%**，收盤不變；TX 完全不變。
- 所有候選均未通過相對鎖定基準的驗證價格門檻，兩市場預設仍為 `night_carry`。八項誤差、方向、時間區塊區間與失敗見 [summary.json](summary.json)。

只執行這組固定參數比較，未再調已看過的 2026 評估期。CPU 約 10 秒，未購買運算。固定選擇後，依事先規則用已知至 2026-09-30 的快取 refit 為未來影子權重，沒有用 2026 成績選擇模型／區間寬度。[checkpoint.json](checkpoint.json) 只含 JSON 係數與版本，不含原始授權資料或 pickle。

## 真實執行與時間證據

下午成功解析官方現貨、指定日盤與當日夜盤，見 [adapter-smoke.json](adapter-smoke.json)。這只證明實際連線／解析可用，擷取在截止之後，不能計入盤前成果。

今天實際執行 `run_shadow`，因已過 08:30，返回 exit code 2，保存 [late-run-failure.json](late-run-failure.json)，沒有生成今天的預測。[initial-monitor.json](initial-monitor.json)：1 次嘗試、1 次失敗、0 個預測／準確率分母。

[GitHub workflow](../../../.github/workflows/preopen-shadow.yml) 已寫入 PR，設工作日 **08:15** 啟動、08:30 前收資料、08:40 前保存數字。GitHub 延遲觸發也視為失敗。排程須在預設分支存在才生效，**目前 PR 未合併，因此尚未啟用，也未實際通過 GitHub Actions 端到端驗證**。

上傳不可覆寫 artifact 後，另從 GitHub artifact API 讀回 created_at／updated_at，下載 ZIP 核對 digest、預測及本地 receipt，要求遠端完成時間也在 08:40 前。另一個 artifact 保存 API 回讀。快照自身仍標示只有本地時鐘證據，hash 正確不能自動成為正式樣本。遠端核對也只證明該 artifact 保存，還需市場／合約、來源、actual 與版本資格稽核。

監控目前產生診斷結果，不將任意匯入的 remote-receipt 檔當作可信證明，不發布正式前向準確率。artifact 保存 90 日，須在到期前匯出原 ZIP、官方 API 回讀與指紋留存。

## 重現與操作

使用 Python 3.12；runtime 固定於 [requirements-shadow.txt](../../../requirements-shadow.txt)。歷史原始資料與逐日研究輸出保持 ignored；本資料夾只保留彙總、計畫、權重及流程證據。

```bash
python -m pip install -r requirements-shadow.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m scripts.prepare_shadow_candidates --output reports/generated/prospective-night-new-run
```

研究重跑必須用新目錄；不覆寫本次 checkpoint／預測。盤前執行沒有 `--date` 或 `--now` 回填選項：

```bash
python -m scripts.run_shadow \
  --checkpoint reports/research/prospective-night-v1/checkpoint.json \
  --output reports/generated/shadow/new-run
```

收盤後以查核過的 `ObservedBar` JSON list 提供 actual；TX 的日期、合約及日盤必須符合預測。取得時間採實際擷取時間，不能反推成盤前可用：

```bash
python -m scripts.monitor_shadow \
  --runs reports/generated/shadow \
  --actuals path/to/verified-daytime-actuals.json \
  --output reports/generated/new-shadow-monitor.json
```

通過 **128 項 pytest**、Ruff、依賴一致性及 workflow YAML 解析檢查。新增測試使用合成時鐘／行情，涵蓋當日與未來標籤隔離、盤中對照、校準後收盤不變、換月前收報酬、遲到拒絕、遠端指紋／截止、缺 actual 不填準確率，以及整筆失敗不混入部分評分。

**待取得證據**：真正盤前來源及遠端 artifact、未見日盤 actual、前向價格比較及區間覆蓋。歷史發布時點未核實，校準樣本較舊，80% 尚非實證覆蓋率。獨立影子流程尚未接成原晨報／Sheet 正式數值來源；既有排程、歷史預測及評估沒有修改。
