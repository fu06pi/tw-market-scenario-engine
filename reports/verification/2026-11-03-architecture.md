# 前兩項架構修正：推送前驗證

範圍：評分／樣本資格一致性，以及盤前 gap／session body 獨立建模。未切換每日排程、晨報或 Google Sheet，未改動歷史預測、實際結果、manifest 與評估。

## 可驗證的改善

| 檢查 | 修正前 | 修正後 | 解讀 |
|---|---|---|---|
| 同批 21 份快照 hash 驗證 | 舊正規化方法通過 11 份 | 相容方法通過 19 份 | 8 份可依原始 JSON 驗證；不重寫 hash，不代表預測更準 |
| 仍未通過 hash 的紀錄 | — | 2026-10-01、2026-11-03 | 持續排除，不將失敗改成通過 |
| 樣本納入 | manifest／evaluation 的 True／False 有矛盾 | 共用 gate、False 優先、衝突明列 | 最新 21 份中 6 份符合保守 legacy 條件，各版本分開彙總 |
| 實際結果無效時的樣本分母 | 舊骨架沒有統一的版本分母 | 可辨識的預測仍計入該模型 observed／excluded 數 | 新增回歸測試，避免失敗樣本從分母消失 |
| 開盤與盤中預測 | 固定夜盤方向基準，缺獨立盤中預測流程 | 各自標準化與擬合 Ridge，分別預測 g／b | 合成測試驗證開高收低，以及只改 gap 特徵不改變 body 預測 |
| 防洩漏與隔離 | 舊 schema 缺少完整的可用時間／合約契約 | 新 schema 驗證截止、前收來源、標籤時間及同合約參考 | 以失敗案例驗證拒絕，並非宣稱原有資料均已滿足新契約 |
| 自動測試 | 原有 4 項通過 | 55 項通過 | 新增 51 項檢查；不是市場績效證據 |

## 已執行檢查與重現

於 Python 3.12 的專案環境執行：

```bash
python -m pytest -q
python -m ruff check src tests examples/split_demo.py
git diff --check
python -m tw_scenario.cli audit . reports/generated/audit-before-push.json
```

測試結果：55 passed；Ruff 與 diff whitespace 檢查通過。合成資料 forecast／freeze／verify／evaluate 另依 [架構文件](../../docs/architecture/AUDIT_AND_SPLIT_V2.md) 的指令驗證。

hash 比較使用同一批檔案、同一執行環境與原檔不變的條件，可重現為：

```python
import json
from pathlib import Path
from tw_scenario.models import PredictionSnapshot
from tw_scenario.snapshot import verify_document, verify_snapshot

payloads = [json.loads(p.read_text()) for p in sorted(Path("predictions").glob("20*.json"))]
old_pass = sum(verify_snapshot(PredictionSnapshot.model_validate(p)) for p in payloads)
new_pass = sum(verify_document(p).valid for p in payloads)
print(len(payloads), old_pass, new_pass)  # 此批資料：21, 11, 19
```

[最新 ledger](../audits/2026-11-03-eligibility-v2-latest.json) 保存資料來源 commit `25c6952b3340cc256ba16472372161eed659d09d`、實作 SHA-256、套件版本、逐筆排除原因及分版本結果。Git 時間只作儲存庫證據，不能取代第三方公開發布證明。資料來源的行情內容未在本次重新向交易所驗證。

## 尚不能證明的改善

目前只有 20 份已保存的 legacy 實際結果，且缺少符合新版 schema 的完整歷史特徵／可用時間資料；新 forecaster 預設至少 60 筆允許的訓練樣本。本次沒有把 legacy 敘事或事後資料轉成假造的歷史盤前特徵。

因此，新模型的真實方向準確率、MAE 改善、區間校準、相對最強基準的增益均尚未量測。合成資料用於確認架構能運作及邊界會拒絕無效輸入，不能當成金融模型優勢。高低點也仍是歷史影線平均基準。

10／27 實際結果仍缺。後續須先補齊可信來源與足量歷史資料，再固定時間切分做樣本外比較；原有市場績效驗收門檻保持不變。
