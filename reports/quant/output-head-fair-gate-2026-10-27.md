# Output-head 公平 baseline gate（2026-10-27）

## 結論

Recent JOINT TimesFM 3 output-head 候選在公平 comparator 下為 `FAIL`。原始 gate 先以區間寬度排除部分點預測較強的 night regression，再拿候選和較弱的 range baseline 比較，因此原本的正改善不能作晉級證據。

本報告固定 `night_regression` 為點預測 comparator；只有模型當日失敗才以同日期 `range` 回退。候選與 comparator 都使用60日、至少40筆已公布殘差的相同因果校準，保留全部240個預測日。

| 市場 | 候選 Close MAE | comparator Close MAE | 公平改善 | 三折改善 | CI95 | gate |
|---|---:|---:|---:|---|---|---|
| TAIEX | 407.22 | 358.12 | -13.71% | -12.97% / -17.21% / -10.87% | [-21.65%, -9.09%] | `FAIL` |
| TX | 404.50 | 381.87 | -5.93% | -0.04% / -4.91% / -9.20% | [-12.24%, -0.06%] | `FAIL` |

TAIEX在2026-06-08的night regression產生非正Low，依固定規則回退range；該日仍保留。TX沒有使用fallback。

## 完整門檻

- **TAIEX**：通過 `ohl_noninferiority, forecast_coverage`；失敗 `close_buffer, fold_majority, bootstrap_lower_nonnegative, interval_coverage, interval_score_noninferior`。
  候選O/H/L/C MAE：98.22 / 275.72 / 299.74 / 407.22點；comparator：135.79 / 264.37 / 306.84 / 358.12點。
- **TX**：通過 `forecast_coverage`；失敗 `close_buffer, fold_majority, bootstrap_lower_nonnegative, ohl_noninferiority, interval_coverage, interval_score_noninferior`。
  候選O/H/L/C MAE：184.53 / 297.82 / 328.83 / 404.50點；comparator：157.10 / 317.07 / 325.77 / 381.87點。

## 停止規則與狀態

Phase-2帳本共扣除3052.35/7200秒，尚餘4147.65秒。wick、recent、JOINT三條路線均未對公平comparator產生真增益，已觸發事前固定的連續三輪無增益停止規則；剩餘額度不再用於盲目搜尋。

這是截至2026-10-08資料的development回溯結果，未開啟新的future holdout，不能視為final通過或獲利證據。正式預設維持最強baseline，`production_switch=false`。

## 重現

```bash
.venv/bin/tw-scenario quant timesfm-head-fair-gate \
  data/processed/cce0208a6ee26a9874f1836c5aa24f374ee4cf74880c2d39afb5748e366a50ce \
  reports/generated/quant/model-search-v2/timesfm3-output-head-walk-forward/75d81bef117162e5a51a \
  reports/generated/quant/model-search-v2/fair-gate-75d81bef117162e5a51a-replay \
  --training-config config/quant/timesfm_output_head_recent.json \
  --comparator-config config/quant/head_fair_comparator.json \
  --calibration-config config/quant/head_interval_recalibration_60.json
```

完整本機結果SHA256：`65b22ffe4bc30bc7724e47997fc810b687a888702cc969d380b5049a1fd8790e`。逐日預測留在Git ignored generated目錄，避免重新散布FinMind原始資料；repo只保存aggregate。
