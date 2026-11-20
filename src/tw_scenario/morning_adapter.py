"""Deterministic export to the observed 16-column V0 interface, without live writes.

Only the frozen default baseline is rendered. News/background cannot change its
numbers. Separate post-close patches preserve A:F and O byte-for-byte.
"""

from __future__ import annotations

import math

from .metrics import realized_direction
from .policy import ScoringPolicy
from .shadow import seal
from .snapshot import verify_document
from .split_forecast import OHLC, TAIPEI, Components, ObservedBar

HEADERS = (
    "日期", "盤前預測項目", "盤前原始假設", "預測方向", "信心度", "盤前證據/原因",
    "實際開盤", "實際盤中", "實際收盤", "驗證結果", "錯誤類型", "實際主因",
    "需要修正的假設", "隔日改進規則", "晨報來源/備註", "覆盤時間",
)
IMMUTABLE = (0, 1, 2, 3, 4, 5, 14)
REVIEW = (*range(6, 14), 15)
LABELS = {"up": "漲", "flat": "平", "down": "跌"}


def validate_forecast(forecast):
    if forecast.get("schema_version") != "prospective-shadow-v1" or not verify_document(forecast).valid:
        raise ValueError("adapter requires an intact prospective shadow snapshot")
    policy = ScoringPolicy.model_validate(forecast["scoring_policy"])
    for market, item in forecast["markets"].items():
        if market not in ("TAIEX", "TX_selected_monthly") or item["inputs"]["market"] != market:
            raise ValueError("unsupported/mismatched forecast market")
        if item["inputs"]["date"] != forecast["trading_date"]:
            raise ValueError("market date differs from forecast date")
        if market == "TX_selected_monthly" and not item["inputs"].get("contract"):
            raise ValueError("TX narrative requires an explicit contract")
        chosen = item["models"][item["default_baseline"]]
        prices = OHLC.model_validate(chosen["prices"])
        components = Components.model_validate(chosen["components"])
        if components.reconstruct(item["inputs"]["previous_close"]) != prices:
            raise ValueError("narrative prices disagree with frozen components")
        for field in OHLC.model_fields:
            bounds = chosen["marginal_80"][field]
            lo, hi = bounds["lower"], bounds["upper"]
            if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo <= hi:
                raise ValueError("invalid frozen narrative intervals")
    return policy


def forecast_id(forecast):
    return "shadow-" + forecast["trading_date"] + "-" + forecast["snapshot_hash"][:16]


def morning_export(forecast, headers):
    if tuple(headers) != HEADERS:
        raise ValueError("observed V0 headers changed; mapping must be reviewed")
    policy = validate_forecast(forecast)
    records = []
    for market, item in forecast["markets"].items():
        row = [""] * 16
        inputs, method = item["inputs"], item["default_baseline"]
        numerical = item["models"][method]
        prices, intervals = numerical["prices"], numerical["marginal_80"]
        reference = inputs["previous_close"]
        directions = {k: LABELS[realized_direction((prices[k] / reference - 1) * 10000,
                                                  policy.flat_threshold_bps).value]
                      for k in ("open", "close")}
        label = "TAIEX" if market == "TAIEX" else "TX" + inputs["contract"]
        row[0], row[1] = forecast["trading_date"], label + " 日盤 OHLC（影子）"
        names = {"open": "開盤", "high": "最高", "low": "最低", "close": "收盤"}
        row[2] = "；".join(f"{names[k]} {prices[k]}" for k in names) + "。名目80%邊際區間：" + "；".join(
            f"{names[k]} [{intervals[k]['lower']}, {intervals[k]['upper']}]" for k in names)
        row[3] = f"開盤{directions['open']}；收盤{directions['close']}（中性門檻{policy.flat_threshold_bps}bp）"
        row[4] = "未估計方向機率；80%為各價位名目邊際區間，前向覆蓋尚未驗證"
        row[5] = (f"已完成TX夜盤報酬 {inputs['night_return_bps']}bp；"
                  f"{inputs['feature_latest_date']} 日盤前收 {reference}；"
                  f"資料截止 {forecast['data_cutoff_at']}；固定模型 {method}。"
                  "日OHLC不提供高低點先後、盤中型態或市場廣度。")
        row[14] = (f"forecast_id={forecast_id(forecast)}; snapshot_hash={forecast['snapshot_hash']}; "
                   f"checkpoint_hash={forecast['checkpoint_hash']}; method={method}; shadow_only")
        records.append({"market": market, "contract": inputs.get("contract"),
                        "method": method, "values": row, "frozen_numerical": numerical})
    return seal({"schema_version": "v0-shadow-export-v1", "forecast_id": forecast_id(forecast),
                 "forecast_hash": forecast["snapshot_hash"], "headers": list(HEADERS),
                 "records": records, "live_sheet_write": False})


def verify_readback(export, observed_rows, reviewed=False):
    if not verify_document(export).valid or len(observed_rows) != len(export["records"]):
        raise ValueError("invalid export or row count changed")
    for expected, observed in zip(export["records"], observed_rows):
        if len(observed) != 16:
            raise ValueError("readback must contain all 16 columns")
        indices = IMMUTABLE if reviewed else range(16)
        if any(observed[i] != expected["values"][i] for i in indices):
            raise ValueError("readback changed frozen morning cells")
    return True


def close_export(forecast, morning, observed_rows, actuals, evaluated_at):
    validate_forecast(forecast)
    if morning["forecast_hash"] != forecast["snapshot_hash"]:
        raise ValueError("morning export belongs to another numerical snapshot")
    if morning != morning_export(forecast, morning["headers"]):
        raise ValueError("morning narrative differs from frozen numerical snapshot")
    verify_readback(morning, observed_rows)
    if evaluated_at.tzinfo is None:
        raise ValueError("aware review timestamp required")
    actual_index = {}
    for raw in actuals:
        actual = ObservedBar.model_validate(raw)
        identity = (actual.trading_date.isoformat(), actual.market, actual.contract)
        if identity in actual_index:
            raise ValueError("duplicate actual identity")
        if actual.available_at > evaluated_at:
            raise ValueError("actual was unavailable at review time")
        actual_index[identity] = actual
    records = []
    policy = ScoringPolicy.model_validate(forecast["scoring_policy"])
    for entry in morning["records"]:
        market, contract = entry["market"], entry["contract"]
        actual = actual_index.get((forecast["trading_date"],
                                   "TX" if market == "TX_selected_monthly" else market, contract))
        if actual is None:
            raise ValueError("exact date/market/contract actual is missing")
        data = forecast["markets"][market]
        reference = data["inputs"]["previous_close"]
        prediction = data["models"][entry["method"]]["prices"]
        prices = {k: getattr(actual, k) for k in OHLC.model_fields}
        bp = {k: (prices[k] / reference - 1) * 10000 for k in ("open", "close")}
        hit = {k: realized_direction(bp[k], policy.flat_threshold_bps) == realized_direction(
            (prediction[k] / reference - 1) * 10000, policy.flat_threshold_bps)
            for k in ("open", "close")}
        patch = {
            6: f"{actual.open}（相對前收 {bp['open']}bp）",
            7: f"最高 {actual.high}；最低 {actual.low}；高低點先後與盤中型態未驗證",
            8: f"{actual.close}（相對前收 {bp['close']}bp）",
            9: "；".join(f"{k}方向{'正確' if hit[k] else '錯誤'}" for k in hit) + "；" +
               "；".join(f"{k}絕對誤差 {abs(prices[k] - prediction[k])}點" for k in prices),
            10: "、".join(f"{k}方向錯誤" for k in hit if not hit[k]) or "方向一致，幅度誤差另列",
            11: "未執行因果歸因；官方日OHLC僅驗證價格，不能據此認定消息主因",
            12: "保留原假設；以後續共同樣本比較候選與鎖定基準",
            13: "依固定版本持續影子驗證，不以當日結果臨時調參",
            15: evaluated_at.astimezone(TAIPEI).isoformat(),
        }
        records.append({"market": market, "contract": contract,
                        "expected_immutable": {str(i): entry["values"][i] for i in IMMUTABLE},
                        "patch_by_zero_based_column": {str(k): v for k, v in patch.items()}})
    return seal({"schema_version": "v0-shadow-review-patch-v1", "forecast_id": morning["forecast_id"],
                 "forecast_hash": forecast["snapshot_hash"], "records": records,
                 "allowed_columns": ["G:N", "P"], "live_sheet_write": False})


def verify_review_readback(patch, observed_rows):
    if not verify_document(patch).valid or len(observed_rows) != len(patch["records"]):
        raise ValueError("invalid review patch or row count changed")
    for entry, row in zip(patch["records"], observed_rows):
        if len(row) != 16:
            raise ValueError("review readback requires all 16 cells")
        expected = entry["expected_immutable"] | entry["patch_by_zero_based_column"]
        if any(row[int(i)] != value for i, value in expected.items()):
            raise ValueError("review readback differs or morning cells were changed")
    return True
