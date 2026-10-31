from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from typing import ClassVar

from tw_scenario.quant import collection
from tw_scenario.quant.backtest import load_dataset
from tw_scenario.quant.contracts import TZ, digest
from tw_scenario.quant.providers import FinMindProvider
from tw_scenario.quant.quality import quality_report


class SyntheticFinMindProvider(FinMindProvider):
    calls: ClassVar[list[dict]] = []

    def __init__(self, cache):
        super().__init__(cache, token=None)

    def download(self, dataset, data_id, start, end):
        assert (end - start).days <= 366
        request = {
            "dataset": dataset,
            "start_date": str(start),
            "end_date": str(end),
        }
        if data_id is not None:
            request["data_id"] = data_id
        self.calls.append(request)
        path = self.cache / f"{digest(request)}.json"
        if path.exists():
            return path

        all_days = []
        cursor = date(2025, 9, 1)
        while cursor <= date(2026, 3, 31):
            if cursor.weekday() < 5:
                all_days.append(cursor)
            cursor += timedelta(days=1)
        requested_days = [day for day in all_days if start <= day <= end]
        if dataset == "TaiwanStockTradingDate":
            records = [{"date": str(day)} for day in requested_days]
        elif dataset == "TaiwanStockPrice":
            records = [
                {
                    "date": str(day),
                    "stock_id": "TAIEX",
                    "Trading_Volume": 1000,
                    "open": 20_000,
                    "max": 20_100,
                    "min": 19_900,
                    "close": 20_050,
                }
                for day in requested_days
                if date(2025, 12, 1) <= day <= date(2025, 12, 2)
            ]
        elif dataset == "TaiwanFuturesDaily":
            records = []
            for day in requested_days:
                if not date(2025, 12, 1) <= day <= date(2025, 12, 2):
                    continue
                for session in ("position", "after_market"):
                    records.append({
                        "date": str(day),
                        "futures_id": "TX",
                        "contract_date": "202512",
                        "open": 20_000,
                        "max": 20_100,
                        "min": 19_900,
                        "close": 20_050,
                        "volume": 100,
                        "settlement_price": 20_040 if session == "position" else 0,
                        "open_interest": 200 if session == "position" else 0,
                        "trading_session": session,
                    })
        else:  # pragma: no cover - the production contract rejects this first
            raise AssertionError(dataset)
        raw = json.dumps({"status": 200, "msg": "success", "data": records}).encode()
        cached = {
            "request": request,
            "api_status": 200,
            "api_message": "success",
            "fetched_at": datetime(2026, 1, 2, 12, tzinfo=TZ).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "records_hash": digest(records),
            "records": records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached), encoding="utf-8")
        return path


def test_finmind_collection_is_immutable_and_has_successor_roll_calendar(monkeypatch, tmp_path):
    SyntheticFinMindProvider.calls = []
    monkeypatch.setattr(collection, "FinMindProvider", SyntheticFinMindProvider)
    start = date(2025, 12, 1)
    end = date(2025, 12, 2)

    output = collection.collect_finmind(start, end, tmp_path)
    repeated = collection.collect_finmind(start, end, tmp_path)

    assert repeated == output
    data, dataset_id = load_dataset(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert dataset_id == output.name == manifest["dataset_id"]
    assert manifest["rows"] == 6
    assert manifest["failures"] == []
    assert manifest["calendar_coverage"]["end"] > str(end)
    assert data.calendar.expiries["202512"] == date(2025, 12, 17)
    assert data.calendar.expiries["202601"] == date(2026, 1, 21)
    assert data.calendar.contract_for(start) == "202512"

    report = quality_report(data.bars, data.calendar, start, end, manifest)
    assert report["status"] == "PASS_RETROSPECTIVE"
    assert report["missing"] == []
    assert report["expected_rows"] == 6
    assert all("token" not in request for request in manifest["source_requests"])


def test_finmind_collection_rejects_current_or_future_dates(tmp_path):
    today = datetime.now(TZ).date()
    try:
        collection.collect_finmind(today, today, tmp_path)
    except ValueError as exc:
        assert "complete prior dates" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("current-day collection must fail closed")
