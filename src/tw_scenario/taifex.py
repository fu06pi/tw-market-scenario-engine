"""Official TX day/night CSV acquisition and explicit session-aware research joins.

The exchange trading date of a night row is its following regular session.
Fetch timestamps are preserved; historical publication times remain unknown.
"""

from __future__ import annotations

import calendar
import csv
import gzip
import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .snapshot import write_immutable_json
from .split_forecast import OHLC, TAIPEI

CSV_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
SESSION_RULE_URL = "https://www.taifex.com.tw/cht/4/aHIntroduction"


@dataclass(frozen=True)
class FuturesRow:
    trading_date: date
    contract: str
    session: str
    prices: OHLC | None
    volume: int | None
    source_file: str


def parse_tx_csv(raw: bytes, source_file: str) -> list[FuturesRow]:
    for encoding in ("utf-8-sig", "cp950"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("unrecognized TAIFEX CSV encoding")
    reader = csv.DictReader(io.StringIO(text))
    required = {
        "交易日期",
        "契約",
        "到期月份(週別)",
        "開盤價",
        "最高價",
        "最低價",
        "收盤價",
        "成交量",
        "交易時段",
    }
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise ValueError("TAIFEX response is not the expected CSV schema")
    result = []
    seen = set()
    for row in reader:
        if row["契約"].strip() != "TX":
            continue
        contract = row["到期月份(週別)"].strip()
        # Weekly and spread contracts are not monthly outright TX observations.
        if not re.fullmatch(r"[12]\d{3}(0[1-9]|1[0-2])", contract):
            continue
        session = {"一般": "day", "盤後": "night"}.get(row["交易時段"].strip())
        if session is None:
            raise ValueError("unknown TAIFEX trading-session label")
        day = datetime.strptime(row["交易日期"].strip(), "%Y/%m/%d").date()
        key = (day, contract, session)
        if key in seen:
            raise ValueError("duplicate date/contract/session in TAIFEX CSV")
        seen.add(key)
        volume_text = row["成交量"].strip().replace(",", "")
        volume = None if volume_text in ("", "-", "--") else int(volume_text)
        if volume is not None and volume < 0:
            raise ValueError("negative futures volume")
        price_values = [
            row[k].strip().replace(",", "") for k in ("開盤價", "最高價", "最低價", "收盤價")
        ]
        if any(v in ("", "-", "--") for v in price_values):
            prices = None
        else:
            # Never use settlement price in place of the last transaction.
            prices = OHLC.model_validate(
                dict(zip(("open", "high", "low", "close"), map(float, price_values)))
            )
        result.append(FuturesRow(day, contract, session, prices, volume, source_file))
    if not result:
        raise ValueError("TAIFEX CSV contains no monthly TX observations")
    return result


def fetch_month(month: str, directory: Path, timeout: float = 25) -> dict:
    year, number = int(month[:4]), int(month[4:])
    start = date(year, number, 1)
    end = date(year, number, calendar.monthrange(year, number)[1])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"TX-{month}.csv"
    metadata_path = directory / f"TX-{month}.metadata.json"
    if path.exists() or metadata_path.exists():
        if not (path.exists() and metadata_path.exists()):
            raise ValueError("incomplete cached download; do not silently overwrite")
        metadata = json.loads(metadata_path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("cached TAIFEX bytes changed")
        cached_rows = parse_tx_csv(path.read_bytes(), str(path))
        if (
            metadata.get("source_url") != CSV_URL
            or metadata.get("form", {}).get("queryStartDate") != start.strftime("%Y/%m/%d")
            or metadata.get("form", {}).get("queryEndDate") != end.strftime("%Y/%m/%d")
            or any(not start <= row.trading_date <= end for row in cached_rows)
            or metadata.get("monthly_tx_rows") != len(cached_rows)
        ):
            raise ValueError("cached TAIFEX date range or metadata mismatch")
        return metadata
    form = {
        "down_type": "1",
        "commodity_id": "TX",
        "queryStartDate": start.strftime("%Y/%m/%d"),
        "queryEndDate": end.strftime("%Y/%m/%d"),
    }
    request = urllib.request.Request(
        CSV_URL,
        data=urllib.parse.urlencode(form).encode(),
        headers={"User-Agent": "tw-scenario-night-study/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    rows = parse_tx_csv(raw, str(path))
    if any(not start <= row.trading_date <= end for row in rows):
        raise ValueError("TAIFEX download ignored requested date range")
    metadata = {
        "source_url": CSV_URL,
        "form": form,
        "fetched_at": datetime.now(TAIPEI).isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "monthly_tx_rows": len(rows),
        "historical_availability_verified": False,
        "available_at": None,
        "availability_basis": "Observed fetch only; historical publication unknown",
        "session_rule_url": SESSION_RULE_URL,
    }
    with path.open("xb") as stream:
        stream.write(raw)
    write_immutable_json(metadata, metadata_path)
    return metadata


def index_rows(rows: list[FuturesRow]) -> dict:
    index = {}
    for row in rows:
        key = (row.trading_date, row.contract, row.session)
        if key in index:
            raise ValueError("overlapping or duplicate monthly CSV records")
        index[key] = row
    return index


def night_feature(day: date, rows: list[FuturesRow], cutoff: datetime) -> dict:
    """Select the earliest actively quoted month using completed night rows only.

    A missing same-contract previous-day last trade is a failure, never a reason
    to substitute settlement or another contract. Session-end is observed-time
    semantics, not an invented historical publication timestamp.
    """
    if (
        cutoff.tzinfo is None
        or cutoff.astimezone(TAIPEI).date() != day
        or cutoff.astimezone(TAIPEI).time() > time(8, 30)
    ):
        raise ValueError("night feature requires aware same-trading-day cutoff")
    indexed = index_rows(rows)
    prior_dates = sorted(
        {r.trading_date for r in rows if r.session == "day" and r.trading_date < day}
    )
    if not prior_dates:
        raise ValueError("missing previous regular trading date")
    previous = prior_dates[-1]
    candidates = sorted(
        (
            r
            for r in rows
            if r.trading_date == day
            and r.session == "night"
            and r.prices is not None
            and r.volume is not None
            and r.volume > 0
        ),
        key=lambda r: r.contract,
    )
    if not candidates:
        raise ValueError("missing completed traded night observation")
    night = candidates[0]
    reference = indexed.get((previous, night.contract, "day"))
    if reference is None or reference.prices is None or not reference.volume:
        raise ValueError("missing same-contract previous daytime last trade")
    # Friday's night ends Saturday even though its exchange trading date is Monday.
    session_start = datetime.combine(previous, time(15), tzinfo=TAIPEI)
    session_end = datetime.combine(previous + timedelta(days=1), time(5), tzinfo=TAIPEI)
    if session_end > cutoff:
        raise ValueError("night session has not finished at cutoff")
    return {
        "trading_date": day.isoformat(),
        "contract": night.contract,
        "reference_date": previous.isoformat(),
        "reference_contract": reference.contract,
        "previous_day_close": reference.prices.close,
        "night_close": night.prices.close,
        "night_return_bps": (night.prices.close / reference.prices.close - 1) * 10000,
        "night_volume": night.volume,
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
        "age_hours": (cutoff - session_end).total_seconds() / 3600,
        "session_ended_by_cutoff": True,
        "historical_availability_verified": False,
        "available_at": None,
        "source_files": [reference.source_file, night.source_file],
    }
