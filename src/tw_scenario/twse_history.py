"""Resumable official TWSE TAIEX monthly OHLC archive (no credentials required)."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Taipei")
SOURCE_PAGE = "https://www.twse.com.tw/zh/indices/taiex/mi-5min-hist.html"
ENDPOINT = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
EARLIEST = date(1999, 1, 5)
FIELDS = ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"]
CSV_FIELDS = ["date", "open", "high", "low", "close"]


class RateLimited(RuntimeError):
    pass


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def iter_months(start: date, end: date):
    cursor = start.replace(day=1)
    while cursor <= end:
        yield cursor
        cursor = date(cursor.year + int(cursor.month == 12), cursor.month % 12 + 1, 1)


def roc_date(value: str) -> date:
    parts = value.strip().split("/")
    if len(parts) != 3:
        raise ValueError("unexpected ROC date format")
    year, month, day = map(int, parts)
    return date(year + 1911, month, day)


def parse_month(
    payload: dict, month: date, as_of: date, earliest: date = EARLIEST
) -> tuple[list[dict], list[str]]:
    if payload.get("stat") != "OK":
        raise ValueError("TWSE response status: " + str(payload.get("stat", "missing")))
    if payload.get("fields") != FIELDS:
        raise ValueError("TWSE OHLC schema changed")
    records = []
    excluded = []
    seen = set()
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("empty month is not a complete archive")
    for row in data:
        if len(row) != 5:
            raise ValueError("unexpected number of OHLC fields")
        day = roc_date(row[0])
        if (day.year, day.month) != (month.year, month.month):
            raise ValueError("response belongs to another month")
        if day < earliest:
            raise ValueError("response predates documented source coverage")
        if day in seen:
            raise ValueError("duplicate source date")
        seen.add(day)
        try:
            prices = [Decimal(str(v).strip().replace(",", "")) for v in row[1:]]
        except InvalidOperation as error:
            raise ValueError("missing/nonnumeric OHLC") from error
        if any(not v.is_finite() or v <= 0 for v in prices):
            raise ValueError("nonfinite/nonpositive OHLC")
        o, h, l, c = prices
        if l > min(o, c) or h < max(o, c):
            raise ValueError("incoherent OHLC")
        if day > as_of:
            excluded.append(day.isoformat())
            continue
        records.append(
            dict(zip(CSV_FIELDS, [day.isoformat(), *[format(v, "f") for v in prices]], strict=True))
        )
    if records != sorted(records, key=lambda r: r["date"]):
        raise ValueError("unsorted source dates")
    return records, excluded


def retry_seconds(value: str | None, now: datetime) -> float:
    if value is None:
        return 15.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - now).total_seconds())
        except (TypeError, ValueError):
            return 15.0


def download(month: date, as_of: date, attempts: int = 3):
    query_date = max(month, EARLIEST)  # Jan 1-4 1999 are outside this source's supported range.
    url = ENDPOINT + "?" + urlencode({"date": query_date.strftime("%Y%m%d"), "response": "json"})
    for attempt in range(attempts):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "tw-market-scenario-engine historical research (monthly, cached)",
                    "Accept": "application/json",
                },
            )
            with urlopen(request, timeout=30) as response:
                body = response.read()
                payload = json.loads(body)
                parse_month(payload, month, as_of)
                return body, {
                    "source_url": url,
                    "retrieved_at": datetime.now(TZ).isoformat(),
                    "as_of_date": as_of.isoformat(),
                    "response_sha256": sha256(body),
                    "http_status": response.status,
                }
        except HTTPError as error:
            if error.code in (403, 429):
                seconds = retry_seconds(error.headers.get("Retry-After"), datetime.now(TZ))
                if error.code == 403 or seconds > 30 or attempt + 1 >= attempts:
                    raise RateLimited(
                        f"HTTP {error.code}; Retry-After {error.headers.get('Retry-After', 'unspecified')}"
                    ) from None
                time.sleep(max(seconds, 5.0))
            elif error.code >= 500 and attempt + 1 < attempts:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
        except (URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 >= attempts:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("download attempts exhausted")


def needs_refresh(meta: dict, month: date, as_of: date) -> bool:
    end = date(month.year, month.month, calendar.monthrange(month.year, month.month)[1])
    previous = date.fromisoformat(meta["as_of_date"])
    return min(as_of, end) > min(previous, end)


def load_or_fetch(folder: Path, month: date, as_of: date, interval: float):
    path = folder / "monthly" / f"{month:%Y-%m}.json"
    metadata = path.with_suffix(".metadata.json")
    cached = None
    if path.exists() or metadata.exists():
        if not path.exists() or not metadata.exists():
            raise ValueError("incomplete monthly cache pair")
        body = path.read_bytes()
        meta = json.loads(metadata.read_text())
        if sha256(body) != meta["response_sha256"]:
            raise ValueError("monthly cache checksum mismatch")
        parse_month(json.loads(body), month, as_of)
        if not needs_refresh(meta, month, as_of):
            return body, meta, False
        cached = (body, meta)
    time.sleep(interval)
    body, meta = download(month, as_of)
    if cached is not None:
        prior, prior_meta = cached
        revision = folder / "revisions" / f"{month:%Y-%m}" / prior_meta["response_sha256"]
        atomic(revision.with_suffix(".json"), prior)
        atomic(revision.with_suffix(".metadata.json"), json_bytes(prior_meta))
    atomic(path, body)
    atomic(metadata, json_bytes(meta))
    return body, meta, True


def write_csv(path: Path, records: list[dict]):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    atomic(path, stream.getvalue().encode("utf-8"))


def archive(folder: Path, as_of: date, start: date = EARLIEST, interval: float = 2.0) -> dict:
    if as_of < EARLIEST or start < EARLIEST.replace(day=1) or start > as_of:
        raise ValueError("invalid supported source interval")
    expected = list(iter_months(start, as_of))
    months = []
    records = []
    failures = []
    excluded = []
    for month in expected:
        try:
            body, meta, downloaded = load_or_fetch(folder, month, as_of, interval)
            rows, ignored = parse_month(json.loads(body), month, as_of)
            records.extend(rows)
            excluded.extend(ignored)
            months.append(
                {
                    "month": month.strftime("%Y-%m"),
                    "rows": len(rows),
                    "first_date": rows[0]["date"] if rows else None,
                    "last_date": rows[-1]["date"] if rows else None,
                    **meta,
                }
            )
            if downloaded or len(months) % 12 == 0:
                print(
                    json.dumps(
                        {
                            "month": month.strftime("%Y-%m"),
                            "rows": len(rows),
                            "months_done": len(months),
                            "months_total": len(expected),
                            "downloaded": downloaded,
                        }
                    ),
                    flush=True,
                )
        except (
            RateLimited,
            ValueError,
            URLError,
            TimeoutError,
            HTTPError,
            json.JSONDecodeError,
        ) as error:
            failures.append(
                {
                    "month": month.strftime("%Y-%m"),
                    "error_type": type(error).__name__,
                    "detail": str(error)[:250],
                }
            )
            # Stop on access/rate limits, schema changes, or corruption. Never hide a hole.
            print(
                json.dumps(
                    {"stopped_at": month.strftime("%Y-%m"), "error_type": type(error).__name__}
                ),
                flush=True,
            )
            break
    records.sort(key=lambda r: r["date"])
    if len({r["date"] for r in records}) != len(records):
        raise ValueError("duplicate dates across months")
    if records:
        write_csv(folder / "daily.csv", records)
        for year in sorted({r["date"][:4] for r in records}):
            write_csv(
                folder / "yearly" / f"{year}.csv",
                [r for r in records if r["date"].startswith(year)],
            )
    manifest = {
        "schema_version": 1,
        "source": "Taiwan Stock Exchange (TWSE) official TAIEX historical OHLC",
        "source_page": SOURCE_PAGE,
        "endpoint": ENDPOINT,
        "documented_earliest_date": str(EARLIEST),
        "as_of_date": str(as_of),
        "requested_start_month": expected[0].strftime("%Y-%m"),
        "requested_end_month": expected[-1].strftime("%Y-%m"),
        "expected_months": len(expected),
        "completed_months": len(months),
        "missing_months": [m.strftime("%Y-%m") for m in expected[len(months) :]],
        "status": "complete_requested_months" if len(months) == len(expected) else "partial",
        "row_count": len(records),
        "first_date": records[0]["date"] if records else None,
        "last_date": records[-1]["date"] if records else None,
        "fields": CSV_FIELDS,
        "price_unit": "index_points",
        "volume_available": False,
        "timezone": "Asia/Taipei",
        "excluded_after_cutoff_dates": excluded,
        "quality": {
            "duplicates": 0,
            "invalid_ohlc": 0,
            "nonfinite_prices": 0,
            "independent_trading_calendar_verified": False,
            "calendar_note": "All rows and all requested months retained; source-omitted trading days not independently audited. Historical Saturday sessions are retained.",
            "historical_publication_time_verified": False,
        },
        "failures": failures,
        "months": months,
    }
    if records:
        manifest["daily_csv_sha256"] = sha256((folder / "daily.csv").read_bytes())
    previous = folder / "manifest.json"
    atomic(previous, json_bytes(manifest))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "status",
                    "completed_months",
                    "expected_months",
                    "row_count",
                    "first_date",
                    "last_date",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def main():
    root = Path(__file__).resolve().parents[2]
    now = datetime.now(TZ)
    # Conservative full-day boundary; no live-day OHLC is included by default.
    cutoff = now.date() - timedelta(days=1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "data/public/twse/taiex")
    parser.add_argument("--as-of", type=date.fromisoformat, default=cutoff)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.as_of > cutoff:
        parser.error("only completed prior calendar days allowed; current-day bars excluded")
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("minimum interval is one second")
    result = archive(args.output, args.as_of, interval=args.interval)
    raise SystemExit(0 if result["status"] == "complete_requested_months" else 2)


if __name__ == "__main__":
    main()
