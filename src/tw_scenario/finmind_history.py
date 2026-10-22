"""Separate, resumable FinMind TAIEX archive and official OpenAPI cross-check."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from .twse_history import FIELDS, TZ, atomic, json_bytes, parse_month, sha256, write_csv

API = "https://api.finmindtrade.com/api/v4/data"
OFFICIAL = "https://openapi.twse.com.tw/v1/indicesReport/MI_5MINS_HIST"


def normalize(rows, year, cutoff):
    result = []
    seen = set()
    for r in rows:
        d = date.fromisoformat(r["date"])
        if d.year != year or d > cutoff or r["stock_id"] != "TAIEX":
            raise ValueError("unexpected date or instrument")
        if d in seen:
            raise ValueError("duplicate date")
        seen.add(d)
        roc = f"{d.year - 1911}/{d.month:02}/{d.day:02}"
        parsed, _ = parse_month(
            {
                "stat": "OK",
                "fields": FIELDS,
                "data": [[roc, r["open"], r["max"], r["min"], r["close"]]],
            },
            d.replace(day=1),
            cutoff,
            earliest=date.min,
        )
        result.extend(parsed)
    return sorted(result, key=lambda r: r["date"])


def cached(path, url, cutoff):
    meta_path = path.with_suffix(".metadata.json")
    if path.exists() and meta_path.exists():
        raw = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if sha256(raw) != meta["sha256"]:
            raise ValueError("cache checksum mismatch")
        if meta["cutoff"] == cutoff.isoformat():
            return json.loads(raw)
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response:
                raw = response.read()
            break
        except HTTPError as error:
            if error.code < 500 or attempt == 2:
                raise
            time.sleep(2**attempt)
        except TimeoutError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    payload = json.loads(raw)
    if isinstance(payload, dict) and payload.get("status") != 200:
        raise ValueError("API unsuccessful status")
    atomic(path, raw)
    atomic(
        meta_path,
        json_bytes(
            {
                "source_url": url,
                "retrieved_at": datetime.now(TZ).isoformat(),
                "cutoff": cutoff.isoformat(),
                "sha256": sha256(raw),
                "historical_publication_time_verified": False,
            }
        ),
    )
    return payload


def archive(root, cutoff, start=1990, offline=False):
    folder = root / "finmind/taiex"
    all_rows, outcomes = [], []
    for year in range(start, cutoff.year + 1):
        end = min(date(year, 12, 31), cutoff)
        url = (
            API
            + "?"
            + urlencode(
                {
                    "dataset": "TaiwanStockPrice",
                    "data_id": "TAIEX",
                    "start_date": f"{year}-01-01",
                    "end_date": end.isoformat(),
                }
            )
        )
        if offline and not (folder / f"yearly_raw/{year}.json").exists():
            outcomes.append(
                {
                    "year": year,
                    "status": "not_downloaded",
                    "reason": "quota_exhausted_before_completion",
                }
            )
            continue
        try:
            payload = cached(folder / f"yearly_raw/{year}.json", url, end)
            rows = normalize(payload["data"], year, end)
            all_rows.extend(rows)
            outcomes.append(
                {"year": year, "status": "downloaded" if rows else "empty", "rows": len(rows)}
            )
            print(year, len(rows), flush=True)
        except HTTPError as error:
            outcomes.append({"year": year, "status": "blocked", "http_status": error.code})
            print(year, "HTTP", error.code, flush=True)
            if error.code in (402, 403, 429):
                break
        except (ValueError, KeyError, OSError) as error:
            outcomes.append({"year": year, "status": "failed", "error_type": type(error).__name__})
            print(year, type(error).__name__, flush=True)
        if not offline:
            time.sleep(2)
    write_csv(folder / "daily.csv", all_rows)
    for year in sorted({r["date"][:4] for r in all_rows}):
        write_csv(
            folder / f"yearly/{year}.csv", [r for r in all_rows if r["date"].startswith(year)]
        )
    atomic(
        folder / "manifest.json",
        json_bytes(
            {
                "source": "FinMind TaiwanStockPrice TAIEX",
                "cutoff": cutoff.isoformat(),
                "requested_start_year": start,
                "outcomes": outcomes,
                "rows": len(all_rows),
                "first_date": all_rows[0]["date"] if all_rows else None,
                "last_date": all_rows[-1]["date"] if all_rows else None,
                "status": "complete_requested_years"
                if len(outcomes) == cutoff.year - start + 1
                and all(x["status"] == "downloaded" for x in outcomes)
                else "partial",
                "units": "index points",
                "volume_included": False,
                "historical_publication_time_verified": False,
                "daily_csv_sha256": sha256((folder / "daily.csv").read_bytes()),
            }
        ),
    )
    return all_rows


def official_check(root, cutoff, finmind):
    folder = root / "twse/taiex/openapi"
    payload = cached(folder / f"{cutoff.isoformat()}.json", OFFICIAL, cutoff)
    rows = []
    for r in payload:
        s = r["Date"]
        d = date(int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:]))
        parsed, _ = parse_month(
            {
                "stat": "OK",
                "fields": FIELDS,
                "data": [
                    [
                        f"{d.year - 1911}/{d.month:02}/{d.day:02}",
                        r["OpeningIndex"],
                        r["HighestIndex"],
                        r["LowestIndex"],
                        r["ClosingIndex"],
                    ]
                ],
            },
            d.replace(day=1),
            cutoff,
        )
        rows.extend(parsed)
    rows.sort(key=lambda r: r["date"])
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError("duplicate official dates")
    write_csv(folder / "daily.csv", rows)
    import csv

    official = list(rows)
    historical = root / "twse/taiex/daily.csv"
    if historical.exists():
        official.extend(csv.DictReader(historical.open()))
    lookup = {r["date"]: r for r in finmind}
    overlaps = [r for r in official if r["date"] in lookup]
    from decimal import Decimal

    differences = [
        {"date": r["date"], "field": k, "twse": r[k], "finmind": lookup[r["date"]][k]}
        for r in overlaps
        for k in ("open", "high", "low", "close")
        if Decimal(r[k]) != Decimal(lookup[r["date"]][k])
    ]
    atomic(
        root / "reconciliation.json",
        json_bytes(
            {
                "cutoff": cutoff.isoformat(),
                "official_rows": len(official),
                "overlap_dates": len(overlaps),
                "differences": differences,
                "note": "Agreement is a source cross-check, not proof of historical as-of availability.",
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/public"))
    parser.add_argument(
        "--as-of", type=date.fromisoformat, default=datetime.now(TZ).date() - timedelta(days=1)
    )
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument(
        "--offline", action="store_true", help="rebuild FinMind outputs using cache only"
    )
    args = parser.parse_args()
    if (
        args.as_of >= datetime.now(TZ).date()
        or args.start_year < 1990
        or args.start_year > args.as_of.year
    ):
        parser.error("require closed dates and start year >=1990")
    rows = archive(args.output, args.as_of, args.start_year, args.offline)
    official_check(args.output, args.as_of, rows)
    manifest = json.loads((args.output / "finmind/taiex/manifest.json").read_text())
    return 0 if manifest["status"] == "complete_requested_years" else 2


if __name__ == "__main__":
    raise SystemExit(main())
