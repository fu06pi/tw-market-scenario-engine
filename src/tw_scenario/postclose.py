"""Official daytime outcomes, with current receipt evidence and no forecast backfill."""

from __future__ import annotations

import gzip
import hashlib
import urllib.parse
import urllib.request
from datetime import date, datetime, time
from pathlib import Path

from .shadow import TWSE, cash_rows, now, seal
from .snapshot import write_immutable_json
from .split_forecast import TAIPEI, ObservedBar
from .taifex import CSV_URL, parse_tx_csv


def fetch(url, directory, form=None):
    started = now()
    with urllib.request.urlopen(
        urllib.request.Request(url, data=urllib.parse.urlencode(form).encode() if form else None,
                               headers={"User-Agent": "tw-scenario-academic-close/1"}),
        timeout=20,
    ) as response:
        raw = response.read(5_000_001)
    received = now()
    if len(raw) > 5_000_000:
        raise ValueError("oversized official actual response")
    digest = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "source.response").open("xb") as stream:
        stream.write(raw)
    receipt = {
        "url": url, "request_started_at": started.isoformat(),
        "fetched_at": received.isoformat(), "available_at": received.isoformat(),
        "raw_sha256": digest,
        "availability_basis": "Observed post-close HTTP receipt; not historical publication",
        "request_method": "POST" if form else "GET", "request_form": form,
    }
    write_immutable_json(receipt, directory / "receipt.json")
    decoded = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    return decoded, receipt


def collect_day(day: date, contract: str, directory: Path):
    """Conservatively wait until 14:00, even for expiring contracts closing earlier."""
    import re

    if not re.fullmatch(r"[12]\d{3}(0[1-9]|1[0-2])", contract):
        raise ValueError("an explicit monthly TX contract is required")
    if now() < datetime.combine(day, time(14), TAIPEI):
        raise ValueError("complete actual collection must wait until 14:00 Taipei")
    directory.mkdir(parents=True, exist_ok=False)
    records, sources, failures = [], [], []
    queries = {
        "TAIEX": TWSE + "?" + urllib.parse.urlencode({
            "response": "json", "date": day.replace(day=1).strftime("%Y%m%d"),
        }),
        "TX": CSV_URL,
    }
    for market, url in queries.items():
        try:
            form = ({"down_type": "1", "commodity_id": "TX",
                     "queryStartDate": day.strftime("%Y/%m/%d"),
                     "queryEndDate": day.strftime("%Y/%m/%d"), "MarketCode": "0"}
                    if market == "TX" else None)
            raw, receipt = fetch(url, directory / market, form)
            sources.append(receipt)
            if market == "TAIEX":
                rows = cash_rows(raw)
                if any(d.year != day.year or d.month != day.month for d in rows):
                    raise ValueError("TWSE actual response returned a different month")
                if day not in rows:
                    raise ValueError("official TAIEX actual missing; never use nearest date")
                prices = rows[day]
            else:
                rows = parse_tx_csv(raw, str(directory / market / "source.response"))
                matched = [r for r in rows if r.trading_date == day and r.contract == contract
                           and r.session == "day" and r.prices is not None and r.volume]
                if len(matched) != 1:
                    raise ValueError("selected TX contract actual missing; no contract substitution")
                prices = matched[0].prices
            record = ObservedBar(
                **prices.model_dump(), trading_date=day, market=market,
                contract=contract if market == "TX" else None,
                available_at=receipt["available_at"], source=url,
                availability_basis=receipt["availability_basis"] + "; raw_sha256=" +
                receipt["raw_sha256"],
            )
            records.append(record.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 -- each market failure remains observable
            failures.append({"market": market, "date": day.isoformat(), "contract":
                             contract if market == "TX" else None,
                             "failure_type": type(exc).__name__, "reason": str(exc)})
    bundle = seal({"schema_version": "official-day-actuals-v1", "recorded_at": now().isoformat(),
                   "trading_date": day.isoformat(), "selected_contract": contract,
                   "actuals": records, "sources": sources, "failures": failures,
                   "status": "complete" if len(records) == 2 else "partial_or_failed",
                   "historical_preopen_evidence": False})
    write_immutable_json(bundle, directory / "actual-bundle.json")
    write_immutable_json(records, directory / "actuals.json")
    return bundle
