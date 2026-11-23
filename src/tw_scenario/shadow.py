"""Prospective shadow collection. Wall-clock receipt time is never backdated."""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import math
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .intervals import bands
from .refinements import predict
from .snapshot import HASH_VERSION, document_hash, verify_document, write_immutable_json
from .split_forecast import OHLC, TAIPEI, encode_components

TWSE = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
TAIFEX = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"


def seal(payload):
    result = {**payload, "hash_version": HASH_VERSION}
    result["snapshot_hash"] = document_hash(result)
    return result


def now():
    return datetime.now(TAIPEI)


def window(at):
    if at.tzinfo is None:
        raise ValueError("aware wall clock required")
    local = at.astimezone(TAIPEI)
    if local.weekday() >= 5 or not time(6) <= local.time() < time(8, 30):
        raise ValueError("shadow collection must start on weekdays at 06:00–08:30 Taipei")
    return local.date()


def fetch(url, directory):
    started = now()
    window(started)
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "tw-scenario-academic-shadow/1"}),
        timeout=20,
    ) as response:
        raw = response.read(5_000_001)
    received = now()
    if len(raw) > 5_000_000:
        raise ValueError("oversized official response")
    if received.date() != started.date() or received.time() > time(8, 30):
        raise ValueError("source receipt missed 08:30 cutoff")
    digest = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{digest}.response").open("xb") as stream:
        stream.write(raw)
    receipt = {
        "url": url, "request_started_at": started.isoformat(),
        "fetched_at": received.isoformat(), "available_at": received.isoformat(),
        "availability_basis": "Observed HTTP receipt on this run; not historical publication",
        "raw_sha256": digest,
    }
    write_immutable_json(receipt, directory / f"{digest}.receipt.json")
    decoded = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    return decoded, receipt


def cash_rows(raw):
    payload = json.loads(raw)
    if payload.get("stat") == "很抱歉，沒有符合條件的資料!" and not payload.get("data"):
        return {}  # For example, the new month before its first regular session.
    expected = ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"]
    if payload.get("stat") != "OK" or payload.get("fields") != expected:
        raise ValueError("TWSE response/schema not confirmed")
    result = {}
    for row in payload["data"]:
        y, m, d = map(int, row[0].split("/"))
        day = date(y + 1911, m, d)
        if day in result:
            raise ValueError("duplicate TWSE day")
        result[day] = OHLC(**dict(zip(
            ("open", "high", "low", "close"),
            [float(v.replace(",", "")) for v in row[1:5]],
        )))
    return result


def futures_rows(raw, day, session):
    text = raw.decode("utf-8")
    if "dataStart" not in text:
        raise ValueError("missing official report data section")
    section = text.split("dataStart", 1)[1]
    clean = lambda s: " ".join(html.unescape(re.sub(r"<[^>]*>", " ", s)).split())
    header = clean(section.split("<table", 1)[0])
    if not re.search(r"日期\s*[：:]\s*" + day.strftime("%Y/%m/%d"), header):
        raise ValueError("TAIFEX ignored requested trading date")
    night_start = None
    if session == "night":
        matched = re.search(r"(\d{4}/\d{2}/\d{2})\s+15:00~次日05:00 盤後", header)
        if not matched:
            raise ValueError("night session identity is not confirmed")
        night_start = datetime.strptime(matched[1], "%Y/%m/%d").date()
    elif session != "day" or "一般交易時段" not in header:
        raise ValueError("day session identity is not confirmed")
    table = section.split("</table>", 1)[0]
    if "最後" not in table or "成交價" not in table:
        raise ValueError("cannot identify last transaction price")
    rows = {}
    for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.DOTALL | re.IGNORECASE):
        cells = [clean(c) for c in re.findall(r"<td\b[^>]*>(.*?)</td>", tr, re.DOTALL | re.IGNORECASE)]
        if (len(cells) < 9 or cells[0] != "TX"
                or not re.fullmatch(r"[12]\d{3}(0[1-9]|1[0-2])", cells[1])):
            continue
        if cells[1] in rows:
            raise ValueError("duplicate monthly TX contract")
        if cells[5] in ("-", "--", "") or int(cells[8].replace(",", "")) <= 0:
            continue
        rows[cells[1]] = OHLC(**dict(zip(
            ("open", "high", "low", "close"),
            [float(v.replace(",", "")) for v in cells[2:6]],
        )))
    if not rows:
        raise ValueError("no traded monthly TX rows")
    return rows, night_start


def collect(directory):
    day = window(now())
    receipts, cash = [], {}
    for month in sorted({day.replace(day=1), (day.replace(day=1) - timedelta(days=1)).replace(day=1)}):
        raw, receipt = fetch(TWSE + "?" + urllib.parse.urlencode({
            "response": "json", "date": month.strftime("%Y%m%d"),
        }), directory)
        monthly = cash_rows(raw)
        if any(d.year != month.year or d.month != month.month for d in monthly):
            raise ValueError("TWSE returned a different month")
        cash.update(monthly)
        receipts.append(receipt)
    previous = sorted(d for d in cash if d < day)
    if len(previous) < 2 or (day - previous[-1]).days > 10:
        raise ValueError("missing or stale previous cash sessions")
    prior, earlier = previous[-1], previous[-2]

    def get_tx(target, session):
        raw, receipt = fetch(TAIFEX + "?" + urllib.parse.urlencode({
            "queryType": "2", "marketCode": "1" if session == "night" else "0",
            "commodity_id": "TX", "queryDate": target.strftime("%Y/%m/%d"),
        }), directory)
        rows, start = futures_rows(raw, target, session)
        receipts.append(receipt)
        return rows, start

    night, start = get_tx(day, "night")
    if start != prior:
        raise ValueError("night session does not follow previous cash session")
    end = datetime.combine(start + timedelta(days=1), time(5), TAIPEI)
    if end > now():
        raise ValueError("unfinished night session")
    contract = min(night)
    if contract < day.strftime("%Y%m"):
        raise ValueError("expired selected contract")
    prior_tx, _ = get_tx(prior, "day")
    earlier_tx, _ = get_tx(earlier, "day")
    if contract not in prior_tx or contract not in earlier_tx:
        raise ValueError("selected contract lacks same-contract previous day prices")
    night_return = 10000 * (night[contract].close / prior_tx[contract].close - 1)
    rows = {}
    for market, p, e in (("TAIEX", cash[prior], cash[earlier]),
                         ("TX_selected_monthly", prior_tx[contract], earlier_tx[contract])):
        lag = encode_components(e.close, p)
        rows[market] = {
            "date": day.isoformat(), "market": market,
            "contract": contract if market != "TAIEX" else None,
            "night_contract": contract, "previous_close": p.close,
            "feature_latest_date": prior.isoformat(), "night_return_bps": night_return,
            "features": [math.expm1(lag.g) * 10000, math.expm1(lag.g + lag.b) * 10000,
                         math.expm1(lag.b) * 10000, (p.high - p.low) / e.close * 10000],
            "night_session_start": start.isoformat(), "night_session_end": end.isoformat(),
            "night_age_hours": (now() - end).total_seconds() / 3600,
        }
    return rows, receipts


def generate(bundle, rows, receipts, at):
    """Validate live receipts again before generating an immutable numerical bundle."""
    if not verify_document(bundle).valid:
        raise ValueError("checkpoint hash mismatch")
    if at.tzinfo is None:
        raise ValueError("aware generation clock required")
    local = at.astimezone(TAIPEI)
    frozen = datetime.fromisoformat(bundle["frozen_at"])
    if (frozen.tzinfo is None or frozen.astimezone(TAIPEI).date() >= local.date()
            or not receipts):
        raise ValueError("checkpoint must predate generation and sources are required")
    if local.time() >= time(8, 40):
        raise ValueError("generation missed 08:40 persistence deadline")
    if set(rows) != set(bundle["markets"]):
        raise ValueError("both independent markets are required")
    for receipt in receipts:
        stamp = datetime.fromisoformat(receipt["fetched_at"])
        if stamp.tzinfo is None or stamp > at or stamp.astimezone(TAIPEI).date() != local.date():
            raise ValueError("receipt must be from this run's forecast date")
        if stamp.astimezone(TAIPEI).time() > time(8, 30):
            raise ValueError("source was received after 08:30")
    output = {}
    for market, row in rows.items():
        if "actual" in row or "target" in row:
            raise ValueError("same-day actual targets cannot enter a preopen snapshot")
        if row["date"] != local.date().isoformat() or row["market"] != market:
            raise ValueError("forecast date or market mismatch")
        models = {}
        for method, checkpoint in bundle["markets"][market]["models"].items():
            components, prices = predict(checkpoint, row)
            interval = bands(bundle["markets"][market]["intervals"][method], prices,
                             row["previous_close"], market, method, row["date"])
            models[method] = {"components": components.model_dump(),
                              "prices": prices.model_dump(), "marginal_80": interval}
        output[market] = {"inputs": row, "models": models,
                          "default_baseline": bundle["markets"][market]["default_baseline"]}
    return seal({
        "schema_version": "prospective-shadow-v1", "status": "shadow_only",
        "trading_date": local.date().isoformat(), "generated_at": at.isoformat(),
        "data_cutoff_at": max(r["fetched_at"] for r in receipts),
        "deadline": datetime.combine(local.date(), time(8, 40), TAIPEI).isoformat(),
        "checkpoint_hash": bundle["snapshot_hash"], "sources": receipts, "markets": output,
        "scoring_policy": bundle["scoring_policy"],
        "local_clock_evidence_only": True, "remote_persistence_verified": False,
        "eligible_for_formal_accuracy": False,
    })


def run(checkpoint: Path, output: Path):
    """No clock/date override: late invocations save failures rather than forecasts."""
    output.mkdir(parents=True, exist_ok=False)
    try:
        window(now())
        bundle = json.loads(checkpoint.read_text())
        rows, receipts = collect(output / "sources")
        forecast = generate(bundle, rows, receipts, now())
        write_immutable_json(forecast, output / "forecast.json")
        persisted = now()
        if persisted >= datetime.fromisoformat(forecast["deadline"]):
            raise ValueError("local numerical persistence missed 08:40")
        write_immutable_json({"local_persisted_at": persisted.isoformat(),
                              "snapshot_hash": forecast["snapshot_hash"],
                              "remote_attestation": "pending GitHub artifact API readback"},
                             output / "local-receipt.json")
        return 0
    except Exception as exc:  # noqa: BLE001 -- boundary must persist every attempted-run failure
        write_immutable_json({"status": "failed", "recorded_at": now().isoformat(),
                              "failure_type": type(exc).__name__, "reason": str(exc),
                              "eligible_for_formal_accuracy": False}, output / "failure.json")
        return 2
