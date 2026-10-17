"""Explicitly retrospective TEJ trial data; never silently treat it as live/as-of data."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tw_scenario.research.data import Bar, canonical_hash, write_new
from tw_scenario.research.tej import AccessError, get_rows, metadata

ROOT = Path(__file__).resolve().parents[2]
cache = ROOT / "data/raw/timesfm-research"
cache.mkdir(parents=True, exist_ok=True)
report = {
    "fetched_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
    "markets": {},
    "permissions": {},
}
for table in ["TWN/AFUTR", "TWN/AFUTRSTD"]:
    try:
        metadata(table)
        report["permissions"][table] = "metadata_access_passed"
    except AccessError as e:
        report["permissions"][table] = str(e)
for market, table, identifier, vol in [
    ("TAIEX", "TRAIL/TAPRCD", "Y9999", "volume"),
    ("TX", "TRAIL/TAFUTR", "TX202603", "vol_d"),
]:
    cache_file = cache / f"{identifier}.json"
    if cache_file.exists():
        source = json.loads(cache_file.read_text())
    else:
        meta = metadata(table)
        names = {column["name"] for column in meta["columns"]}
        columns = ["coid", "mdate", "open_d", "high_d", "low_d", "close_d", vol]
        if market == "TX":
            columns += ["due_m", "cashnm_c"]
        if not set(columns) <= names:
            raise ValueError("required verified fields missing")
        raw = get_rows(table, identifier, columns)
        source = {
            "table": table,
            "identifier": identifier,
            "fetched_at": report["fetched_at"],
            "metadata": meta,
            "rows": raw,
        }
        write_new(cache_file, source)
    raw = sorted(source["rows"], key=lambda r: r["mdate"])
    if any(row["coid"] != identifier for row in raw):
        raise ValueError("identifier mismatch")
    dates = [row["mdate"][:10] for row in raw]
    if len(set(dates)) != len(dates):
        raise ValueError("duplicate source dates")
    missing = [
        row["mdate"][:10]
        for row in raw
        if any(row[f + "_d"] is None for f in ["open", "high", "low", "close"])
    ]
    # Retain an explicit contiguous valid suffix; no interpolated or zero-filled prices.
    after = max(missing) if missing else ""
    clean = [row for row in raw if row["mdate"][:10] > after]
    bars = []
    for row in clean:
        if market == "TX" and (
            not row["due_m"].startswith("2026-03") or row["cashnm_c"] != "加權指數"
        ):
            raise ValueError("TX underlying/month mismatch")
        bars.append(
            Bar(
                market=market,
                contract=identifier if market == "TX" else None,
                trading_date=row["mdate"][:10],
                volume=row[vol],
                **{f: row[f + "_d"] for f in ["open", "high", "low", "close"]},
            )
        )
    dataset = {
        "schema_version": 1,
        "source": "TEJ trial",
        "table": table,
        "identifier": identifier,
        "fetched_at": source["fetched_at"],
        "historical_available_at_verified": False,
        "calendar_verified": False,
        "session": "day candidate; official session audit pending",
        "price_unit": "index_points; generic metadata NTD, identified index underlying",
        "volume_unit": "contracts" if market == "TX" else "thousand_shares",
        "missing_ohlc_dates": missing,
        "excluded_dates": [d for d in dates if d <= after],
        "selection_rule": "contiguous clean suffix following last missing OHLC",
        "bars": [bar.model_dump(mode="json") for bar in bars],
    }
    dataset["checksum"] = canonical_hash(dataset)
    path = ROOT / "data/processed/timesfm-research" / f"{market}-{dataset['checksum'][:12]}.json"
    if not path.exists():
        write_new(path, dataset)
    report["markets"][market] = {
        "file": str(path.relative_to(ROOT)),
        "checksum": dataset["checksum"],
        "rows": len(bars),
        "first": str(bars[0].trading_date),
        "last": str(bars[-1].trading_date),
        "missing_source_ohlc_count": len(missing),
        "excluded_before_clean_window_count": len(dataset["excluded_dates"]),
        "live_ready": False,
        "historical_asof_verified": False,
    }
report_path = ROOT / "reports/generated/timesfm-research/data-audit.json"
if not report_path.exists():
    write_new(report_path, report)
else:
    report_path = report_path.with_name(
        "data-audit-" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"
    )
    write_new(report_path, report)
print(json.dumps(report, indent=2))
