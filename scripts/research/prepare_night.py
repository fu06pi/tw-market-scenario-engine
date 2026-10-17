"""Collect completed official night-session quotes only through validation end."""

import json
from pathlib import Path

from tw_scenario.research.data import canonical_hash, load_dataset, write_new
from tw_scenario.research.taifex import fetch_night

root = Path(__file__).resolve().parents[2]
audit = json.loads((root / "reports/generated/timesfm-research/data-audit.json").read_text())
_, taiex = load_dataset(root / audit["markets"]["TAIEX"]["file"])
raw = json.loads((root / "data/raw/timesfm-research/TX202603.json").read_text())["rows"]
by_day = {row["mdate"][:10]: row for row in raw}
days = [str(b.trading_date) for b in taiex if "2025-03-20" <= str(b.trading_date) <= "2025-11-30"]
cache = root / "data/raw/taifex-night"
cache.mkdir(parents=True, exist_ok=True)
quotes = []
for number, day in enumerate(days):
    if not (cache / f"{day}-TX202603-night.json").exists():
        import time

        time.sleep(2.0)  # Single request stream; respect official service load/rate limits.
    quotes.append(fetch_night(cache, day))
    if (number + 1) % 25 == 0:
        print(f"Night cache progress: {number + 1}/{len(days)} dates", flush=True)
trading_days = [str(b.trading_date) for b in taiex]
series = {}
missing = {}
for quote in quotes:
    day = quote["date"]
    i = trading_days.index(day)
    previous = by_day.get(trading_days[i - 1]) if i else None
    if quote["close"] is not None and previous and previous["close_d"] is not None:
        import math

        series[day] = math.log(quote["close"] / previous["close_d"])
    else:
        missing[day] = (
            quote["reason"] if quote["close"] is None else "same_contract_previous_close_missing"
        )
result = {
    "source": "TAIFEX official night / TEJ same-contract previous daytime close",
    "contract": "TX202603",
    "series": series,
    "missing": missing,
    "final_test_requested": False,
    "historical_publication_verified": False,
    "available_by_0830": "session ends 05:00; historical publication delay unverified",
    "missing_policy": "explicit mask; never create zero-return bars",
}
result["checksum"] = canonical_hash(result)
path = root / "data/processed/timesfm-research/night-TX202603.json"
if path.exists():
    if json.loads(path.read_text())["checksum"] != result["checksum"]:
        raise ValueError("existing night dataset differs; new version required")
else:
    write_new(path, result)
print(
    json.dumps(
        {
            "quoted_days": len(series),
            "missing_days": len(missing),
            "file": str(path.relative_to(root)),
            "checksum": result["checksum"],
        },
        indent=2,
    )
)
