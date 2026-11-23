"""Fetch official outcomes after 14:00. Historical dates are outcomes, not forecasts."""

import argparse
from datetime import date
from pathlib import Path

from tw_scenario.postclose import collect_day
from tw_scenario.shadow import now

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=now().date())
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect_day(args.date, args.contract, args.output)
    print({"status": result["status"], "actuals": len(result["actuals"]),
           "failures": result["failures"]})
    raise SystemExit(0 if result["status"] == "complete" else 2)
