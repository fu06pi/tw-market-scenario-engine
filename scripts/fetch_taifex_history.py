"""Download bounded one-month TX CSVs from the official historical form."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tw_scenario.taifex import fetch_month


def main(start, end, directory):
    months = []
    year, month = int(start[:4]), int(start[4:])
    while f"{year:04d}{month:02d}" <= end:
        months.append(f"{year:04d}{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    def fetch(value):
        result = fetch_month(value, directory)
        print(
            json.dumps(
                {"month": value, "rows": result["monthly_tx_rows"], "sha256": result["sha256"]}
            ),
            flush=True,
        )

    # Small bounded concurrency; cache is immutable and resumable per month.
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, months))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--directory", type=Path, default=Path("data/raw/taifex"))
    args = parser.parse_args()
    main(args.start, args.end, args.directory)
