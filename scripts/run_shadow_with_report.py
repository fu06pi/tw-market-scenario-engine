"""Freeze numerical shadow output, then export the observed morning-report format."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tw_scenario.morning_adapter import HEADERS, morning_export
from tw_scenario.shadow import now, run
from tw_scenario.snapshot import write_immutable_json


def main(checkpoint, output):
    code = run(checkpoint, output)
    if code:
        return code
    try:
        forecast = json.loads((output / "forecast.json").read_text())
        report = morning_export(forecast, HEADERS)
        write_immutable_json(report, output / "morning-export.json")
        at = now()
        if at >= datetime.fromisoformat(forecast["deadline"]):
            raise ValueError("morning export missed 08:40; do not claim on-time integration")
        write_immutable_json({"report_frozen_at": at.isoformat(),
                              "forecast_hash": forecast["snapshot_hash"],
                              "report_hash": report["snapshot_hash"], "live_sheet_write": False},
                             output / "morning-receipt.json")
        return 0
    except Exception as exc:  # noqa: BLE001 -- boundary must preserve integration failures
        write_immutable_json({"status": "failed", "recorded_at": now().isoformat(),
                              "reason": str(exc), "eligible_for_formal_accuracy": False},
                             output / "failure.json")
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.checkpoint, args.output))
