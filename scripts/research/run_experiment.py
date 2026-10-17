"""Run the predeclared CPU budget; never evaluate sealed final-test data."""

import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from tw_scenario.research.experiment import run

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/research/timesfm3-recovery.json")
parser.add_argument("--audit", default="reports/generated/timesfm-research/data-audit.json")
args = parser.parse_args()
folder = (
    root
    / "reports/generated/timesfm-research"
    / (datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
)
run(root, root / args.config, root / args.audit, folder)
print("Completed validation-only run:", folder)
