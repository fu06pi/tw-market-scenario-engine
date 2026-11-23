"""Run now, without a historical clock override; save late/source failures."""

import argparse
from pathlib import Path

from tw_scenario.shadow import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.checkpoint, args.output))
