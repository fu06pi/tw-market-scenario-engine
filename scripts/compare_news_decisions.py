"""Compare independently frozen news decisions to unchanged quantitative baselines."""

import argparse
import json
from pathlib import Path

from tw_scenario.news_comparison import evaluate_pairs
from tw_scenario.snapshot import write_immutable_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, nargs="+", required=True)
    parser.add_argument("--decisions", type=Path, nargs="*", default=[])
    parser.add_argument("--actuals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_pairs([json.loads(p.read_text()) for p in args.forecasts],
                            [json.loads(p.read_text()) for p in args.decisions],
                            json.loads(args.actuals.read_text()))
    write_immutable_json(report, args.output)
