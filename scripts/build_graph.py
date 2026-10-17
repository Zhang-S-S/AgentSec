#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from comment_network.analysis import AnalysisConfig, build_and_analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the directed comment graph")
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--labeled-only", action="store_true")
    args = parser.parse_args()
    report = build_and_analyze(AnalysisConfig(
        source_db=args.source_db,
        output_dir=args.output_dir,
        labels=args.labels,
        seed=args.seed,
        permutations=args.permutations,
        labeled_only=args.labeled_only,
    ))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
