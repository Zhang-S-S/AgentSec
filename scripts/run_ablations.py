#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = (
    "full",
    "without_residual_topk",
    "pure_mil",
    "without_graph",
    "without_directionality",
    "without_auxiliary",
    "without_text_supervision",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgentSec ablation suite")
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--graph-arrays", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-splits-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    entry = Path(__file__).with_name("train_fivefold.py")
    for variant in VARIANTS:
        command = [
            sys.executable,
            str(entry),
            "--embedding-dir", str(args.embedding_dir),
            "--graph-arrays", str(args.graph_arrays),
            "--output-root", str(args.output_root / variant),
            "--variant", variant,
            "--device", args.device,
            "--num-workers", str(args.num_workers),
        ]
        if args.reference_splits_root:
            command.extend([
                "--reference-splits-root", str(args.reference_splits_root)
            ])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
