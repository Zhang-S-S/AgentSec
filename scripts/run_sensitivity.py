#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def _values(text: str, cast):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARA sensitivity experiments")
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--graph-arrays", type=Path, required=True)
    parser.add_argument("--reference-splits-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--post-top-k", default="3")
    parser.add_argument("--comment-top-k", default="5")
    parser.add_argument("--neighbors", default="4")
    parser.add_argument("--text-loss-weight", default="1.0")
    parser.add_argument("--max-content", default="128")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    entry = Path(__file__).with_name("train_fivefold.py")
    grid = itertools.product(
        _values(args.post_top_k, int),
        _values(args.comment_top_k, int),
        _values(args.neighbors, int),
        _values(args.text_loss_weight, float),
        _values(args.max_content, int),
    )
    for post_k, comment_k, neighbors, text_weight, max_content in grid:
        name = (
            f"post{post_k}_comment{comment_k}_neighbors{neighbors}_"
            f"text{text_weight:g}_content{max_content}"
        )
        command = [
            sys.executable, str(entry),
            "--embedding-dir", str(args.embedding_dir),
            "--graph-arrays", str(args.graph_arrays),
            "--reference-splits-root", str(args.reference_splits_root),
            "--output-root", str(args.output_root / name),
            "--post-top-k", str(post_k),
            "--comment-top-k", str(comment_k),
            "--neighbors", str(neighbors),
            "--text-loss-weight", str(text_weight),
            "--max-posts", str(max_content),
            "--max-comments", str(max_content),
            "--device", args.device,
            "--num-workers", str(args.num_workers),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
