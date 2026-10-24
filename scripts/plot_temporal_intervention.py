#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


METHOD_STYLE = {
    "AgentSec": {
        "label": "AgentSec",
        "color": "#3B5B92",
        "marker": "o",
        "linestyle": "-",
    },
    "AgentSec + historical out-degree": {
        "label": "AgentSec + Max Degree",
        "color": "#D86B5B",
        "marker": "s",
        "linestyle": "-",
    },
    "Historical out-degree": {
        "label": "Max Degree",
        "color": "#55A786",
        "marker": "^",
        "linestyle": "--",
    },
    "Random": {
        "label": "Random",
        "color": "#8B8B8B",
        "marker": "D",
        "linestyle": ":",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--recent-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.summary.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if args.font and args.font.exists():
        font_manager.fontManager.addfont(str(args.font))
        font_name = font_manager.FontProperties(fname=str(args.font)).get_name()
    else:
        font_name = "serif"

    plt.rcParams.update({
        "font.family": font_name,
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    grouped = {}
    for row in payload["results"]:
        grouped.setdefault(row["method"], []).append(row)

    method_style = dict(METHOD_STYLE)
    if args.recent_summary:
        with args.recent_summary.open(encoding="utf-8") as handle:
            recent_payload = json.load(handle)
        for row in recent_payload["new_results"]:
            grouped.setdefault(row["method"], []).append(row)
        method_style = {
            "AgentSec + recent out-degree": {
                "label": "AgentSec",
                "color": "#D86B5B",
                "marker": "s",
                "linestyle": "-",
            },
            "Recent out-degree (all accounts)": {
                "label": "Out-degree",
                "color": "#55A786",
                "marker": "^",
                "linestyle": "--",
            },
            "Random": METHOD_STYLE["Random"],
        }

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for method, style in method_style.items():
        rows = sorted(grouped[method], key=lambda item: item["intervention_ratio"])
        x = np.array([row["intervention_ratio"] * 100 for row in rows])
        y = np.array([row["exposure_reduction"] * 100 for row in rows])
        ax.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6.0,
            markerfacecolor="white",
            markeredgewidth=1.4,
            zorder=3,
        )
        if method == "Random":
            std = np.array([
                row.get("exposure_reduction_std", 0.0) * 100 for row in rows
            ])
            ax.fill_between(
                x,
                y - std,
                y + std,
                color=style["color"],
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )

    ax.set_xlabel("Intervention budget (%)")
    ax.set_ylabel("Exposure reduction (%)")
    ax.set_xlim(-1, 51)
    ax.set_ylim(0, 50)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="both", linestyle="--", linewidth=0.7, color="#BDBDBD", alpha=0.55)
    ax.legend(loc="upper left", frameon=True, ncol=1, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = args.output_dir / f"temporal_intervention_exposure.{suffix}"
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        print(output)


if __name__ == "__main__":
    main()
