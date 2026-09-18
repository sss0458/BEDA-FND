#!/usr/bin/env python3
"""Plot the structural ablation table from evaluation metric files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("weibo", "weibo21")
VARIANTS = ("full", "without_drcm", "without_bem", "without_refine", "without_dcea", "without_all")
LABELS = {
    "full": "Full",
    "without_drcm": "w/o DRCM",
    "without_bem": "w/o BEM",
    "without_refine": "w/o Refine",
    "without_dcea": "w/o DCEA",
    "without_all": "w/o All",
}
COLORS = ("#2F5D8A", "#74A387", "#79A7D3", "#9275AE", "#D98E5F", "#A9A9A9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--results-root",
        type=Path,
        help="Directory containing <dataset>/<variant>/metrics.json",
    )
    source.add_argument(
        "--reference-results", type=Path,
        help="Aggregate accuracy JSON, such as results/ablation_accuracy.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    values = {}
    if args.reference_results:
        reference = json.loads(args.reference_results.expanduser().read_text())
        if reference["metric"] != "accuracy" or reference["unit"] != "percent":
            raise ValueError("Reference results must contain accuracy in percent")
        rows = {row["variant"]: row for row in reference["results"]}
        for dataset in DATASETS:
            for variant in VARIANTS:
                values[(dataset, variant)] = float(rows[variant][dataset])
    else:
        results_root = args.results_root.expanduser().resolve()
        for dataset in DATASETS:
            for variant in VARIANTS:
                path = results_root / dataset / variant / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                values[(dataset, variant)] = json.loads(path.read_text())["accuracy"] * 100

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 2.8), sharey=True)
    all_values = np.asarray(list(values.values()))
    lower = max(0.0, float(all_values.min()) - 0.8)
    upper = min(100.0, float(all_values.max()) + 0.8)
    for axis, dataset in zip(axes, DATASETS):
        heights = [values[(dataset, variant)] for variant in VARIANTS]
        bars = axis.bar(range(len(VARIANTS)), heights, color=COLORS, width=0.68)
        full = heights[0]
        for index, (bar, height) in enumerate(zip(bars, heights)):
            label = f"{height:.2f}"
            if index:
                label += f"\n$\\downarrow${full - height:.2f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.08,
                label,
                ha="center",
                va="bottom",
                fontsize=7.8,
            )
        axis.set_title("Weibo" if dataset == "weibo" else "Weibo21")
        axis.set_xticks(range(len(VARIANTS)), [LABELS[item] for item in VARIANTS], rotation=15)
        axis.set_ylim(lower, upper)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Accuracy (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_accuracy.png", dpi=350, bbox_inches="tight")
    figure.savefig(output_dir / "ablation_accuracy.pdf", bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
