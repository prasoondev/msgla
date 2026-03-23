#!/usr/bin/env python3
"""Plot perplexity vs. position bucket for GLA/MSGLA result JSON files.

By default this script reads four model result files and writes one plot per
shared dataset. With the provided files, that produces two plots:
- pg19
- slimpajama-test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

MAX_PLOT_TOKENS = 32_768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot perplexity vs. position bucket for multiple models.",
    )
    parser.add_argument(
        "--gla-json",
        default="benchmark_gla/results/slimpg19/gla.json",
        help="Path to GLA JSON results.",
    )
    parser.add_argument(
        "--msgla12-json",
        default="benchmark_msgla/results/12/slimpg19/msgla_12_results.json",
        help="Path to MSGLA-12 JSON results.",
    )
    parser.add_argument(
        "--msgla124-json",
        default="benchmark_msgla/results/124/slimpg19/msgla_124_results.json",
        help="Path to MSGLA-124 JSON results.",
    )
    parser.add_argument(
        "--msgla1248-json",
        default="benchmark_msgla/results/1248/slimpg19/msgla_1248_results.json",
        help="Path to MSGLA-1248 JSON results.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/perplexity_vs_position",
        help="Directory where dataset-specific plots are written.",
    )
    parser.add_argument(
        "--max-plot-tokens",
        type=int,
        default=MAX_PLOT_TOKENS,
        help="Upper bound (inclusive) for x-axis bucket positions.",
    )
    return parser.parse_args()


def load_curves_by_dataset(json_path: Path) -> Dict[str, Dict[int, float]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")

    payload = json.loads(json_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {json_path}")

    curves: Dict[str, Dict[int, float]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue

        dataset = row.get("dataset")
        raw_curve = row.get("position_bucket_perplexity")
        if not isinstance(dataset, str) or not isinstance(raw_curve, dict):
            continue

        parsed_curve: Dict[int, float] = {}
        for pos, ppl in raw_curve.items():
            try:
                parsed_curve[int(pos)] = float(ppl)
            except (TypeError, ValueError):
                continue

        if parsed_curve:
            curves[dataset] = parsed_curve

    if not curves:
        raise ValueError(f"No valid dataset curves found in {json_path}")

    return curves


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.split("/")[-1].replace(" ", "_")


def maybe_plot(
    dataset_name: str,
    curves_by_label: Dict[str, Dict[int, float]],
    output_path: Path,
    max_plot_tokens: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(11, 5))
    all_y_values = []

    for label, curve in curves_by_label.items():
        xs_tokens = sorted(k for k in curve.keys() if k <= max_plot_tokens)
        if not xs_tokens:
            continue

        xs_k = [x / 1_000 for x in xs_tokens]
        ys_ppl = [curve[x] for x in xs_tokens]
        all_y_values.extend(ys_ppl)

        ax.plot(xs_k, ys_ppl, marker="o", markersize=3, linewidth=1.5, label=label)

    if not all_y_values:
        plt.close(fig)
        return False

    ax.set_xlim(0, max_plot_tokens / 1_000)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Position Bucket (K tokens)", fontsize=12)

    y_min = min(all_y_values)
    y_max = max(all_y_values)
    if math.isclose(y_min, y_max):
        pad = max(0.5, abs(y_min) * 0.05)
    else:
        pad = (y_max - y_min) * 0.08
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_ylabel("Perplexity", fontsize=12)

    ax.set_title(f"{dataset_name} Perplexity vs. Position Bucket", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()

    model_paths = {
        "GLA": Path(args.gla_json),
        "MSGLA-12": Path(args.msgla12_json),
        "MSGLA-124": Path(args.msgla124_json),
        "MSGLA-1248": Path(args.msgla1248_json),
    }

    curves_by_model_and_dataset: Dict[str, Dict[str, Dict[int, float]]] = {
        label: load_curves_by_dataset(path) for label, path in model_paths.items()
    }

    shared_datasets = set.intersection(
        *(
            set(curves_by_dataset.keys())
            for curves_by_dataset in curves_by_model_and_dataset.values()
        )
    )

    if not shared_datasets:
        raise RuntimeError("No shared datasets found across all four result files.")

    output_dir = Path(args.output_dir)
    for dataset_name in sorted(shared_datasets):
        curves_by_label = {
            label: curves_by_dataset[dataset_name]
            for label, curves_by_dataset in curves_by_model_and_dataset.items()
        }

        output_path = (
            output_dir / f"{dataset_slug(dataset_name)}_perplexity_vs_position.png"
        )
        plotted = maybe_plot(
            dataset_name=dataset_name,
            curves_by_label=curves_by_label,
            output_path=output_path,
            max_plot_tokens=args.max_plot_tokens,
        )

        if plotted:
            print(f"Saved plot: {output_path}")
        else:
            print(
                f"Skipped dataset (no points <= {args.max_plot_tokens}): {dataset_name}"
            )


if __name__ == "__main__":
    main()
