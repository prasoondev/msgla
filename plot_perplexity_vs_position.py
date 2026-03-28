#!/usr/bin/env python3
"""Plot perplexity vs. position bucket for GLA + MSGLA slimpg19 result JSON files.

By default this script discovers all variants in:
benchmark_msgla/results/7B/*/slimpg19/*.json
and also includes benchmark_gla/results/7B/slimpg19/gla_results.json.
It writes one combined figure with subplots for selected shared datasets
(preferentially pg19 and slimpajama-test).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List

MAX_PLOT_TOKENS = 32_768

# Okabe-Ito color palette (widely used as colorblind-friendly).
COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot perplexity vs. position bucket for multiple models.",
    )
    parser.add_argument(
        "--results-root",
        default="benchmark_msgla/results/7B",
        help="Root directory that contains per-variant folders (e.g., 12, 14, 24).",
    )
    parser.add_argument(
        "--gla-json",
        default="benchmark_gla/results/7B/slimpg19/gla_results.json",
        help="Path to GLA slimpg19 JSON results.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/perplexity_vs_position",
        help="Directory where plot(s) are written.",
    )
    parser.add_argument(
        "--combined-output-name",
        default="pg19_slimpajama_perplexity_vs_position.png",
        help="Filename for the combined multi-dataset figure.",
    )
    parser.add_argument(
        "--max-plot-tokens",
        type=int,
        default=MAX_PLOT_TOKENS,
        help="Upper bound (inclusive) for x-axis bucket positions.",
    )
    return parser.parse_args()


def variant_sort_key(variant_name: str) -> tuple[int, int | str]:
    try:
        return (0, int(variant_name))
    except ValueError:
        return (1, variant_name)


def discover_variant_jsons(results_root: Path) -> Dict[str, Path]:
    if not results_root.exists():
        raise FileNotFoundError(f"Missing results root: {results_root}")

    candidate_paths = sorted(
        p for p in results_root.glob("*/slimpg19/*.json") if p.is_file()
    )
    if not candidate_paths:
        raise FileNotFoundError(
            f"No result JSONs found at {results_root}/<variant>/slimpg19/*.json"
        )

    variant_to_path: Dict[str, Path] = {}
    for path in candidate_paths:
        variant = path.parent.parent.name
        if variant in variant_to_path:
            print(
                "Warning: multiple slimpg19 JSON files found for variant "
                f"{variant}; using {variant_to_path[variant]} and skipping {path}"
            )
            continue
        variant_to_path[variant] = path

    model_paths: Dict[str, Path] = {}
    for variant in sorted(variant_to_path.keys(), key=variant_sort_key):
        model_paths[f"MSGLA-{variant}"] = variant_to_path[variant]

    return model_paths


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


def _select_datasets_for_combined_plot(shared_datasets: set[str]) -> List[str]:
    preferred_keywords = ("pg19", "slimpajama")
    selected: List[str] = []

    for keyword in preferred_keywords:
        for dataset_name in sorted(shared_datasets):
            if keyword in dataset_name.lower() and dataset_name not in selected:
                selected.append(dataset_name)
                break

    if not selected:
        selected = sorted(shared_datasets)[:2]

    return selected


def maybe_plot_combined(
    dataset_names: List[str],
    curves_by_model_and_dataset: Dict[str, Dict[str, Dict[int, float]]],
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

    if not dataset_names:
        return False

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 20,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(dataset_names),
        figsize=(7.2 * len(dataset_names), 6.6),
        sharey=False,
    )
    if len(dataset_names) == 1:
        axes = [axes]

    model_labels = sorted(curves_by_model_and_dataset.keys(), key=variant_sort_key)
    color_by_model = {
        label: color
        for label, color in zip(model_labels, itertools.cycle(COLORBLIND_PALETTE))
    }

    plotted_any = False
    for ax, dataset_name in zip(axes, dataset_names):
        all_y_values = []
        for label in model_labels:
            curve = curves_by_model_and_dataset.get(label, {}).get(dataset_name)
            if not curve:
                continue

            xs_tokens = sorted(k for k in curve.keys() if k <= max_plot_tokens)
            if not xs_tokens:
                continue

            xs_k = [x / 1_000 for x in xs_tokens]
            ys_ppl = [curve[x] for x in xs_tokens]
            all_y_values.extend(ys_ppl)

            ax.plot(
                xs_k,
                ys_ppl,
                marker="o",
                markersize=8,
                markeredgewidth=1.1,
                linewidth=2.4,
                label=label,
                color=color_by_model[label],
            )
            plotted_any = True

        ax.set_xlim(0, max_plot_tokens / 1_000)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlabel("Position Bucket (K tokens)")

        if all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            if math.isclose(y_min, y_max):
                pad = max(0.5, abs(y_min) * 0.05)
            else:
                pad = (y_max - y_min) * 0.08
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        ax.set_ylabel("Perplexity")
        ax.set_title(dataset_slug(dataset_name))
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)

    if not plotted_any:
        plt.close(fig)
        return False

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()

    msgla_model_paths = discover_variant_jsons(Path(args.results_root))
    print(
        f"Discovered {len(msgla_model_paths)} MSGLA variants under {args.results_root}."
    )
    for label, path in msgla_model_paths.items():
        print(f"  {label}: {path}")

    model_paths: Dict[str, Path] = {"GLA": Path(args.gla_json)}
    model_paths.update(msgla_model_paths)
    print(f"  GLA: {args.gla_json}")

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
        raise RuntimeError("No shared datasets found across discovered result files.")

    output_dir = Path(args.output_dir)
    selected_datasets = _select_datasets_for_combined_plot(shared_datasets)
    output_path = output_dir / args.combined_output_name
    plotted = maybe_plot_combined(
        dataset_names=selected_datasets,
        curves_by_model_and_dataset=curves_by_model_and_dataset,
        output_path=output_path,
        max_plot_tokens=args.max_plot_tokens,
    )
    if plotted:
        print(f"Saved combined plot: {output_path}")
        print(f"Included datasets: {', '.join(selected_datasets)}")
    else:
        print(
            "Skipped combined plot (no points <= "
            f"{args.max_plot_tokens}) for datasets: {', '.join(selected_datasets)}"
        )


if __name__ == "__main__":
    main()
