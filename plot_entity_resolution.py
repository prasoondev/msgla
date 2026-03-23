#!/usr/bin/env python3
"""Plot named entity resolution retrieval for Entity C across four models.

This script reads the standard entity-resolution JSON outputs for:
- GLA
- MSGLA-12
- MSGLA-124
- MSGLA-1248

and writes a single compact figure:
- Retrieval score (ret_C) vs query gap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ENTITY_ID = "C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot named entity resolution for Entity C across models.",
    )
    parser.add_argument(
        "--gla-json",
        default="benchmark_gla/results/entity_resolution/gla_entity_full.json",
        help="Path to GLA entity-resolution JSON.",
    )
    parser.add_argument(
        "--msgla12-json",
        default="benchmark_msgla/results/12/entity_resolution/msgla_entity_full.json",
        help="Path to MSGLA-12 entity-resolution JSON.",
    )
    parser.add_argument(
        "--msgla124-json",
        default="benchmark_msgla/results/124/entity_resolution/msgla_entity_full.json",
        help="Path to MSGLA-124 entity-resolution JSON.",
    )
    parser.add_argument(
        "--msgla1248-json",
        default="benchmark_msgla/results/1248/entity_resolution/msgla_entity_full.json",
        help="Path to MSGLA-1248 entity-resolution JSON.",
    )
    parser.add_argument(
        "--output",
        default="plots/entity_resolution/named_entity_resolution_entity_c.png",
        help="Output image path.",
    )
    return parser.parse_args()


def load_rows(json_path: Path) -> List[Dict[str, float]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")

    payload = json.loads(json_path.read_text())
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(f"Unexpected JSON schema in {json_path}")

    results = payload["results"]
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' to be a list in {json_path}")

    parsed_rows: List[Dict[str, float]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        if "query_gap" not in row:
            continue

        parsed: Dict[str, float] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                continue

        if "query_gap" in parsed:
            parsed_rows.append(parsed)

    if not parsed_rows:
        raise ValueError(f"No valid rows found in {json_path}")

    parsed_rows.sort(key=lambda r: r["query_gap"])
    return parsed_rows


def plot_entity_c(
    model_rows: Dict[str, List[Dict[str, float]]], output_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc

    ret_key = f"ret_{ENTITY_ID}"
    fig, ax_ret = plt.subplots(1, 1, figsize=(8, 4.8))

    all_ret: List[float] = []

    for model_label, rows in model_rows.items():
        points = [
            (row["query_gap"] / 1000.0, row[ret_key]) for row in rows if ret_key in row
        ]
        xs = [x for x, _ in points]
        ys_ret = [y for _, y in points]
        if ys_ret:
            ax_ret.plot(
                xs,
                ys_ret,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=model_label,
            )
            all_ret.extend(ys_ret)

    ax_ret.set_ylabel("Retrieval score (ret_C)")
    ax_ret.set_xlabel("Query gap (K tokens)")

    ax_ret.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    ax_ret.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    ax_ret.grid(True, which="major", alpha=0.35)

    if all_ret:
        lo = min(all_ret)
        hi = max(all_ret)
        pad = 0.08 * (hi - lo) if hi > lo else 0.5
        ax_ret.set_ylim(lo - pad, hi + pad)

    handles, labels = ax_ret.get_legend_handles_labels()
    if handles:
        ax_ret.legend(
            handles,
            labels,
            loc="upper left",
            frameon=True,
        )

    fig.suptitle("Named Entity Resolution for Entity C", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    model_paths = {
        "GLA": Path(args.gla_json),
        "MSGLA-12": Path(args.msgla12_json),
        "MSGLA-124": Path(args.msgla124_json),
        "MSGLA-1248": Path(args.msgla1248_json),
    }
    model_rows = {label: load_rows(path) for label, path in model_paths.items()}

    output_path = Path(args.output)
    plot_entity_c(model_rows, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
