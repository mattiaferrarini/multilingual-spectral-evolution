"""
Visualization script for geometry analysis results.

Generates three plot types from a metrics CSV:
  - training_curves : metric vs checkpoint (one line per language)
  - layer_profiles  : metric vs layer     (one line per checkpoint)
  - heatmaps        : checkpoint x layer  (colored by metric)

Usage examples:
  # All plots, all data
  python geometry_analysis/visualize.py --csv results/fuxi.csv

  # Only training curves, RankMe only, last-token aggregation
  python geometry_analysis/visualize.py --csv results/fuxi.csv \
      --plot-types training_curves --metrics rankme --aggregations last

  # Heatmaps for two languages, all metrics
  python geometry_analysis/visualize.py --csv results/fuxi.csv \
      --plot-types heatmaps --languages English French
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

METRIC_LABELS = {
    "rankme": "RankMe",
    "pr": "Participation Ratio",
    "alpha_req": "AlphaReQ",
    "top_5_var": "Top-5 Variance",
    "top_10_var": "Top-10 Variance",
    "top_20_var": "Top-20 Variance",
    "top_50_var": "Top-50 Variance",
    "top_100_var": "Top-100 Variance",
}


# ── Checkpoint ordering ───────────────────────────────────────────────────────

def _ckpt_key(name):
    if str(name).lower() == "main":
        return float("inf")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([BT]?)$", str(name), re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    # Apertus format: step{N}-tokens{M}[BT]
    m = re.match(r"step\d+-tokens(\d+)([BT])", str(name), re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    return float("inf") - 1


def sort_checkpoints(checkpoints):
    return sorted(checkpoints, key=_ckpt_key)


def layer_num(layer_name):
    m = re.search(r"(\d+)", str(layer_name))
    return int(m.group(1)) if m else 0


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _language_colors(languages):
    palette = sns.color_palette("tab10", n_colors=max(len(languages), 1))
    return {lang: palette[i] for i, lang in enumerate(languages)}


def _checkpoint_colors(checkpoints):
    cmap = plt.get_cmap("viridis")
    n = max(len(checkpoints), 1)
    return {ckpt: cmap(i / (n - 1) if n > 1 else 0.5) for i, ckpt in enumerate(checkpoints)}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ── Plot type 1: training curves ─────────────────────────────────────────────

def plot_training_curves(df, metrics, languages, layers, aggregations, output_dir, model_label):
    checkpoints = sort_checkpoints(df["checkpoint"].unique())
    x = range(len(checkpoints))
    lang_colors = _language_colors(languages)

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        for layer in layers:
            for agg in aggregations:
                subset = df[(df["layer"] == layer) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(10, 5))
                for lang in languages:
                    vals = (
                        subset[subset["dataset"] == lang]
                        .set_index("checkpoint")
                        .reindex(checkpoints)[metric]
                    )
                    ax.plot(x, vals, marker="o", label=lang,
                            color=lang_colors[lang], linewidth=2, markersize=5)

                ax.set_xticks(list(x))
                ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=9)
                ax.set_xlabel("Checkpoint (tokens seen)")
                ax.set_ylabel(metric_label)
                title = f"{metric_label} over training\n{layer} | agg={agg}"
                if model_label:
                    title = f"[{model_label}] " + title
                ax.set_title(title)
                ax.legend(title="Language", bbox_to_anchor=(1.02, 1), loc="upper left")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                fname = f"training_curve_{metric}_{layer}_{agg}.png"
                _save(fig, os.path.join(output_dir, "training_curves", fname))


# ── Plot type 2: layer profiles ───────────────────────────────────────────────

def plot_layer_profiles(df, metrics, languages, layers, aggregations, output_dir, model_label):
    checkpoints = sort_checkpoints(df["checkpoint"].unique())
    ckpt_colors = _checkpoint_colors(checkpoints)
    layer_nums = sorted(layers, key=layer_num)
    x = [layer_num(l) for l in layer_nums]

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        for lang in languages:
            for agg in aggregations:
                subset = df[(df["dataset"] == lang) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(10, 5))
                for ckpt in checkpoints:
                    vals = (
                        subset[subset["checkpoint"] == ckpt]
                        .set_index("layer")
                        .reindex(layer_nums)[metric]
                    )
                    ax.plot(x, vals, marker="o", label=ckpt,
                            color=ckpt_colors[ckpt], linewidth=2, markersize=5)

                ax.set_xlabel("Layer")
                ax.set_ylabel(metric_label)
                title = f"{metric_label} by layer — {lang} | agg={agg}"
                if model_label:
                    title = f"[{model_label}] " + title
                ax.set_title(title)
                ax.legend(title="Checkpoint", bbox_to_anchor=(1.02, 1), loc="upper left")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                fname = f"layer_profile_{metric}_{lang}_{agg}.png"
                _save(fig, os.path.join(output_dir, "layer_profiles", fname))


# ── Plot type 3: heatmaps ─────────────────────────────────────────────────────

def plot_heatmaps(df, metrics, languages, layers, aggregations, output_dir, model_label):
    checkpoints = sort_checkpoints(df["checkpoint"].unique())
    layer_nums_sorted = sorted(layers, key=layer_num)
    x_labels = [layer_num(l) for l in layer_nums_sorted]

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        for lang in languages:
            for agg in aggregations:
                subset = df[(df["dataset"] == lang) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                matrix = np.full((len(checkpoints), len(layer_nums_sorted)), np.nan)
                for r, ckpt in enumerate(checkpoints):
                    for c, layer in enumerate(layer_nums_sorted):
                        row = subset[(subset["checkpoint"] == ckpt) & (subset["layer"] == layer)]
                        if not row.empty:
                            matrix[r, c] = row[metric].values[0]

                n_cells = len(checkpoints) * len(layer_nums_sorted)
                fig_w = max(8, len(layer_nums_sorted) * 0.7)
                fig_h = max(4, len(checkpoints) * 0.5)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                sns.heatmap(
                    matrix,
                    xticklabels=x_labels,
                    yticklabels=checkpoints,
                    cmap="viridis",
                    annot=n_cells <= 120,
                    fmt=".1f",
                    ax=ax,
                    cbar_kws={"label": metric_label},
                )
                ax.set_xlabel("Layer")
                ax.set_ylabel("Checkpoint")
                title = f"{metric_label} heatmap — {lang} | agg={agg}"
                if model_label:
                    title = f"[{model_label}] " + title
                ax.set_title(title)
                plt.tight_layout()

                fname = f"heatmap_{metric}_{lang}_{agg}.png"
                _save(fig, os.path.join(output_dir, "heatmaps", fname))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate geometry analysis visualizations from a metrics CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv", required=True, help="Path to metrics CSV file.")
    p.add_argument("--output-dir", default="plots",
                   help="Root directory for output PNG files (default: plots/).")
    p.add_argument("--model", default="",
                   help="Model name shown in plot titles (e.g. FuxiTranyu-8B).")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Metrics to plot. Default: all columns in CSV.")
    p.add_argument("--languages", nargs="+", default=None,
                   help="Languages to include. Default: all in CSV.")
    p.add_argument("--layers", nargs="+", default=None,
                   help="Layers to include (e.g. layer_0 layer_12). Default: all.")
    p.add_argument("--aggregations", nargs="+", default=None,
                   help="Aggregations to include (last mean). Default: all.")
    p.add_argument(
        "--plot-types", nargs="+",
        default=["training_curves", "layer_profiles", "heatmaps"],
        choices=["training_curves", "layer_profiles", "heatmaps"],
        help="Which plot types to generate (default: all three).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    metric_cols = [c for c in df.columns
                   if c not in {"checkpoint", "dataset", "layer", "aggregation"}]
    metrics      = args.metrics      or metric_cols
    languages    = args.languages    or sorted(df["dataset"].unique())
    layers       = args.layers       or sorted(df["layer"].unique(), key=layer_num)
    aggregations = args.aggregations or sorted(df["aggregation"].unique())

    print(f"Metrics:      {metrics}")
    print(f"Languages:    {languages}")
    print(f"Layers:       {layers}")
    print(f"Aggregations: {aggregations}")
    print(f"Plot types:   {args.plot_types}")
    print(f"Output dir:   {args.output_dir}/\n")

    kw = dict(
        metrics=metrics, languages=languages, layers=layers,
        aggregations=aggregations, output_dir=args.output_dir,
        model_label=args.model,
    )

    if "training_curves" in args.plot_types:
        print("[1/3] Training curves...")
        plot_training_curves(df, **kw)

    if "layer_profiles" in args.plot_types:
        print("\n[2/3] Layer profiles...")
        plot_layer_profiles(df, **kw)

    if "heatmaps" in args.plot_types:
        print("\n[3/3] Heatmaps...")
        plot_heatmaps(df, **kw)

    print(f"\nDone — all plots saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
