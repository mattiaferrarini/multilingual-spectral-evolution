"""
Q1: Where (in depth) and when (in tokens) do languages stratify the most?

Builds a heatmap of cross-language std(metric) over (layer x checkpoint).
A bright cell means the 14 languages disagree strongly on that metric at
that layer/checkpoint — i.e. the model's geometry there is language-specific.

Companion CV heatmap normalises by the mean to remove the trivial "high
metric magnitude => high std" effect.

Usage:
  python visualization/visualize_stratification.py \
      --csv results/fuxi_fine_wiki.csv --experiment fuxi_fine_wiki \
      --model "FuxiTranyu-8B"

  # Also re-run after dropping suspected tokenization outliers:
  python visualization/visualize_stratification.py \
      --csv results/fuxi_fine_wiki.csv --experiment fuxi_fine_wiki \
      --model "FuxiTranyu-8B" --drop-outliers German Japanese
"""

import argparse
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", font_scale=1.0)


def _ckpt_key(name):
    """Sort checkpoint labels by token count (in billions)."""
    s = str(name)
    if s.lower() == "main":
        return float("inf")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([BT]?)$", s, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    m = re.match(r"step\d+-tokens(\d+)([BT])", s, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    return float("inf") - 1


def _layer_num(layer_name):
    m = re.search(r"(\d+)", str(layer_name))
    return int(m.group(1)) if m else 0


def _ckpt_tick(name):
    s = str(name)
    if s.lower() == "main":
        return "main"
    m = re.match(r"step\d+-tokens(\d+)([BT])", s, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s


def _compute_tables(df, metric, aggregation, exclude=None):
    sub = df[df["aggregation"] == aggregation].copy()
    if exclude:
        sub = sub[~sub["dataset"].isin(exclude)]

    # rows = (checkpoint, layer), cols = dataset
    pivot = sub.pivot_table(index=["checkpoint", "layer"], columns="dataset", values=metric)
    n_langs = pivot.notna().sum(axis=1)
    std = pivot.std(axis=1, ddof=0)
    mean = pivot.mean(axis=1)
    cv = std / mean.replace(0, np.nan)

    def _to_matrix(series):
        wide = series.reset_index().pivot(index="layer", columns="checkpoint", values=0
                                          if series.name is None else series.name)
        ckpts = sorted(wide.columns, key=_ckpt_key)
        layers = sorted(wide.index, key=_layer_num)
        return wide.reindex(index=layers, columns=ckpts), layers, ckpts

    std.name = "std"
    cv.name = "cv"
    mean.name = "mean"
    n_langs.name = "n"

    return (_to_matrix(std), _to_matrix(cv), _to_matrix(mean), _to_matrix(n_langs))


def _plot_heatmap(matrix, layers, ckpts, title, cbar_label, output_path, cmap="magma", vmax=None):
    fig_w = max(10, len(ckpts) * 0.22 + 4)
    fig_h = max(6, len(layers) * 0.22 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix.values,
        xticklabels=[_ckpt_tick(c) for c in ckpts],
        yticklabels=[_layer_num(l) for l in layers],
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": cbar_label},
        vmax=vmax,
        linewidths=0,
    )
    ax.set_xlabel("Checkpoint (tokens)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_path}")


def _summary_stats(std_matrix, layers, ckpts, top_n=5):
    arr = std_matrix.values
    flat = []
    for i, l in enumerate(layers):
        for j, c in enumerate(ckpts):
            v = arr[i, j]
            if np.isfinite(v):
                flat.append((v, _layer_num(l), _ckpt_tick(c)))
    flat.sort(reverse=True)
    print(f"  top {top_n} std cells:")
    for v, l, c in flat[:top_n]:
        print(f"    layer={l:>2}  ckpt={c:>10}  std={v:.2f}")

    layer_means = np.nanmean(arr, axis=1)
    ckpt_means = np.nanmean(arr, axis=0)
    peak_layer_idx = int(np.nanargmax(layer_means))
    peak_ckpt_idx = int(np.nanargmax(ckpt_means))
    print(f"  layer with highest avg std: layer_{_layer_num(layers[peak_layer_idx])}  "
          f"(mean std={layer_means[peak_layer_idx]:.2f})")
    print(f"  checkpoint with highest avg std: {_ckpt_tick(ckpts[peak_ckpt_idx])}  "
          f"(mean std={ckpt_means[peak_ckpt_idx]:.2f})")


def run(df, metric, aggregation, output_dir, experiment, model_label, exclude=None, tag=""):
    print(f"\n=== {metric} | agg={aggregation} | exclude={exclude or 'none'} ===")
    (std_m, std_l, std_c), (cv_m, cv_l, cv_c), (_, _, _), (n_m, _, _) = _compute_tables(
        df, metric, aggregation, exclude=exclude
    )
    n_langs_used = int(np.nanmedian(n_m.values))
    print(f"  languages used per cell (median): {n_langs_used}")

    suffix = (f"_{tag}" if tag else "")
    base_title = f"std({metric}) across languages | agg={aggregation}"
    if model_label:
        base_title = f"[{model_label}] " + base_title
    if exclude:
        base_title += f"  (excluded: {', '.join(exclude)})"

    base_dir = os.path.join(output_dir, experiment, "stratification")
    _plot_heatmap(
        std_m, std_l, std_c, base_title,
        cbar_label=f"std({metric}) across {n_langs_used} languages",
        output_path=os.path.join(base_dir, f"stratification_std_{metric}_{aggregation}{suffix}.png"),
        cmap="magma",
    )
    _plot_heatmap(
        cv_m, cv_l, cv_c,
        title=base_title.replace(f"std({metric})", f"CV({metric})"),
        cbar_label=f"CV ({metric}) across languages",
        output_path=os.path.join(base_dir, f"stratification_cv_{metric}_{aggregation}{suffix}.png"),
        cmap="magma",
    )

    _summary_stats(std_m, std_l, std_c)
    return std_m, cv_m


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--experiment", required=True, help="subfolder under output-dir (e.g. fuxi_fine_wiki)")
    p.add_argument("--metric", default="rankme")
    p.add_argument("--aggregation", default="last")
    p.add_argument("--output-dir", default="plots")
    p.add_argument("--model", default="")
    p.add_argument("--drop-outliers", nargs="+", default=None,
                   help="languages to exclude in a companion plot (e.g. German Japanese)")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")
    langs = sorted(df["dataset"].unique())
    layers = sorted(df["layer"].unique(), key=_layer_num)
    ckpts = sorted(df["checkpoint"].unique(), key=_ckpt_key)
    print(f"  {len(langs)} languages | {len(layers)} layers | {len(ckpts)} checkpoints")

    run(df, args.metric, args.aggregation, args.output_dir, args.experiment, args.model)
    if args.drop_outliers:
        run(df, args.metric, args.aggregation, args.output_dir, args.experiment, args.model,
            exclude=args.drop_outliers, tag="no_outliers")


if __name__ == "__main__":
    main()
