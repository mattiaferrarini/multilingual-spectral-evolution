"""
Plot RankMe geometry predictor → ECLeKTic transfer performance correlations.

Reads correlation_results.csv from the path in the analysis config and writes three plot types:

  layer_summary/  — one PNG per target metric.
                    x = layer, y = correlation. Lines = predictor. Filled = p < 0.05.
  timeseries/     — one PNG per (layer, target metric).
                    x = training tokens. Lines = predictor.
  heatmaps/       — one PNG per target metric.
                    rows = layers, cols = predictor × corr metric.

Outer loop runs once per ckpt_collapse method found in the results CSV,
saving each method's plots into its own subdirectory.

Usage:
    python eclektic_plot.py --config configs/eclektic_apertus.yaml \\
                             --analysis-config configs/eclektic_correlation_analysis.yaml
"""

import argparse
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from checkpoints import _checkpoint_sort_key
from eclektic_targets import ALL_TARGETS as TARGETS

PREDICTORS = ["abs_diff", "signed_diff", "min_rankme", "norm_asym"]

CORR_TYPES = [
    ("pearson_r",  "pearson_p",  "Pearson r"),
    ("spearman_r", "spearman_p", "Spearman r"),
    ("kendall_r",  "kendall_p",  "Kendall τ"),
]

PRED_COLORS = plt.cm.tab10.colors[:len(PREDICTORS)]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to per-model ECLeKTic config YAML")
    parser.add_argument("--analysis-config", required=True, help="Path to shared correlation analysis config YAML")
    return parser.parse_args()


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(analysis_cfg, model):
    corr = analysis_cfg["correlation"]
    return {k: v.format(model=model) if isinstance(v, str) else v for k, v in corr.items()}


def _token_count(ckpt):
    v = _checkpoint_sort_key(ckpt)
    return v if np.isfinite(v) else None


def _sorted_layers(layer_series):
    return sorted(layer_series.unique(), key=lambda x: int(x.split("_")[1]))


def _plot_layer_summary(corr_df, out_dir, predictors, targets, corr_types):
    """
    One PNG per target metric.
    Layout: len(corr_types) rows × 1 col.
    x = layer index, y = pooled correlation. Lines = predictor. Filled = p < 0.05.
    """
    pred_colors = plt.cm.tab10.colors[:len(predictors)]

    os.makedirs(out_dir, exist_ok=True)
    pooled = corr_df[corr_df["scope"] == "pooled"].copy()
    layers = _sorted_layers(pooled["layer"])
    layer_idx = {l: i for i, l in enumerate(layers)}
    n_layers = len(layers)
    n_rows = len(corr_types)
    x_ticks = list(range(n_layers))
    x_labels = [l.split("_")[1] for l in layers]

    for target in targets:
        target_data = pooled[pooled["normalization"] == target]

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(max(10, 0.4 * n_layers), 4 * n_rows),
            sharex=True,
            squeeze=False,
        )
        fig.suptitle(
            f"RankMe predictability across layers  |  {target}",
            fontsize=12,
        )

        for row_idx, (r_col, p_col, label) in enumerate(corr_types):
            ax = axes[row_idx, 0]

            for pred_idx, pred in enumerate(predictors):
                color = pred_colors[pred_idx]
                s = (
                    target_data[target_data["predictor"] == pred]
                    .copy()
                    .dropna(subset=[r_col])
                )
                if s.empty:
                    continue
                s["x"] = s["layer"].map(layer_idx)
                s = s.sort_values("x")
                ax.plot(s["x"], s[r_col], color=color, linewidth=1.4)
                sig = s[s[p_col] < 0.05]
                nonsig = s[s[p_col] >= 0.05]
                ax.scatter(sig["x"], sig[r_col], color=color, s=30, zorder=3)
                ax.scatter(
                    nonsig["x"], nonsig[r_col],
                    color=color, s=30, facecolors="none", zorder=3,
                )

            ymin, ymax = ax.get_ylim()
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(label, fontsize=10)
            ax.set_xticks(x_ticks)

        axes[-1, 0].set_xticklabels(x_labels, fontsize=7)
        axes[-1, 0].set_xlabel("Layer", fontsize=10)

        pred_handles = [
            mpatches.Patch(color=pred_colors[i], label=p)
            for i, p in enumerate(predictors)
        ]
        axes[0, 0].legend(handles=pred_handles, fontsize=8, loc="upper right")

        fig.text(
            0.5, 0.01,
            "filled = p < 0.05,  hollow = p ≥ 0.05",
            ha="center", fontsize=8, color="gray",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        path = os.path.join(out_dir, f"{target}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _plot_timeseries(corr_df, out_dir, predictors, targets, corr_types):
    """
    One PNG per (layer, target metric).
    Layout: len(corr_types) rows × 1 col.
    x = training tokens (B), y = correlation. Lines = predictor.
    """
    pred_colors = plt.cm.tab10.colors[:len(predictors)]

    os.makedirs(out_dir, exist_ok=True)

    per_ckpt = corr_df[corr_df["scope"] == "per_ckpt"].copy()
    per_ckpt["tokens_B"] = per_ckpt["checkpoint"].map(_token_count)
    per_ckpt = per_ckpt[per_ckpt["tokens_B"].notna()]

    if per_ckpt.empty:
        return

    layers = _sorted_layers(per_ckpt["layer"])
    n_rows = len(corr_types)

    for layer in layers:
        layer_data = per_ckpt[per_ckpt["layer"] == layer]

        for target in targets:
            subset = layer_data[layer_data["normalization"] == target]

            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(10, 4 * n_rows),
                sharex=True,
                squeeze=False,
            )
            fig.suptitle(
                f"RankMe → transfer correlation  |  {target}  |  {layer}",
                fontsize=12,
            )

            for row_idx, (r_col, p_col, label) in enumerate(corr_types):
                ax = axes[row_idx, 0]

                for pred_idx, pred in enumerate(predictors):
                    color = pred_colors[pred_idx]
                    s = (
                        subset[subset["predictor"] == pred]
                        .sort_values("tokens_B")
                        .dropna(subset=[r_col])
                    )
                    if s.empty:
                        continue
                    ax.plot(s["tokens_B"], s[r_col], color=color, linewidth=1.0)
                    sig = s[s[p_col] < 0.05]
                    nonsig = s[s[p_col] >= 0.05]
                    ax.scatter(sig["tokens_B"], sig[r_col], color=color, s=20, marker="o", zorder=3)
                    ax.scatter(
                        nonsig["tokens_B"], nonsig[r_col],
                        color=color, s=20, marker="o", facecolors="none", zorder=3,
                    )

                ymin, ymax = ax.get_ylim()
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_ylim(ymin, ymax)
                ax.set_ylabel(label, fontsize=10)

            pred_handles = [
                mpatches.Patch(color=pred_colors[i], label=pred)
                for i, pred in enumerate(predictors)
            ]
            axes[0, 0].legend(handles=pred_handles, fontsize=7, loc="upper right")
            axes[-1, 0].set_xlabel("Tokens (B)", fontsize=10)

            fig.text(
                0.5, 0.01,
                "filled = p < 0.05,  hollow = p ≥ 0.05",
                ha="center", fontsize=8, color="gray",
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            path = os.path.join(out_dir, f"{layer}_{target}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)


def _plot_heatmap(corr_df, out_dir, predictors, targets, corr_types):
    """
    One PNG per target metric.
    rows = layers (ascending), cols = predictor × corr metric.
    Significant cells (p < 0.05) colored blue→red; non-significant gray.
    """
    os.makedirs(out_dir, exist_ok=True)

    pooled = corr_df[corr_df["scope"] == "pooled"].copy()
    layers = _sorted_layers(pooled["layer"])

    col_defs = [
        (pred, r_col, p_col, f"{pred}\n{name}")
        for pred in predictors
        for r_col, p_col, name in corr_types
    ]
    col_labels = [d[3] for d in col_defs]
    n_cols = len(col_defs)
    n_rows = len(layers)
    layer_labels = [l.replace("layer_", "") for l in layers]

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")

    for target in targets:
        target_data = pooled[pooled["normalization"] == target]

        fig, ax = plt.subplots(
            1, 1,
            figsize=(max(3.5, 0.55 * n_cols) + 1.5, 0.15 * n_rows + 2.5),
            constrained_layout=True,
        )

        r_mat = np.full((n_rows, n_cols), np.nan)
        sig_mat = np.zeros((n_rows, n_cols), dtype=bool)

        for row_i, layer in enumerate(layers):
            layer_rows = target_data[target_data["layer"] == layer]
            for col_j, (pred, r_col, p_col, _) in enumerate(col_defs):
                match = layer_rows[layer_rows["predictor"] == pred]
                if match.empty:
                    continue
                r_val = match[r_col].values[0]
                p_val = match[p_col].values[0]
                if pd.notna(r_val):
                    r_mat[row_i, col_j] = r_val
                if pd.notna(p_val) and p_val < 0.05:
                    sig_mat[row_i, col_j] = True

        display = np.where(sig_mat, r_mat, np.nan)
        im = ax.imshow(display, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        for row_i in range(n_rows):
            for col_j in range(n_cols):
                if sig_mat[row_i, col_j] and not np.isnan(r_mat[row_i, col_j]):
                    ax.text(
                        col_j, row_i,
                        f"{r_mat[row_i, col_j]:.2f}",
                        ha="center", va="center", fontsize=6,
                    )

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(layer_labels, fontsize=7)

        fig.suptitle(f"Pooled correlations  |  {target}", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.6, label="correlation")
        path = os.path.join(out_dir, f"{target}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    analysis_cfg = _load_config(args.analysis_config)

    short_model = cfg["model"]["name"].split("/")[-1]
    paths = _resolve_paths(analysis_cfg, short_model)

    corr_df = pd.read_csv(paths["results_csv"])

    collapse_values = (
        sorted(corr_df["ckpt_collapse"].unique().tolist())
        if "ckpt_collapse" in corr_df.columns
        else ["none"]
    )

    for method in collapse_values:
        if "ckpt_collapse" in corr_df.columns:
            method_df = corr_df[corr_df["ckpt_collapse"] == method].copy()
        else:
            method_df = corr_df

        active_predictors = [p for p in PREDICTORS if p in method_df["predictor"].unique()]
        active_targets = [t for t in TARGETS if t in method_df["normalization"].unique()]
        active_corr_types = [(r, p, lbl) for r, p, lbl in CORR_TYPES if r in method_df.columns]

        layer_summary_dir = os.path.join(paths["layer_summary_dir"], method)
        timeseries_dir    = os.path.join(paths["timeseries_dir"],    method)
        heatmaps_dir      = os.path.join(paths["heatmaps_dir"],      method)

        _plot_layer_summary(method_df, layer_summary_dir,
                            active_predictors, active_targets, active_corr_types)
        _plot_timeseries(method_df, timeseries_dir,
                         active_predictors, active_targets, active_corr_types)
        _plot_heatmap(method_df, heatmaps_dir,
                      active_predictors, active_targets, active_corr_types)

        print(f"[{method}] layer_summary → {layer_summary_dir}")
        print(f"[{method}] timeseries    → {timeseries_dir}")
        print(f"[{method}] heatmaps      → {heatmaps_dir}")


if __name__ == "__main__":
    main()
