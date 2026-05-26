"""
Plot RankMe → future transfer correlation results produced by xnli_correlate_ahead.py.

Reads correlation_results.csv from the path in the config and writes three plot types:

  lag_summary/  — PRIMARY: one PNG per (layer, normalization).
                  x = T (lag), y = correlation. Shows how far ahead RankMe can predict.
  timeseries/   — one PNG per (layer, normalization). x = training tokens.
                  Color = predictor, line style = T value.
  heatmaps/     — one PNG per (layer, normalization). Rows = T, cols = predictor × metric.

Usage:
    python xnli_plot_ahead.py --config configs/xnli_apertus.yaml \
                               --analysis-config configs/xnli_correlation_ahead_analysis.yaml
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

PREDICTORS = ["abs_diff", "signed_diff", "min_rankme", "norm_asym"]
NORMALIZATIONS = ["row_norm", "col_norm"]

CORR_TYPES = [
    ("pearson_r",  "pearson_p",  "Pearson r"),
    ("spearman_r", "spearman_p", "Spearman r"),
    ("kendall_r",  "kendall_p",  "Kendall τ"),
]

T_LINESTYLES = ["-", "--", ":", "-."]
PRED_COLORS = plt.cm.tab10.colors[:len(PREDICTORS)]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to per-model XNLI config YAML")
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


def _plot_lag_summary(corr_df, lag_dir, k_values, t_values):
    """
    Primary plot: how does correlation change as prediction horizon T grows?

    One PNG per (layer, normalization).
    Layout: rows = 3 corr types, cols = k values.
    x = T, y = correlation. Lines = predictors. Filled = p < 0.05, hollow = p >= 0.05.
    """
    os.makedirs(lag_dir, exist_ok=True)
    pooled = corr_df[corr_df["scope"] == "pooled"].copy()
    layers = _sorted_layers(pooled["layer"])
    k_sorted = sorted(k_values)

    for layer in layers:
        layer_data = pooled[pooled["layer"] == layer]

        for norm in NORMALIZATIONS:
            fig, axes = plt.subplots(
                3, len(k_sorted),
                figsize=(4 * len(k_sorted), 9),
                sharey="row",
                squeeze=False,
            )
            fig.suptitle(
                f"RankMe predictability vs lag T  |  {norm}  |  {layer}",
                fontsize=12,
            )

            for col_idx, k in enumerate(k_sorted):
                subset = layer_data[
                    (layer_data["normalization"] == norm) & (layer_data["k"] == k)
                ]

                for row_idx, (r_col, p_col, label) in enumerate(CORR_TYPES):
                    ax = axes[row_idx, col_idx]

                    for pred_idx, pred in enumerate(PREDICTORS):
                        s = (
                            subset[subset["predictor"] == pred]
                            .sort_values("t")
                            .dropna(subset=[r_col])
                        )
                        if s.empty:
                            continue
                        color = PRED_COLORS[pred_idx]
                        ax.plot(s["t"], s[r_col], color=color, linewidth=1.5, label=pred)
                        sig = s[s[p_col] < 0.05]
                        nonsig = s[s[p_col] >= 0.05]
                        ax.scatter(sig["t"], sig[r_col], color=color, s=40, zorder=3)
                        ax.scatter(
                            nonsig["t"], nonsig[r_col],
                            color=color, s=40, facecolors="none", zorder=3,
                        )

                    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                    ax.set_xticks(t_values)
                    if row_idx == 0:
                        ax.set_title(f"k={k}", fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(label, fontsize=10)
                    if row_idx == 0 and col_idx == len(k_sorted) - 1:
                        ax.legend(fontsize=8)

            for col_idx in range(len(k_sorted)):
                axes[-1, col_idx].set_xlabel("T (steps ahead)", fontsize=10)

            fig.text(
                0.5, 0.01,
                "filled = p < 0.05,  hollow = p ≥ 0.05",
                ha="center", fontsize=8, color="gray",
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            path = os.path.join(lag_dir, f"{layer}_{norm}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)


def _plot_timeseries(corr_df, timeseries_dir, k_values, t_values):
    """
    Timeseries of per-checkpoint correlations over training.

    One PNG per (layer, normalization).
    Layout: rows = 3 corr types, cols = k values.
    Color = predictor. Line style = T value. Filled/hollow markers for significance.
    """
    os.makedirs(timeseries_dir, exist_ok=True)

    per_ckpt = corr_df[corr_df["scope"] == "per_ckpt"].copy()
    per_ckpt["tokens_B"] = per_ckpt["checkpoint"].map(_token_count)
    per_ckpt = per_ckpt[per_ckpt["tokens_B"].notna()]

    layers = _sorted_layers(per_ckpt["layer"])
    k_sorted = sorted(k_values)
    t_sorted = sorted(t_values)
    t_ls = {t: T_LINESTYLES[i % len(T_LINESTYLES)] for i, t in enumerate(t_sorted)}

    for layer in layers:
        layer_data = per_ckpt[per_ckpt["layer"] == layer]

        for norm in NORMALIZATIONS:
            fig, axes = plt.subplots(
                3, len(k_sorted),
                figsize=(4 * len(k_sorted), 9),
                sharex=True,
                squeeze=False,
            )
            fig.suptitle(
                f"RankMe → transfer correlation  |  {norm}  |  {layer}",
                fontsize=12,
            )

            for col_idx, k in enumerate(k_sorted):
                subset = layer_data[
                    (layer_data["normalization"] == norm) & (layer_data["k"] == k)
                ]

                for row_idx, (r_col, p_col, label) in enumerate(CORR_TYPES):
                    ax = axes[row_idx, col_idx]

                    for pred_idx, pred in enumerate(PREDICTORS):
                        color = PRED_COLORS[pred_idx]
                        for t in t_sorted:
                            s = (
                                subset[
                                    (subset["predictor"] == pred) & (subset["t"] == t)
                                ]
                                .sort_values("tokens_B")
                                .dropna(subset=[r_col])
                            )
                            if s.empty:
                                continue
                            ls = t_ls[t]
                            ax.plot(s["tokens_B"], s[r_col], color=color, linestyle=ls, linewidth=1.0)
                            sig = s[s[p_col] < 0.05]
                            nonsig = s[s[p_col] >= 0.05]
                            ax.scatter(sig["tokens_B"], sig[r_col], color=color, s=20, marker="o", zorder=3)
                            ax.scatter(
                                nonsig["tokens_B"], nonsig[r_col],
                                color=color, s=20, marker="o", facecolors="none", zorder=3,
                            )

                    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                    if row_idx == 0:
                        ax.set_title(f"k={k}", fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(label, fontsize=10)

            # Two-part legend on top-right subplot
            legend_ax = axes[0, -1]
            pred_handles = [
                mpatches.Patch(color=PRED_COLORS[i], label=pred)
                for i, pred in enumerate(PREDICTORS)
            ]
            t_handles = [
                mlines.Line2D([], [], color="black", linestyle=t_ls[t], linewidth=1.2, label=f"T={t}")
                for t in t_sorted
            ]
            legend_ax.legend(
                handles=pred_handles + t_handles,
                fontsize=7, ncol=1, loc="upper right",
            )

            for col_idx in range(len(k_sorted)):
                axes[-1, col_idx].set_xlabel("Tokens (B)", fontsize=10)

            fig.text(
                0.5, 0.01,
                "filled = p < 0.05,  hollow = p ≥ 0.05",
                ha="center", fontsize=8, color="gray",
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            path = os.path.join(timeseries_dir, f"{layer}_{norm}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)


def _plot_heatmap_by_t(corr_df, heatmaps_dir, k_values, t_values):
    """
    Compact heatmap: rows = T values, cols = predictor × corr metric (12 cols).

    One PNG per (layer, normalization). k values as side-by-side heatmaps.
    Significant cells (p < 0.05) colored blue→red; non-significant gray.
    """
    os.makedirs(heatmaps_dir, exist_ok=True)

    pooled = corr_df[corr_df["scope"] == "pooled"].copy()
    layers = _sorted_layers(pooled["layer"])
    k_sorted = sorted(k_values)
    t_sorted = sorted(t_values)

    col_defs = [
        (pred, r_col, p_col, f"{pred}\n{name}")
        for pred in PREDICTORS
        for r_col, p_col, name in CORR_TYPES
    ]
    col_labels = [d[3] for d in col_defs]
    n_cols = len(col_defs)
    n_rows = len(t_sorted)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")

    for layer in layers:
        layer_data = pooled[pooled["layer"] == layer]

        for norm in NORMALIZATIONS:
            norm_data = layer_data[layer_data["normalization"] == norm]

            fig, axes = plt.subplots(
                1, len(k_sorted),
                figsize=(3.5 * len(k_sorted), 0.6 * n_rows + 2.5),
                squeeze=False,
            )

            ims = []
            for col_idx, k in enumerate(k_sorted):
                ax = axes[0, col_idx]
                k_data = norm_data[norm_data["k"] == k]

                r_mat = np.full((n_rows, n_cols), np.nan)
                sig_mat = np.zeros((n_rows, n_cols), dtype=bool)

                for row_i, t in enumerate(t_sorted):
                    t_rows = k_data[k_data["t"] == t]
                    for col_j, (pred, r_col, p_col, _) in enumerate(col_defs):
                        match = t_rows[t_rows["predictor"] == pred]
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
                ims.append(im)

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
                ax.set_yticklabels([f"T={t}" for t in t_sorted] if col_idx == 0 else [], fontsize=8)
                ax.set_title(f"k={k}", fontsize=10)

            fig.suptitle(f"Pooled correlations by lag T  |  {norm}  |  {layer}", fontsize=12)
            fig.colorbar(ims[-1], ax=axes[0, :], shrink=0.6, label="correlation")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            path = os.path.join(heatmaps_dir, f"{layer}_{norm}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    analysis_cfg = _load_config(args.analysis_config)

    short_model = cfg["model"]["name"].split("/")[-1]
    paths = _resolve_paths(analysis_cfg, short_model)

    corr_df = pd.read_csv(paths["results_csv"])

    t_values = sorted(corr_df["t"].unique().tolist())
    k_values = sorted(corr_df["k"].unique().tolist())

    _plot_lag_summary(corr_df, paths["lag_dir"], k_values, t_values)
    print(f"Saved lag summary plots to {paths['lag_dir']}")

    _plot_timeseries(corr_df, paths["timeseries_dir"], k_values, t_values)
    print(f"Saved timeseries plots to {paths['timeseries_dir']}")

    _plot_heatmap_by_t(corr_df, paths["heatmaps_dir"], k_values, t_values)
    print(f"Saved heatmaps to {paths['heatmaps_dir']}")


if __name__ == "__main__":
    main()
