"""
Plot RankMe → transfer correlation results produced by xnli_correlate_within_ckpt.py.

Reads pairs.csv and correlation_results.csv from the paths in the config and writes:
  timeseries/ — one PNG per (layer, normalization); subplots are correlation metrics × k values
  heatmaps/   — one PNG per normalization; rows=layers, cols=predictor×metric, one heatmap per k

Usage:
    python xnli_plot_within_ckpt.py --config configs/xnli_apertus.yaml \
                                     --analysis-config configs/xnli_correlation_analysis.yaml
"""

import argparse
import os

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


def _plot_timeseries(corr_df, timeseries_dir, k_values):
    """
    One PNG per (layer, normalization).
    Layout: 3 rows (Pearson/Spearman/Kendall) × len(k_values) columns.
    Filled markers = p < 0.05, hollow = p >= 0.05.
    """
    os.makedirs(timeseries_dir, exist_ok=True)

    per_ckpt = corr_df[corr_df["scope"] == "per_ckpt"].copy()
    per_ckpt["tokens_B"] = per_ckpt["checkpoint"].map(_token_count)
    per_ckpt = per_ckpt[per_ckpt["tokens_B"].notna()]

    layers = _sorted_layers(per_ckpt["layer"])
    k_sorted = sorted(k_values)

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

                    for pred in PREDICTORS:
                        s = (
                            subset[subset["predictor"] == pred]
                            .sort_values("tokens_B")
                            .dropna(subset=[r_col])
                        )
                        if s.empty:
                            continue
                        line, = ax.plot(s["tokens_B"], s[r_col], linewidth=1.2, label=pred)
                        color = line.get_color()
                        sig = s[s[p_col] < 0.05]
                        nonsig = s[s[p_col] >= 0.05]
                        ax.scatter(sig["tokens_B"], sig[r_col], color=color, s=30, zorder=3)
                        ax.scatter(
                            nonsig["tokens_B"], nonsig[r_col],
                            color=color, s=30, facecolors="none", zorder=3,
                        )

                    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                    if row_idx == 0:
                        ax.set_title(f"k={k}", fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(label, fontsize=10)
                    if row_idx == 0 and col_idx == len(k_sorted) - 1:
                        ax.legend(fontsize=8, ncol=1)

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


def _plot_heatmaps(corr_df, heatmaps_dir, k_values):
    """
    One PNG per normalization.
    Each PNG: len(k_values) heatmaps side by side.
    Rows = layers (ascending), cols = predictor × correlation metric (12 cols).
    Significant cells (p < 0.05) are colored blue→red and show the value.
    Non-significant cells are gray with no label.
    """
    os.makedirs(heatmaps_dir, exist_ok=True)

    pooled = corr_df[corr_df["scope"] == "pooled"].copy()
    layers = _sorted_layers(pooled["layer"])
    k_sorted = sorted(k_values)

    # Column definitions: (predictor, r_col, p_col, short_label)
    col_defs = [
        (pred, r_col, p_col, f"{pred}\n{name}")
        for pred in PREDICTORS
        for r_col, p_col, name in CORR_TYPES
    ]
    col_labels = [d[3] for d in col_defs]
    n_cols = len(col_defs)
    n_rows = len(layers)
    layer_labels = [l.replace("layer_", "") for l in layers]

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")

    for norm in NORMALIZATIONS:
        norm_data = pooled[pooled["normalization"] == norm]

        fig, axes = plt.subplots(
            1, len(k_sorted),
            figsize=(3.5 * len(k_sorted), 0.35 * n_rows + 2.5),
            squeeze=False,
        )

        ims = []
        for col_idx, k in enumerate(k_sorted):
            ax = axes[0, col_idx]
            k_data = norm_data[norm_data["k"] == k]

            r_mat = np.full((n_rows, n_cols), np.nan)
            sig_mat = np.zeros((n_rows, n_cols), dtype=bool)

            for row_i, layer in enumerate(layers):
                layer_rows = k_data[k_data["layer"] == layer]
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
            ax.set_yticklabels(layer_labels if col_idx == 0 else [], fontsize=7)
            ax.set_title(f"k={k}", fontsize=10)

        fig.suptitle(f"Pooled correlations  |  {norm}", fontsize=12)
        fig.colorbar(ims[-1], ax=axes[0, :], shrink=0.6, label="correlation")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(heatmaps_dir, f"{norm}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    analysis_cfg = _load_config(args.analysis_config)

    short_model = cfg["model"]["name"].split("/")[-1]
    paths = _resolve_paths(analysis_cfg, short_model)

    pairs_df = pd.read_csv(paths["pairs_csv"])
    corr_df = pd.read_csv(paths["results_csv"])

    k_values = sorted(pairs_df["k"].unique().tolist())

    _plot_timeseries(corr_df, paths["timeseries_dir"], k_values)
    print(f"Saved timeseries plots to {paths['timeseries_dir']}")

    _plot_heatmaps(corr_df, paths["heatmaps_dir"], k_values)
    print(f"Saved heatmaps to {paths['heatmaps_dir']}")


if __name__ == "__main__":
    main()
