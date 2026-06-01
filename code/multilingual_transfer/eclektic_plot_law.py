"""
Plot scaling-law predictor → ECLeKTic transfer performance correlations.

Reads correlation_results.csv from the analysis config and writes two plot types:

  lollipop/  — PRIMARY: one PNG per target metric.
               Rows = corr types; x = predictor; y = pooled correlation.
               Filled circles = p < 0.05. Predictors grouped and coloured by
               scaling-law parameter family.
  heatmaps/  — one PNG per corr type.
               Rows = 48 law predictors (grouped by parameter);
               cols = target metrics.

Usage:
    python eclektic_plot_law.py --config configs/eclektic_law_fuxi.yaml \\
                                 --analysis-config configs/eclektic_law_correlation_analysis.yaml
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from geometry_predictors import LAW_CURVE_PARAMS, LAW_PHASE1_PARAMS, LAW_PREDICTORS

LAW_PARAMS = LAW_PHASE1_PARAMS + LAW_CURVE_PARAMS   # 6 families
LAW_SUFFIXES = [
    "abs_diff", "signed_diff", "min", "max",
    "norm_asym", "abs_ratio", "signed_ratio", "log_ratio",
]
N_SUFFIX = len(LAW_SUFFIXES)

SUFFIX_SHORT = {
    "abs_diff":     "abs_d",
    "signed_diff":  "sgn_d",
    "min":          "min",
    "max":          "max",
    "norm_asym":    "n_asy",
    "abs_ratio":    "abs_r",
    "signed_ratio": "sgn_r",
    "log_ratio":    "log_r",
}
PARAM_SHORT = {
    "alpha":               "α",
    "A":                   "A",
    "drop_to_min":         "drop",
    "recovery":            "rec",
    "drop_minus_recovery": "d-r",
    "drop_over_recovery":  "d/r",
}

GROUP_COLORS = list(plt.cm.tab10.colors[:len(LAW_PARAMS)])

CORR_TYPES = [
    ("pearson_r",  "pearson_p",  "Pearson r"),
    ("spearman_r", "spearman_p", "Spearman r"),
    ("kendall_r",  "kendall_p",  "Kendall τ"),
]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-config", required=True)
    return parser.parse_args()


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(analysis_cfg, model):
    corr = analysis_cfg["correlation"]
    return {k: v.format(model=model) if isinstance(v, str) else v for k, v in corr.items()}


def _pred_color(pred_index):
    return GROUP_COLORS[pred_index // N_SUFFIX]


def _plot_lollipop(corr_df, out_dir, targets, corr_types):
    """
    One PNG per target metric.
    Rows = corr types; x = predictor (48, grouped); y = pooled correlation.
    Stems coloured by parameter family. Filled = p < 0.05.
    """
    os.makedirs(out_dir, exist_ok=True)
    pooled = corr_df[corr_df["scope"] == "pooled"].copy()
    n_preds = len(LAW_PREDICTORS)
    n_groups = len(LAW_PARAMS)
    pred_idx = {p: i for i, p in enumerate(LAW_PREDICTORS)}
    suf_xlabels = [SUFFIX_SHORT[LAW_SUFFIXES[i % N_SUFFIX]] for i in range(n_preds)]

    for target in targets:
        target_data = pooled[pooled["normalization"] == target]
        n_rows = len(corr_types)

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(max(14, 0.28 * n_preds), 3.5 * n_rows),
            sharex=True,
            squeeze=False,
        )
        fig.suptitle(
            f"Law predictor → ECLeKTic  |  {target}",
            fontsize=12,
        )

        for row_idx, (r_col, p_col, label) in enumerate(corr_types):
            ax = axes[row_idx, 0]
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=0)

            for g in range(1, n_groups):
                ax.axvline(g * N_SUFFIX - 0.5, color="lightgray", linewidth=0.8,
                           linestyle=":", zorder=0)

            for pred in LAW_PREDICTORS:
                xi = pred_idx[pred]
                color = _pred_color(xi)
                row = target_data[target_data["predictor"] == pred]
                if row.empty:
                    continue
                r_val = row[r_col].values[0]
                p_val = row[p_col].values[0] if p_col in row.columns else np.nan
                if pd.isna(r_val):
                    continue
                ax.vlines(xi, 0, r_val, colors=color, linewidth=1.2, zorder=1)
                sig = pd.notna(p_val) and p_val < 0.05
                if sig:
                    ax.scatter(xi, r_val, color=color, s=25, zorder=3)
                else:
                    ax.scatter(xi, r_val, s=25, facecolors="none",
                               edgecolors=color, linewidths=1.0, zorder=3)

            ax.set_ylabel(label, fontsize=10)
            ax.set_xlim(-0.8, n_preds - 0.2)
            for g, param in enumerate(LAW_PARAMS):
                x_center = g * N_SUFFIX + (N_SUFFIX - 1) / 2
                ax.text(
                    x_center, ax.get_ylim()[1],
                    PARAM_SHORT.get(param, param),
                    ha="center", va="bottom", fontsize=8,
                    color=GROUP_COLORS[g], fontweight="bold",
                )

        ax_last = axes[-1, 0]
        ax_last.set_xticks(range(n_preds))
        ax_last.set_xticklabels(suf_xlabels, rotation=90, fontsize=7)
        ax_last.set_xlabel("Comparison formula  (grouped by parameter family)", fontsize=10)

        handles = [
            mpatches.Patch(color=GROUP_COLORS[g],
                           label=PARAM_SHORT.get(LAW_PARAMS[g], LAW_PARAMS[g]))
            for g in range(n_groups)
        ]
        axes[0, 0].legend(handles=handles, fontsize=8, loc="upper right",
                          title="Family", ncol=3)

        fig.text(
            0.5, 0.01,
            "filled = p < 0.05,  hollow = p ≥ 0.05",
            ha="center", fontsize=8, color="gray",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        safe_name = target.replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)


def _plot_heatmap(corr_df, out_dir, targets, corr_types):
    """
    One PNG per corr type.
    Rows = 48 law predictors (grouped by param family);
    cols = target metrics.
    Significant cells coloured blue→red; others gray.
    """
    os.makedirs(out_dir, exist_ok=True)
    pooled = corr_df[corr_df["scope"] == "pooled"].copy()

    n_preds = len(LAW_PREDICTORS)
    n_groups = len(LAW_PARAMS)
    n_cols = len(targets)

    row_labels = [
        f"{PARAM_SHORT.get(LAW_PARAMS[i // N_SUFFIX], LAW_PARAMS[i // N_SUFFIX])} "
        f"{SUFFIX_SHORT.get(LAW_SUFFIXES[i % N_SUFFIX], LAW_SUFFIXES[i % N_SUFFIX])}"
        for i in range(n_preds)
    ]

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")

    for r_col, p_col, ct_label in corr_types:
        if r_col not in pooled.columns:
            continue

        fig, ax = plt.subplots(
            1, 1,
            figsize=(max(4, 1.2 * n_cols) + 1.5, max(4, 0.22 * n_preds) + 1.0),
            constrained_layout=True,
        )

        r_mat = np.full((n_preds, n_cols), np.nan)
        sig_mat = np.zeros((n_preds, n_cols), dtype=bool)

        for row_i, pred in enumerate(LAW_PREDICTORS):
            pred_rows = pooled[pooled["predictor"] == pred]
            for col_j, target in enumerate(targets):
                match = pred_rows[pred_rows["normalization"] == target]
                if match.empty:
                    continue
                r_val = match[r_col].values[0]
                p_val = match[p_col].values[0] if p_col in match.columns else np.nan
                if pd.notna(r_val):
                    r_mat[row_i, col_j] = r_val
                if pd.notna(p_val) and p_val < 0.05:
                    sig_mat[row_i, col_j] = True

        display = np.where(sig_mat, r_mat, np.nan)
        im = ax.imshow(display, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        for row_i in range(n_preds):
            for col_j in range(n_cols):
                if sig_mat[row_i, col_j] and not np.isnan(r_mat[row_i, col_j]):
                    ax.text(col_j, row_i, f"{r_mat[row_i, col_j]:.2f}",
                            ha="center", va="center", fontsize=5.5)

        for g in range(1, n_groups):
            ax.axhline(g * N_SUFFIX - 0.5, color="white", linewidth=1.5)

        col_labels = [t.replace("_", "\n") for t in targets]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_preds))
        ax.set_yticklabels(row_labels, fontsize=7)

        for tick_label in ax.get_yticklabels():
            idx = row_labels.index(tick_label.get_text())
            tick_label.set_color(GROUP_COLORS[idx // N_SUFFIX])

        fig.suptitle(f"Law predictor correlations (pooled)  |  {ct_label}", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.5, label="correlation")
        safe_name = r_col.replace("_r", "")
        fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    analysis_cfg = _load_config(args.analysis_config)

    short_model = cfg["model"]["name"].split("/")[-1]
    paths = _resolve_paths(analysis_cfg, short_model)

    corr_df = pd.read_csv(paths["results_csv"])

    active_targets = sorted(corr_df["normalization"].unique().tolist())
    active_corr_types = [(r, p, lbl) for r, p, lbl in CORR_TYPES if r in corr_df.columns]

    lollipop_dir = paths["lollipop_dir"]
    heatmaps_dir = paths["heatmaps_dir"]

    _plot_lollipop(corr_df, lollipop_dir, active_targets, active_corr_types)
    print(f"lollipop → {lollipop_dir}")

    _plot_heatmap(corr_df, heatmaps_dir, active_targets, active_corr_types)
    print(f"heatmaps → {heatmaps_dir}")


if __name__ == "__main__":
    main()
