"""
Plot RankMe → future transfer correlation results from xnli_correlate_ahead_collapsed.py.

Reads correlation_results.csv (with `layer_collapse` and `t` columns, no `layer` column)
and writes three plot types:

  lag_summary/  — PRIMARY: one PNG per (normalization, k).
                  x = T (lag), y = pooled correlation.
                  Color = predictor, line style = collapse method.
  timeseries/   — one PNG per (collapse_method, normalization). x = training tokens.
                  Color = predictor, line style = T value.
  bar_summary/  — one PNG per (normalization, k).
                  Grouped bar chart: x = T, bars = predictor, groups = collapse method.
                  Solid bar = p < 0.05, hatched = p ≥ 0.05.

Usage:
    python xnli_plot_ahead_collapsed.py \\
        --config configs/xnli_apertus.yaml \\
        --analysis-config configs/xnli_correlation_ahead_collapsed_analysis.yaml
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

METHOD_LINESTYLES = ["-", "--", ":", "-."]
T_LINESTYLES = ["-", "--", ":", "-."]
PRED_COLORS = plt.cm.tab10.colors[:len(PREDICTORS)]
BAR_HATCH = "//"


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


def _plot_lag_summary(corr_df, lag_dir, k_values, t_values, collapse_methods,
                      predictors=None, normalizations=None, corr_types=None):
    """
    One PNG per (normalization, k).
    x = T (lag), y = pooled correlation.
    Color = predictor, line style = collapse method. Filled marker = p < 0.05.
    """
    if predictors is None:
        predictors = PREDICTORS
    if normalizations is None:
        normalizations = NORMALIZATIONS
    if corr_types is None:
        corr_types = CORR_TYPES

    pred_colors = plt.cm.tab10.colors[:len(predictors)]
    method_ls = {m: METHOD_LINESTYLES[i % len(METHOD_LINESTYLES)] for i, m in enumerate(collapse_methods)}

    os.makedirs(lag_dir, exist_ok=True)
    pooled = corr_df[corr_df["scope"] == "pooled"].copy()
    t_sorted = sorted(t_values)
    k_sorted = sorted(k_values)
    n_rows = len(corr_types)

    for norm in normalizations:
        norm_data = pooled[pooled["normalization"] == norm]

        for k in k_sorted:
            k_data = norm_data[norm_data["k"] == k]

            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(max(6, 1.5 * len(t_sorted)), 4 * n_rows),
                sharex=True,
                squeeze=False,
            )
            fig.suptitle(
                f"RankMe predictability vs lag  |  {norm}  |  k={k}",
                fontsize=12,
            )

            for row_idx, (r_col, p_col, label) in enumerate(corr_types):
                ax = axes[row_idx, 0]

                for pred_idx, pred in enumerate(predictors):
                    color = pred_colors[pred_idx]
                    for method in collapse_methods:
                        s = (
                            k_data[(k_data["predictor"] == pred) & (k_data["layer_collapse"] == method)]
                            .copy()
                            .dropna(subset=[r_col])
                            .sort_values("t")
                        )
                        if s.empty:
                            continue
                        ls = method_ls[method]
                        ax.plot(s["t"], s[r_col], color=color, linestyle=ls, linewidth=1.4)
                        sig = s[s[p_col] < 0.05]
                        nonsig = s[s[p_col] >= 0.05]
                        ax.scatter(sig["t"], sig[r_col], color=color, s=30, zorder=3)
                        ax.scatter(
                            nonsig["t"], nonsig[r_col],
                            color=color, s=30, facecolors="none", zorder=3,
                        )

                ymin, ymax = ax.get_ylim()
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_ylim(ymin, ymax)
                ax.set_ylabel(label, fontsize=10)
                ax.set_xticks(t_sorted)

            axes[-1, 0].set_xlabel("T (lag)", fontsize=10)

            pred_handles = [
                mpatches.Patch(color=pred_colors[i], label=p)
                for i, p in enumerate(predictors)
            ]
            method_handles = [
                mlines.Line2D([], [], color="black", linestyle=method_ls[m],
                              linewidth=1.2, label=m)
                for m in collapse_methods
            ]
            axes[0, 0].legend(
                handles=pred_handles + method_handles, fontsize=8, loc="best",
            )

            fig.text(
                0.5, 0.01,
                "filled = p < 0.05,  hollow = p ≥ 0.05",
                ha="center", fontsize=8, color="gray",
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            path = os.path.join(lag_dir, f"{norm}_k{k}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)


def _plot_timeseries(corr_df, timeseries_dir, k_values, t_values, collapse_methods,
                     predictors=None, normalizations=None, corr_types=None):
    """
    One PNG per (collapse_method, normalization).
    x = training tokens. Rows = corr types, cols = k values.
    Color = predictor, line style = T value. Filled/hollow markers for significance.
    """
    if predictors is None:
        predictors = PREDICTORS
    if normalizations is None:
        normalizations = NORMALIZATIONS
    if corr_types is None:
        corr_types = CORR_TYPES

    pred_colors = plt.cm.tab10.colors[:len(predictors)]
    t_sorted = sorted(t_values)
    t_ls = {t: T_LINESTYLES[i % len(T_LINESTYLES)] for i, t in enumerate(t_sorted)}
    k_sorted = sorted(k_values)
    n_rows = len(corr_types)

    os.makedirs(timeseries_dir, exist_ok=True)

    per_ckpt = corr_df[corr_df["scope"] == "per_ckpt"].copy()
    per_ckpt["tokens_B"] = per_ckpt["checkpoint"].map(_token_count)
    per_ckpt = per_ckpt[per_ckpt["tokens_B"].notna()]

    for method in collapse_methods:
        method_data = per_ckpt[per_ckpt["layer_collapse"] == method]

        for norm in normalizations:
            fig, axes = plt.subplots(
                n_rows, len(k_sorted),
                figsize=(max(8, 5 * len(k_sorted)), 4 * n_rows),
                sharex=True,
                squeeze=False,
            )
            fig.suptitle(
                f"RankMe → transfer correlation  |  {norm}  |  {method}",
                fontsize=12,
            )

            for col_idx, k in enumerate(k_sorted):
                subset = method_data[
                    (method_data["normalization"] == norm) & (method_data["k"] == k)
                ]

                for row_idx, (r_col, p_col, label) in enumerate(corr_types):
                    ax = axes[row_idx, col_idx]

                    for pred_idx, pred in enumerate(predictors):
                        color = pred_colors[pred_idx]
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

                    ymin, ymax = ax.get_ylim()
                    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                    ax.set_ylim(ymin, ymax)
                    if row_idx == 0:
                        ax.set_title(f"k={k}", fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(label, fontsize=10)

            legend_ax = axes[0, -1]
            pred_handles = [
                mpatches.Patch(color=pred_colors[i], label=pred)
                for i, pred in enumerate(predictors)
            ]
            t_handles = [
                mlines.Line2D([], [], color="black", linestyle=t_ls[t], linewidth=1.2, label=f"T={t}")
                for t in t_sorted
            ]
            legend_ax.legend(handles=pred_handles + t_handles, fontsize=7, ncol=1, loc="upper right")

            for col_idx in range(len(k_sorted)):
                axes[-1, col_idx].set_xlabel("Tokens (B)", fontsize=10)

            fig.text(
                0.5, 0.01,
                "filled = p < 0.05,  hollow = p ≥ 0.05",
                ha="center", fontsize=8, color="gray",
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            path = os.path.join(timeseries_dir, f"{method}_{norm}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)


def _plot_bar_summary(corr_df, bar_dir, k_values, t_values, collapse_methods,
                      predictors=None, normalizations=None, corr_types=None):
    """
    One PNG per (normalization, k).
    Grouped bar chart: x positions = collapse_method × T (with a gap between methods).
    Bars = predictors (color-coded). Solid = p < 0.05, hatched = p ≥ 0.05.
    Rows = corr types.
    """
    if predictors is None:
        predictors = PREDICTORS
    if normalizations is None:
        normalizations = NORMALIZATIONS
    if corr_types is None:
        corr_types = CORR_TYPES

    pred_colors = plt.cm.tab10.colors[:len(predictors)]
    t_sorted = sorted(t_values)
    k_sorted = sorted(k_values)
    n_preds = len(predictors)
    n_rows = len(corr_types)

    os.makedirs(bar_dir, exist_ok=True)
    pooled = corr_df[corr_df["scope"] == "pooled"].copy()

    # Build x-positions: for each method, n_t groups; groups separated by a gap between methods
    bar_width = 0.8 / n_preds
    group_gap = 0.5          # extra space between collapse methods
    positions = {}           # (method, t) -> group center x
    x = 0.0
    for method in collapse_methods:
        for t in t_sorted:
            positions[(method, t)] = x
            x += 1.0
        x += group_gap
    total_width = x - group_gap

    # x-tick positions and labels
    tick_xs, tick_labels = [], []
    for method in collapse_methods:
        method_center = np.mean([positions[(method, t)] for t in t_sorted])
        tick_xs.append(method_center)
        tick_labels.append(method)

    for norm in normalizations:
        norm_data = pooled[pooled["normalization"] == norm]

        for k in k_sorted:
            k_data = norm_data[norm_data["k"] == k]

            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(max(8, 0.8 * total_width * n_preds), 4 * n_rows),
                squeeze=False,
            )
            fig.suptitle(
                f"Pooled correlations  |  {norm}  |  k={k}",
                fontsize=12,
            )

            for row_idx, (r_col, p_col, label) in enumerate(corr_types):
                ax = axes[row_idx, 0]

                for pred_idx, pred in enumerate(predictors):
                    color = pred_colors[pred_idx]
                    offset = (pred_idx - n_preds / 2 + 0.5) * bar_width

                    for method in collapse_methods:
                        for t in t_sorted:
                            row = k_data[
                                (k_data["predictor"] == pred) &
                                (k_data["layer_collapse"] == method) &
                                (k_data["t"] == t)
                            ]
                            if row.empty or pd.isna(row[r_col].values[0]):
                                continue
                            r_val = row[r_col].values[0]
                            p_val = row[p_col].values[0] if p_col in row.columns else np.nan
                            sig = pd.notna(p_val) and p_val < 0.05
                            center = positions[(method, t)]
                            ax.bar(
                                center + offset, r_val,
                                width=bar_width * 0.9,
                                color=color,
                                hatch="" if sig else BAR_HATCH,
                                edgecolor="white" if sig else color,
                                linewidth=0.5,
                            )

                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel(label, fontsize=10)
                ax.set_xticks(tick_xs)
                ax.set_xticklabels(tick_labels, fontsize=9)

                # T value annotations below each group
                for method in collapse_methods:
                    for t in t_sorted:
                        ax.text(
                            positions[(method, t)], ax.get_ylim()[0],
                            f"T={t}", ha="center", va="top", fontsize=6, color="gray",
                        )

            pred_handles = [
                mpatches.Patch(color=pred_colors[i], label=pred)
                for i, pred in enumerate(predictors)
            ]
            sig_handles = [
                mpatches.Patch(facecolor="lightgray", edgecolor="gray", label="p < 0.05 (solid)"),
                mpatches.Patch(facecolor="lightgray", edgecolor="gray",
                               hatch=BAR_HATCH, label="p ≥ 0.05 (hatched)"),
            ]
            axes[0, 0].legend(handles=pred_handles + sig_handles, fontsize=8, loc="upper right")

            fig.tight_layout()
            path = os.path.join(bar_dir, f"{norm}_k{k}.png")
            fig.savefig(path, dpi=150)
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
    collapse_methods = corr_df["layer_collapse"].unique().tolist()

    active_predictors = [p for p in PREDICTORS if p in corr_df["predictor"].unique()]
    active_normalizations = [n for n in NORMALIZATIONS if n in corr_df["normalization"].unique()]
    active_corr_types = [(r, p, lbl) for r, p, lbl in CORR_TYPES if r in corr_df.columns]

    _plot_lag_summary(corr_df, paths["lag_dir"], k_values, t_values, collapse_methods,
                      predictors=active_predictors, normalizations=active_normalizations,
                      corr_types=active_corr_types)
    print(f"Saved lag summary plots to {paths['lag_dir']}")

    _plot_timeseries(corr_df, paths["timeseries_dir"], k_values, t_values, collapse_methods,
                     predictors=active_predictors, normalizations=active_normalizations,
                     corr_types=active_corr_types)
    print(f"Saved timeseries plots to {paths['timeseries_dir']}")

    _plot_bar_summary(corr_df, paths["bar_dir"], k_values, t_values, collapse_methods,
                      predictors=active_predictors, normalizations=active_normalizations,
                      corr_types=active_corr_types)
    print(f"Saved bar summary plots to {paths['bar_dir']}")


if __name__ == "__main__":
    main()
