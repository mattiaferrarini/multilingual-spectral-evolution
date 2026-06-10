"""
Plot RankMe geometry predictor → ECLeKTic transfer performance correlations
from eclektic_correlate_collapsed.py (layer-collapsed geometry).

Reads correlation_results.csv (with `layer_collapse` and `ckpt_collapse` columns)
and writes two plot types per ckpt_collapse method (subdirectory per method):

  {ckpt_method}/bar_summary/  — one PNG per target metric.
                  Grouped bar chart: groups = layer_collapse method, bars = predictor.
                  Solid bar = p < 0.05, hatched = p ≥ 0.05.
  {ckpt_method}/timeseries/   — one PNG per (layer_collapse_method, target metric).
                  x = training tokens. Color = predictor.

Usage:
    python eclektic_plot_collapsed.py \\
        --config configs/eclektic_apertus.yaml \\
        --analysis-config configs/eclektic_correlation_collapsed_analysis.yaml
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from checkpoints import _checkpoint_sort_key
from eclektic_targets import ALL_TARGETS as TARGETS
from geometry_predictors import PREDICTORS

CORR_TYPES = [
    ("pearson_r",  "pearson_p",  "Pearson r"),
    ("spearman_r", "spearman_p", "Spearman r"),
    ("kendall_r",  "kendall_p",  "Kendall τ"),
]

BAR_HATCH = "//"


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


def _plot_bar_summary(corr_df, out_dir, collapse_methods, predictors, targets, corr_types):
    """
    One PNG per target metric.
    Grouped bar chart: groups = layer_collapse methods, bars = predictors.
    Solid = p < 0.05, hatched = p ≥ 0.05. Rows = corr types.
    """
    pred_colors = plt.cm.tab10.colors[:len(predictors)]
    n_preds = len(predictors)
    n_rows = len(corr_types)

    os.makedirs(out_dir, exist_ok=True)
    pooled = corr_df[corr_df["scope"] == "pooled"].copy()

    bar_width = 0.8 / n_preds
    positions = {m: float(i) for i, m in enumerate(collapse_methods)}

    for target in targets:
        target_data = pooled[pooled["normalization"] == target]

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(max(5, 1.5 * len(collapse_methods) * n_preds), 4 * n_rows),
            squeeze=False,
        )
        fig.suptitle(f"Pooled correlations  |  {target}", fontsize=12)

        for row_idx, (r_col, p_col, label) in enumerate(corr_types):
            ax = axes[row_idx, 0]

            for pred_idx, pred in enumerate(predictors):
                color = pred_colors[pred_idx]
                offset = (pred_idx - n_preds / 2 + 0.5) * bar_width

                for method in collapse_methods:
                    row = target_data[
                        (target_data["predictor"] == pred) &
                        (target_data["layer_collapse"] == method)
                    ]
                    if row.empty or pd.isna(row[r_col].values[0]):
                        continue
                    r_val = row[r_col].values[0]
                    p_val = row[p_col].values[0] if p_col in row.columns else np.nan
                    sig = pd.notna(p_val) and p_val < 0.05
                    center = positions[method]
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
            ax.set_xticks(list(positions.values()))
            ax.set_xticklabels(list(positions.keys()), fontsize=9)

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
        path = os.path.join(out_dir, f"{target}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _plot_timeseries(corr_df, out_dir, collapse_methods, predictors, targets, corr_types):
    """
    One PNG per (layer_collapse_method, target metric).
    x = training tokens (B). Rows = corr types. Lines = predictor.
    """
    pred_colors = plt.cm.tab10.colors[:len(predictors)]
    n_rows = len(corr_types)

    os.makedirs(out_dir, exist_ok=True)

    per_ckpt = corr_df[corr_df["scope"] == "per_ckpt"].copy()
    per_ckpt["tokens_B"] = per_ckpt["checkpoint"].map(_token_count)
    per_ckpt = per_ckpt[per_ckpt["tokens_B"].notna()]

    if per_ckpt.empty:
        return

    for method in collapse_methods:
        method_data = per_ckpt[per_ckpt["layer_collapse"] == method]

        for target in targets:
            subset = method_data[method_data["normalization"] == target]

            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(10, 4 * n_rows),
                sharex=True,
                squeeze=False,
            )
            fig.suptitle(
                f"RankMe → transfer correlation  |  {target}  |  {method}",
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
            path = os.path.join(out_dir, f"{method}_{target}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    analysis_cfg = _load_config(args.analysis_config)

    short_model = cfg["model"]["name"].split("/")[-1]
    paths = _resolve_paths(analysis_cfg, short_model)

    corr_df = pd.read_csv(paths["results_csv"])

    ckpt_collapse_values = (
        sorted(corr_df["ckpt_collapse"].unique().tolist())
        if "ckpt_collapse" in corr_df.columns
        else ["none"]
    )

    for ckpt_method in ckpt_collapse_values:
        if "ckpt_collapse" in corr_df.columns:
            method_df = corr_df[corr_df["ckpt_collapse"] == ckpt_method].copy()
        else:
            method_df = corr_df

        collapse_methods = method_df["layer_collapse"].unique().tolist()
        active_predictors = [p for p in PREDICTORS if p in method_df["predictor"].unique()]
        active_targets = [t for t in TARGETS if t in method_df["normalization"].unique()]
        active_corr_types = [(r, p, lbl) for r, p, lbl in CORR_TYPES if r in method_df.columns]

        bar_dir        = os.path.join(paths["bar_summary_dir"], ckpt_method)
        timeseries_dir = os.path.join(paths["timeseries_dir"],  ckpt_method)

        _plot_bar_summary(method_df, bar_dir, collapse_methods,
                          active_predictors, active_targets, active_corr_types)
        print(f"[{ckpt_method}] bar_summary → {bar_dir}")

        _plot_timeseries(method_df, timeseries_dir, collapse_methods,
                         active_predictors, active_targets, active_corr_types)
        print(f"[{ckpt_method}] timeseries  → {timeseries_dir}")


if __name__ == "__main__":
    main()
