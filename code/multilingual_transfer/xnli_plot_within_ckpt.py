"""
Plot RankMe → transfer correlation results produced by xnli_correlate_within_ckpt.py.

Reads pairs.csv and correlation_results.csv from the paths in the config and writes:
  scatter/   — predictor vs transfer score, one plot per (predictor, normalization, k)
  timeseries/ — Spearman/Pearson/Kendall r over training, one plot per (normalization, k)

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


def _plot_scatter(pairs_df, scatter_dir, k_values):
    """One scatter plot per (predictor, normalization, k), coloured by checkpoint."""
    os.makedirs(scatter_dir, exist_ok=True)
    checkpoints = sorted(pairs_df["checkpoint"].unique(), key=_checkpoint_sort_key)
    cmap = plt.colormaps["viridis"]
    ckpt_color = {c: cmap(i / max(len(checkpoints) - 1, 1)) for i, c in enumerate(checkpoints)}

    for pred in PREDICTORS:
        for norm in NORMALIZATIONS:
            for k in k_values:
                subset = pairs_df[pairs_df["k"] == k].dropna(subset=[pred, norm])
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(6, 5))
                for ckpt in checkpoints:
                    s = subset[subset["checkpoint"] == ckpt]
                    if not s.empty:
                        ax.scatter(s[pred], s[norm], color=ckpt_color[ckpt], alpha=0.5, s=18)

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(checkpoints) - 1))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_ticks([0, len(checkpoints) - 1])
                cbar.set_ticklabels([checkpoints[0], checkpoints[-1]])
                cbar.set_label("checkpoint", fontsize=9)

                ax.set_xlabel(pred, fontsize=11)
                ax.set_ylabel(norm, fontsize=11)
                ax.set_title(f"{pred} vs {norm}  |  k={k}", fontsize=12)
                fig.tight_layout()
                path = os.path.join(scatter_dir, f"{pred}_{norm}_k{k}.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)


def _plot_timeseries(corr_df, timeseries_dir, k_values):
    """
    One plot per (normalization, k): three subplots (Pearson / Spearman / Kendall),
    each showing all predictors as lines over training tokens.
    Skips checkpoints without a finite token count (e.g. 'main').
    Filled markers = p < 0.05 (permutation), hollow = p >= 0.05.
    """
    os.makedirs(timeseries_dir, exist_ok=True)
    per_ckpt = corr_df[corr_df["scope"] == "per_ckpt"].copy()
    per_ckpt["tokens_B"] = per_ckpt["checkpoint"].map(_token_count)
    per_ckpt = per_ckpt[per_ckpt["tokens_B"].notna()]

    corr_types = [
        ("pearson_r",  "pearson_p",  "Pearson r"),
        ("spearman_r", "spearman_p", "Spearman r"),
        ("kendall_r",  "kendall_p",  "Kendall τ"),
    ]

    for norm in NORMALIZATIONS:
        for k in k_values:
            subset = per_ckpt[(per_ckpt["normalization"] == norm) & (per_ckpt["k"] == k)]
            if subset.empty:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
            fig.suptitle(f"RankMe → transfer correlation over training  |  {norm}  |  k={k}", fontsize=12)

            for ax, (r_col, p_col, label) in zip(axes, corr_types):
                for pred in PREDICTORS:
                    s = subset[subset["predictor"] == pred].sort_values("tokens_B").dropna(subset=[r_col])
                    if s.empty:
                        continue
                    line, = ax.plot(s["tokens_B"], s[r_col], linewidth=1.2, label=pred)
                    color = line.get_color()
                    sig = s[s[p_col] < 0.05]
                    nonsig = s[s[p_col] >= 0.05]
                    ax.scatter(sig["tokens_B"], sig[r_col], color=color, s=30, zorder=3)
                    ax.scatter(nonsig["tokens_B"], nonsig[r_col], color=color, s=30,
                               facecolors="none", zorder=3)
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_ylabel(label, fontsize=10)
                ax.legend(fontsize=8, ncol=2)

            axes[-1].set_xlabel("Tokens (B)", fontsize=11)
            fig.text(0.5, 0.01, "filled = p < 0.05,  hollow = p ≥ 0.05", ha="center", fontsize=8, color="gray")
            fig.tight_layout(rect=[0, 0.03, 1, 1])
            path = os.path.join(timeseries_dir, f"{norm}_k{k}.png")
            fig.savefig(path, dpi=150)
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

    _plot_scatter(pairs_df, paths["scatter_dir"], k_values)
    print(f"Saved scatter plots to {paths['scatter_dir']}")

    _plot_timeseries(corr_df, paths["timeseries_dir"], k_values)
    print(f"Saved timeseries plots to {paths['timeseries_dir']}")


if __name__ == "__main__":
    main()
