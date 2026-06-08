"""
Plot RankMe → future transfer correlation results from xnli_correlate_ahead_collapsed.py.

Reads correlation_results.csv (with `layer_collapse`, `ckpt_collapse`, and `t` columns)
and writes three plot types per ckpt_collapse method (subdirectory per method):

  {ckpt_method}/lag_summary/  — PRIMARY: one PNG per (normalization, k).
                  x = T (lag), y = pooled correlation.
                  Color = predictor, line style = layer_collapse method.
  {ckpt_method}/timeseries/   — one PNG per (layer_collapse_method, normalization). x = training tokens.
                  Color = predictor, line style = T value.
  {ckpt_method}/bar_summary/  — one PNG per (normalization, k).
                  Grouped bar chart: x = T, bars = predictor, groups = layer_collapse method.
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
import seaborn as sns
import yaml

from checkpoints import _checkpoint_sort_key
from geometry_predictors import PREDICTORS
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
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]



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

            axes[-1, 0].set_xlabel("T (checkpoint lag)", fontsize=10)

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


_CORR_TYPE_MAP = {
    "pearson":  ("pearson_r",  "pearson_p",  "Pearson r"),
    "spearman": ("spearman_r", "spearman_p", "Spearman r"),
    "kendall":  ("kendall_r",  "kendall_p",  "Kendall τ"),
}


def plot_timeseries_comparison(
    config1, analysis_config1,
    config2, analysis_config2,
    corr_type="spearman",
    predictor=None,
    normalization="row_norm",
    k=8,
    collapse_method="average_rankme",
    save_path=None,
):
    """
    Plot per-checkpoint correlation timeseries for two models side by side.

    Each subplot shows the specified correlation coefficient on the y-axis vs.
    training tokens on the x-axis.  One line per lag T value; filled markers
    indicate p < 0.05, hollow markers p ≥ 0.05.

    Parameters
    ----------
    config1, config2 : str | dict
        Paths to per-model XNLI config YAMLs (or already-loaded dicts).
    analysis_config1, analysis_config2 : str | dict
        Paths to correlation-analysis config YAMLs (or already-loaded dicts).
    corr_type : {"pearson", "spearman", "kendall"}
        Which correlation coefficient to plot.
    predictor : str | None
        Geometry predictor name.  Defaults to the first predictor in PREDICTORS.
    normalization : {"row_norm", "col_norm"}
    k : int
        Number of in-context examples.
    collapse_method : str
        layer_collapse method (e.g. "average_rankme", "average_predictors").
    save_path : str | None
        If given, save the figure to this path (.svg recommended).

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_theme(style="white", font_scale=1.0)

    if corr_type not in _CORR_TYPE_MAP:
        raise ValueError(f"corr_type must be one of {list(_CORR_TYPE_MAP)}, got {corr_type!r}")
    r_col, p_col, corr_label = _CORR_TYPE_MAP[corr_type]

    if predictor is None:
        predictor = PREDICTORS[0]

    def _load(cfg):
        return cfg if isinstance(cfg, dict) else _load_config(cfg)

    def _base_dir(cfg_path):
        """Walk up from config file to the first ancestor that contains a 'code/' dir."""
        if isinstance(cfg_path, dict):
            return None
        d = os.path.dirname(os.path.abspath(cfg_path))
        while d and d != os.path.dirname(d):
            if os.path.isdir(os.path.join(d, "code")):
                return d
            d = os.path.dirname(d)
        return None

    def _abs(path, base):
        if base is None or os.path.isabs(path):
            return path
        return os.path.join(base, path)

    cfg1 = _load(config1);  acfg1 = _load(analysis_config1)
    cfg2 = _load(config2);  acfg2 = _load(analysis_config2)

    base1 = _base_dir(analysis_config1)
    base2 = _base_dir(analysis_config2)

    model1 = cfg1["model"]["name"].split("/")[-1]
    model2 = cfg2["model"]["name"].split("/")[-1]

    step1 = cfg1["model"].get("checkpoint_step") or 1
    step2 = cfg2["model"].get("checkpoint_step") or 1

    paths1 = _resolve_paths(acfg1, model1)
    paths2 = _resolve_paths(acfg2, model2)

    def _prepare(paths, base):
        df = pd.read_csv(_abs(paths["results_csv"], base))
        df = df[df["scope"] == "per_ckpt"].copy()
        if "ckpt_collapse" in df.columns:
            df = df[df["ckpt_collapse"] == "none"]
        df = df[
            (df["layer_collapse"] == collapse_method) &
            (df["normalization"] == normalization) &
            (df["k"] == k) &
            (df["predictor"] == predictor)
        ].copy()
        df["tokens_B"] = df["checkpoint"].map(_token_count)
        return df[df["tokens_B"].notna()]

    df1 = _prepare(paths1, base1)
    df2 = _prepare(paths2, base2)

    t_values = sorted(set(df1["t"].unique()) | set(df2["t"].unique()))
    t_marker = {t: _MARKERS[i % len(_MARKERS)] for i, t in enumerate(t_values)}
    t_color  = {}
    t_display_label = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(
        f"{corr_label}  |  {predictor}  |  {normalization}  |  k={k}  |  {collapse_method}",
        fontsize=11,
    )

    for ax, df, model_name, step in zip(axes, [df1, df2], [model1, model2], [step1, step2]):
        for t in t_values:
            s = df[df["t"] == t].sort_values("tokens_B").dropna(subset=[r_col])
            if s.empty:
                continue
            t_scaled = t * step
            (line,) = ax.plot(s["tokens_B"], s[r_col], linewidth=1.4, label=f"T={t_scaled}")
            color = line.get_color()
            t_color.setdefault(t, color)
            t_display_label.setdefault(t, f"T={t_scaled}")
            marker = t_marker[t]
            sig = s[s[p_col] < 0.05]
            nonsig = s[s[p_col] >= 0.05]
            ax.scatter(sig["tokens_B"], sig[r_col], color=color, marker=marker, s=30, zorder=3)
            ax.scatter(nonsig["tokens_B"], nonsig[r_col],
                       color=color, marker=marker, s=30, facecolors="none", zorder=3)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Tokens (B)", fontsize=9)

    axes[0].set_ylabel(corr_label, fontsize=9)

    t_handles = [
        mlines.Line2D([], [], color=t_color[t], marker=t_marker[t],
                      linewidth=1.4, markersize=6, label=t_display_label[t])
        for t in t_values if t in t_color
    ]
    if t_handles:
        fig.legend(handles=t_handles, loc="lower center", ncol=len(t_handles),
                   fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.text(0.5, -0.18, "filled = p < 0.05,  hollow = p ≥ 0.05",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_lag_summary_comparison(
    config1, analysis_config1,
    config2, analysis_config2,
    corr_type="spearman",
    predictors=None,
    normalization="row_norm",
    k=8,
    collapse_method="average_rankme",
    save_path=None,
):
    """
    Plot pooled correlation vs lag T for two models side by side.

    Each subplot shows the specified correlation coefficient on the y-axis vs.
    T (lag) on the x-axis.  One line per predictor, color-coded; filled markers
    indicate p < 0.05, hollow markers p ≥ 0.05.

    Parameters
    ----------
    config1, config2 : str | dict
        Paths to per-model XNLI config YAMLs (or already-loaded dicts).
    analysis_config1, analysis_config2 : str | dict
        Paths to correlation-analysis config YAMLs (or already-loaded dicts).
    corr_type : {"pearson", "spearman", "kendall"}
        Which correlation coefficient to plot.
    predictors : list[str] | None
        Geometry predictor names to include.  Defaults to all PREDICTORS.
    normalization : {"row_norm", "col_norm"}
    k : int | list[int]
        Number of in-context examples.  Pass a list to show one line per
        (predictor, k) combination; k values are distinguished by line style.
    collapse_method : str
        layer_collapse method (e.g. "average_rankme", "average_predictors").
    save_path : str | None
        If given, save the figure to this path (.svg recommended).

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_theme(style="white", font_scale=1.0)

    if corr_type not in _CORR_TYPE_MAP:
        raise ValueError(f"corr_type must be one of {list(_CORR_TYPE_MAP)}, got {corr_type!r}")
    r_col, p_col, corr_label = _CORR_TYPE_MAP[corr_type]

    if predictors is None:
        predictors = PREDICTORS
    k_values = sorted(k) if isinstance(k, (list, tuple)) else [k]

    def _load(cfg):
        return cfg if isinstance(cfg, dict) else _load_config(cfg)

    def _base_dir(cfg_path):
        if isinstance(cfg_path, dict):
            return None
        d = os.path.dirname(os.path.abspath(cfg_path))
        while d and d != os.path.dirname(d):
            if os.path.isdir(os.path.join(d, "code")):
                return d
            d = os.path.dirname(d)
        return None

    def _abs(path, base):
        if base is None or os.path.isabs(path):
            return path
        return os.path.join(base, path)

    cfg1 = _load(config1);  acfg1 = _load(analysis_config1)
    cfg2 = _load(config2);  acfg2 = _load(analysis_config2)

    base1 = _base_dir(analysis_config1)
    base2 = _base_dir(analysis_config2)

    model1 = cfg1["model"]["name"].split("/")[-1]
    model2 = cfg2["model"]["name"].split("/")[-1]

    step1 = cfg1["model"].get("checkpoint_step") or 1
    step2 = cfg2["model"].get("checkpoint_step") or 1

    paths1 = _resolve_paths(acfg1, model1)
    paths2 = _resolve_paths(acfg2, model2)

    def _prepare(paths, base):
        df = pd.read_csv(_abs(paths["results_csv"], base))
        df = df[df["scope"] == "pooled"].copy()
        if "ckpt_collapse" in df.columns:
            df = df[df["ckpt_collapse"] == "none"]
        return df[
            (df["layer_collapse"] == collapse_method) &
            (df["normalization"] == normalization) &
            (df["k"].isin(k_values)) &
            (df["predictor"].isin(predictors))
        ].copy()

    df1 = _prepare(paths1, base1)
    df2 = _prepare(paths2, base2)

    k_linestyle  = {kv: METHOD_LINESTYLES[i % len(METHOD_LINESTYLES)] for i, kv in enumerate(k_values)}
    pred_marker  = {p: _MARKERS[i % len(_MARKERS)] for i, p in enumerate(predictors)}
    k_label = {kv: f"k={kv}" for kv in k_values}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    k_str = str(k_values[0]) if len(k_values) == 1 else ", ".join(str(kv) for kv in k_values)
    fig.suptitle(
        f"{corr_label}  |  {normalization}  |  k={k_str}  |  {collapse_method}",
        fontsize=11,
    )

    pred_color = {}
    for ax, df, model_name, step in zip(axes, [df1, df2], [model1, model2], [step1, step2]):
        t_values_scaled = sorted(t * step for t in df["t"].unique())
        for pred in predictors:
            for kv in k_values:
                s = (
                    df[(df["predictor"] == pred) & (df["k"] == kv)]
                    .sort_values("t")
                    .dropna(subset=[r_col])
                )
                if s.empty:
                    continue
                t_scaled = s["t"] * step
                ls = k_linestyle[kv]
                label = pred if len(k_values) == 1 else f"{pred}  {k_label[kv]}"
                (line,) = ax.plot(t_scaled, s[r_col], linestyle=ls, linewidth=1.4, label=label)
                color = line.get_color()
                marker = pred_marker[pred]
                pred_color.setdefault(pred, color)
                sig = s[s[p_col] < 0.05]
                nonsig = s[s[p_col] >= 0.05]
                ax.scatter(sig["t"] * step, sig[r_col], color=color, marker=marker, s=30, zorder=3)
                ax.scatter(nonsig["t"] * step, nonsig[r_col],
                           color=color, marker=marker, s=30, facecolors="none", zorder=3)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(t_values_scaled)
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("T (checkpoint lag)", fontsize=9)

    axes[0].set_ylabel(corr_label, fontsize=9)


    pred_handles = [
        mlines.Line2D([], [], color=pred_color[p], marker=pred_marker[p],
                      linewidth=1.4, markersize=6, label=p)
        for p in predictors if p in pred_color
    ]
    k_handles = [
        mlines.Line2D([], [], color="black", linestyle=k_linestyle[kv],
                      linewidth=1.2, label=k_label[kv])
        for kv in k_values
    ] if len(k_values) > 1 else []
    n_legend_cols = len(pred_handles) + len(k_handles)
    fig.legend(handles=pred_handles + k_handles, loc="lower center", ncol=n_legend_cols,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.08))

    fig.text(0.5, -0.18, "filled = p < 0.05,  hollow = p ≥ 0.05",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


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

        t_values = sorted(method_df["t"].unique().tolist())
        k_values = sorted(method_df["k"].unique().tolist())
        collapse_methods = method_df["layer_collapse"].unique().tolist()

        active_predictors = [p for p in PREDICTORS if p in method_df["predictor"].unique()]
        active_normalizations = [n for n in NORMALIZATIONS if n in method_df["normalization"].unique()]
        active_corr_types = [(r, p, lbl) for r, p, lbl in CORR_TYPES if r in method_df.columns]

        lag_dir        = os.path.join(paths["lag_dir"],        ckpt_method)
        timeseries_dir = os.path.join(paths["timeseries_dir"], ckpt_method)
        bar_dir        = os.path.join(paths["bar_dir"],        ckpt_method)

        _plot_lag_summary(method_df, lag_dir, k_values, t_values, collapse_methods,
                          predictors=active_predictors, normalizations=active_normalizations,
                          corr_types=active_corr_types)
        print(f"[{ckpt_method}] lag_summary → {lag_dir}")

        _plot_timeseries(method_df, timeseries_dir, k_values, t_values, collapse_methods,
                         predictors=active_predictors, normalizations=active_normalizations,
                         corr_types=active_corr_types)
        print(f"[{ckpt_method}] timeseries  → {timeseries_dir}")

        _plot_bar_summary(method_df, bar_dir, k_values, t_values, collapse_methods,
                          predictors=active_predictors, normalizations=active_normalizations,
                          corr_types=active_corr_types)
        print(f"[{ckpt_method}] bar_summary → {bar_dir}")


if __name__ == "__main__":
    main()
