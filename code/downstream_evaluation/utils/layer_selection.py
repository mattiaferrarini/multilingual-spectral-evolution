"""
Layer selection utilities for downstream correlation analysis.

Computes cross-language spectral metric stratification (std across languages
at each layer, averaged over checkpoints) and plots the result for both models.
Supports any column in the geometry CSV (e.g. rankme, alpha_req).
"""

import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .checkpoints import sort_checkpoints


def layer_num(name: str) -> int:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0


def compute_stratification(csv, agg: str = "last", metric: str = "rankme",
                           n_first_checkpoints: int | None = None) -> tuple:
    """
    For every layer compute std and mean of `metric` across languages
    (each language value is its mean over checkpoints).

    If n_first_checkpoints is given, only the first N checkpoints (by token
    count) are used — simulating early-training layer selection.

    Returns (layer_numbers, stds, means).
    """
    df = pd.read_csv(csv)
    df = df[df["aggregation"] == agg].copy()

    if n_first_checkpoints is not None:
        sorted_ckpts = sort_checkpoints(df["checkpoint"].unique())
        keep = set(sorted_ckpts[:n_first_checkpoints])
        df = df[df["checkpoint"].isin(keep)]

    layers = sorted(df["layer"].unique(), key=layer_num)
    lnums  = [layer_num(l) for l in layers]

    mean_df = df.groupby(["layer", "dataset"])[metric].mean().reset_index()

    stds, means = [], []
    for layer in layers:
        vals = mean_df[mean_df["layer"] == layer][metric].values
        stds.append(float(vals.std())  if len(vals) > 1 else 0.0)
        means.append(float(vals.mean()) if len(vals) > 0 else 0.0)

    return lnums, np.array(stds), np.array(means)


def layer_selection_sweep(csv, agg: str = "last", metric: str = "rankme") -> pd.DataFrame:
    """
    For each N from 1 to the total checkpoint count, compute cross-language
    stratification using only the first N checkpoints and record which layer
    is selected (highest std).

    Returns a DataFrame with columns:
        n_checkpoints, selected_layer, peak_std, stable
    where `stable` is True when the selected layer is the same as for N-1.
    """
    df_full = pd.read_csv(csv)
    df_full = df_full[df_full["aggregation"] == agg]
    all_ckpts = sort_checkpoints(df_full["checkpoint"].unique())
    total = len(all_ckpts)

    records = []
    prev_layer = None
    for n in range(1, total + 1):
        lnums, stds, _ = compute_stratification(csv, agg=agg, metric=metric,
                                                 n_first_checkpoints=n)
        peak_idx = int(np.argmax(stds))
        layer    = lnums[peak_idx]
        records.append({
            "n_checkpoints":  n,
            "selected_layer": layer,
            "peak_std":       round(float(stds[peak_idx]), 4),
            "stable":         layer == prev_layer,
        })
        prev_layer = layer
    return pd.DataFrame(records)


_METRIC_LABELS = {
    "rankme":    "RankMe",
    "alpha_req": "AlphaReQ",
}


def plot_stratification(models: list[dict], metric: str = "rankme",
                        n_first_checkpoints: int | None = None) -> plt.Figure:
    """
    Two-panel figure: cross-language metric std vs. layer for each model.

    Required keys per model dict: label, csv, color.
    `metric` must be a column in the geometry CSV (e.g. "rankme", "alpha_req").
    `n_first_checkpoints` limits to the first N checkpoints for stratification.
    """
    metric_label = _METRIC_LABELS.get(metric, metric)
    n_label = f"first {n_first_checkpoints} checkpoints" if n_first_checkpoints else "all checkpoints"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Cross-language {metric_label} spread by layer  ({n_label})\n"
        "Higher std = languages more discriminable = stronger signal for correlation analysis",
        fontsize=13, y=1.03,
    )

    for ax, cfg in zip(axes, models):
        lnums, stds, _ = compute_stratification(cfg["csv"], metric=metric,
                                                 n_first_checkpoints=n_first_checkpoints)
        peak_idx   = int(np.argmax(stds))
        peak_layer = lnums[peak_idx]
        peak_std   = stds[peak_idx]

        ax.plot(lnums, stds,
                color=cfg["color"], lw=2.2, marker="o", ms=4, zorder=3)
        ax.fill_between(lnums, stds, alpha=0.12, color=cfg["color"])

        ax.axvline(peak_layer, color=cfg["color"], lw=1.8, ls="--", zorder=2)
        on_left      = peak_layer < max(lnums) * 0.7
        x_pts        = 40 if on_left else -40
        ha           = "left" if on_left else "right"
        ax.annotate(
            f"layer {peak_layer}\nstd = {peak_std:.2f}",
            xy=(peak_layer, peak_std),
            xytext=(x_pts, -30),
            textcoords="offset points",
            fontsize=9, color=cfg["color"], fontweight="bold",
            ha=ha, va="top",
            arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.2),
        )

        ax.set_title(cfg["label"], fontsize=12, pad=8)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(f"Std({metric_label}) across languages", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


def show_stratification(model_keys: list, data: dict,
                        metric: str = "rankme",
                        n_first_checkpoints: int | None = None) -> None:
    """
    Build, display, and save the cross-language stratification figure.

    Wraps plot_stratification() with the data-dict convention used by the
    notebook: data[m] must contain cfg (with model_label, rankme_csv,
    plots_dir) and color.

    `metric` is passed to plot_stratification (e.g. "rankme", "alpha_req").
    `n_first_checkpoints` limits stratification to the first N checkpoints.
    The output filename reflects the metric used.
    """
    models = [
        dict(
            label     = data[m]["cfg"]["model_label"],
            csv       = data[m]["cfg"]["rankme_csv"],
            color     = data[m]["color"],
            model_key = m,
        )
        for m in model_keys
    ]
    fig = plot_stratification(models, metric=metric,
                              n_first_checkpoints=n_first_checkpoints)
    fname = f"{metric}_layer_stratification.png"
    save_path = data[model_keys[0]]["cfg"]["plots_dir"] / fname
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def stratification_summary(models: list[dict]) -> pd.DataFrame:
    """
    Return a tidy DataFrame with peak-layer stats for each model.
    """
    rows = []
    for cfg in models:
        lnums, stds, _ = compute_stratification(cfg["csv"])
        peak_idx   = int(np.argmax(stds))
        peak_layer = lnums[peak_idx]
        peak_std   = stds[peak_idx]
        top5       = sorted(range(len(stds)), key=lambda i: stds[i], reverse=True)[:5]
        rows.append({
            "model":          cfg["label"],
            "selected layer": f"layer_{peak_layer}",
            "selected std":   round(peak_std, 1),
            "top 5 layers":   [f"layer_{lnums[i]}" for i in top5],
        })
    return pd.DataFrame(rows)
