"""
Layer selection utilities for downstream correlation analysis.

Computes cross-language RankMe stratification (std across languages at each
layer, averaged over checkpoints) and plots the result for both models.
"""

import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def layer_num(name: str) -> int:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0


def compute_stratification(csv, agg: str = "last") -> tuple:
    """
    For every layer compute std and mean of RankMe across languages
    (each language value is its mean over all checkpoints).

    Returns (layer_numbers, stds, means).
    """
    df = pd.read_csv(csv)
    df = df[df["aggregation"] == agg].copy()

    layers = sorted(df["layer"].unique(), key=layer_num)
    lnums  = [layer_num(l) for l in layers]

    mean_df = df.groupby(["layer", "dataset"])["rankme"].mean().reset_index()

    stds, means = [], []
    for layer in layers:
        vals = mean_df[mean_df["layer"] == layer]["rankme"].values
        stds.append(float(vals.std())  if len(vals) > 1 else 0.0)
        means.append(float(vals.mean()) if len(vals) > 0 else 0.0)

    return lnums, np.array(stds), np.array(means)


def plot_stratification(models: list[dict]) -> plt.Figure:
    """
    Two-panel figure: cross-language RankMe std vs. layer for each model.

    Required keys per model dict: label, csv, color
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Cross-language RankMe spread by layer  (checkpoint mean)\n"
        "Higher std = languages more discriminable = stronger signal for correlation analysis",
        fontsize=13, y=1.03,
    )

    for ax, cfg in zip(axes, models):
        lnums, stds, _ = compute_stratification(cfg["csv"])
        peak_idx   = int(np.argmax(stds))
        peak_layer = lnums[peak_idx]
        peak_std   = stds[peak_idx]

        ax.plot(lnums, stds,
                color=cfg["color"], lw=2.2, marker="o", ms=4, zorder=3)
        ax.fill_between(lnums, stds, alpha=0.12, color=cfg["color"])

        ax.axvline(peak_layer, color=cfg["color"], lw=1.8, ls="--", zorder=2)
        x_offset = 1.5 if peak_layer < max(lnums) * 0.7 else -9
        ax.annotate(
            f"layer {peak_layer}  ← selected\nstd = {peak_std:.0f}",
            xy=(peak_layer, peak_std),
            xytext=(peak_layer + x_offset, peak_std * 0.82),
            fontsize=9, color=cfg["color"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.2),
        )

        ax.set_title(cfg["label"], fontsize=12, pad=8)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Std(RankMe) across languages", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


def show_stratification(model_keys: list, data: dict) -> None:
    """
    Build, display, and save the cross-language stratification figure.

    Wraps plot_stratification() with the data-dict convention used by the
    notebook: data[m] must contain cfg (with model_label, rankme_csv,
    plots_dir) and color.
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
    fig = plot_stratification(models)
    save_path = data[model_keys[0]]["cfg"]["plots_dir"] / "layer_stratification.png"
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
