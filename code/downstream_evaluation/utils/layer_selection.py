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
    Optional key: current_layer — if present, draws a reference annotation for
                  that layer (used by select_analysis_layer.ipynb for comparison).
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

        # Peak layer
        ax.axvline(peak_layer, color=cfg["color"], lw=1.8, ls="--", zorder=2)
        x_offset = 1.5 if peak_layer < max(lnums) * 0.7 else -9
        ax.annotate(
            f"layer {peak_layer}  ← selected\nstd = {peak_std:.0f}",
            xy=(peak_layer, peak_std),
            xytext=(peak_layer + x_offset, peak_std * 0.82),
            fontsize=9, color=cfg["color"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.2),
        )

        # Optional reference layer annotation
        ref_layer = cfg.get("current_layer")
        if ref_layer is not None and ref_layer in lnums:
            current_std = stds[lnums.index(ref_layer)]
            pct         = 100 * current_std / peak_std
            ax.axvline(ref_layer, color="gray", lw=1.4, ls=":", zorder=2)
            ax.annotate(
                f"layer {ref_layer}  (previously used)\nstd = {current_std:.0f}",
                xy=(ref_layer, current_std),
                xytext=(ref_layer - 9, current_std + peak_std * 0.15),
                fontsize=8.5, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
            )
            ax.text(
                0.97, 0.97,
                f"Previously used layer retains\nonly {pct:.0f}% of peak signal",
                transform=ax.transAxes, fontsize=8.5, color="dimgray",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="lightgray", alpha=0.9),
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


def stratification_summary(models: list[dict]) -> pd.DataFrame:
    """
    Return a tidy DataFrame with peak and current-layer stats for each model.
    """
    rows = []
    for cfg in models:
        lnums, stds, _ = compute_stratification(cfg["csv"])
        peak_idx     = int(np.argmax(stds))
        peak_layer   = lnums[peak_idx]
        peak_std     = stds[peak_idx]
        current_std  = stds[lnums.index(cfg["current_layer"])]
        top5         = sorted(range(len(stds)), key=lambda i: stds[i], reverse=True)[:5]
        rows.append({
            "model":              cfg["label"],
            "selected layer":     f"layer_{peak_layer}",
            "selected std":       round(peak_std, 1),
            "previously used":    f"layer_{cfg['current_layer']}",
            "previous std":       round(current_std, 1),
            "signal retained (%)": round(100 * current_std / peak_std, 1),
            "top 5 layers":       [f"layer_{lnums[i]}" for i in top5],
        })
    return pd.DataFrame(rows)
