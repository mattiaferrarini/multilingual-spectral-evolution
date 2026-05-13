from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

from .checkpoints import apply_token_formatter, ckpt_to_tokens, format_tokens

PHASE_COLORS = {"entropy_seeking": "green", "compression_seeking": "darkorange"}
PHASE_LABELS = {"entropy_seeking": "Entropy-seeking", "compression_seeking": "Compression-seeking"}


def plot_rankme_phases(df_layer, df_phases, checkpoints_all, token_counts,
                       langs_sorted, model_label, layer, aggregation, plots_dir) -> None:
    """Per-language RankMe trajectory with entropy/compression phase shading."""
    ncols = 3
    nrows = -(-len(langs_sorted) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)

    for idx, lang in enumerate(langs_sorted):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        rv  = np.array([sub.loc[c, "rankme"] if c in sub.index else np.nan
                        for c in checkpoints_all])
        ax.plot(token_counts, rv, marker="o", lw=2, color="steelblue", ms=5)

        lang_row = df_phases[df_phases["language"] == lang]
        if not lang_row.empty:
            for pname, pcolor in PHASE_COLORS.items():
                p = lang_row.iloc[0]["phases"].get(pname)
                if p is not None:
                    ax.axvspan(p[0], p[1], alpha=0.15, color=pcolor)

        ax.set_title(lang, fontsize=10, fontweight="bold")
        apply_token_formatter(ax)
        ax.xaxis.label.set_size(8)
        ax.set_ylabel("RankMe", fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(langs_sorted), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    green_patch  = mpatches.Patch(color="green",      alpha=0.5, label="Entropy-seeking")
    orange_patch = mpatches.Patch(color="darkorange",  alpha=0.5, label="Compression-seeking")
    fig.legend(handles=[green_patch, orange_patch], loc="lower right", fontsize=9)
    fig.suptitle(f"[{model_label}] RankMe training phases — {layer}, agg={aggregation}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = Path(plots_dir) / "rankme_phases.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")


def plot_overlay(df_eval, df_layer, df_phases, df_grokking, task_languages, random_chance,
                 checkpoints_all, token_counts, langs_sorted, model_label, plots_dir) -> None:
    """Dual-axis RankMe + accuracy overlay per language, with phase shading."""
    for task in ["m_mmlu", "xcopa"]:
        df_task = df_eval[df_eval["task"] == task]
        overlap = [l for l in task_languages[task]
                   if l in langs_sorted and l in df_task["language"].unique()]
        if not overlap:
            print(f"[INFO] No overlapping languages for {task}")
            continue

        ncols = 3
        nrows = -(-len(overlap) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)

        for idx, lang in enumerate(overlap):
            ax  = axes[idx // ncols][idx % ncols]
            ax2 = ax.twinx()

            sub = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
            rv  = np.array([sub.loc[c, "rankme"] if c in sub.index else np.nan
                            for c in checkpoints_all])
            ax.plot(token_counts, rv, marker="o", lw=2, color="steelblue", ms=5)
            ax.set_ylabel("RankMe", color="steelblue", fontsize=8)
            ax.tick_params(axis="y", colors="steelblue")

            sub_e = df_task[df_task["language"] == lang].sort_values(
                "checkpoint", key=lambda s: s.map(ckpt_to_tokens))
            etoks = [ckpt_to_tokens(c) for c in sub_e["checkpoint"]]
            ax2.plot(etoks, sub_e["accuracy"], marker="s", lw=2, color="crimson", ms=5, ls="--")
            ax2.axhline(random_chance[task], color="gray", ls=":", alpha=0.6, lw=1)
            ax2.set_ylabel("Accuracy", color="crimson", fontsize=8)
            ax2.tick_params(axis="y", colors="crimson")

            lang_row = df_phases[df_phases["language"] == lang]
            if not lang_row.empty:
                for pname, pcolor in PHASE_COLORS.items():
                    p = lang_row.iloc[0]["phases"].get(pname)
                    if p:
                        ax.axvspan(p[0], p[1], alpha=0.08, color=pcolor)

            if not df_grokking.empty:
                grow = df_grokking[(df_grokking["task"] == task) &
                                   (df_grokking["language"] == lang)]
                if not grow.empty and not np.isnan(grow.iloc[0]["grokking_tokens"]):
                    ax.axvline(grow.iloc[0]["grokking_tokens"],
                               color="purple", ls="--", alpha=0.7, lw=1.5)

            ax.set_title(lang, fontsize=10, fontweight="bold")
            apply_token_formatter(ax)
            ax.xaxis.label.set_size(8)
            ax.grid(True, alpha=0.25)

        for idx in range(len(overlap), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"[{model_label}] RankMe vs {task.upper()} accuracy",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = Path(plots_dir) / f"overlay_{task}.png"
        plt.savefig(path)
        plt.show()
        print(f"Saved: {path}")


def plot_correlation_scatter(df_grokking, df_phases, model_label, plots_dir) -> None:
    """Scatter: compression onset vs peak accuracy, one panel per task."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, task in enumerate(["m_mmlu", "xcopa"]):
        ax   = axes[ax_idx]
        df_t = df_grokking[df_grokking["task"] == task] if not df_grokking.empty else pd.DataFrame()

        if df_t.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(task.upper(), fontsize=11)
            continue

        merged = df_t.merge(df_phases[["language", "compression_onset_tokens"]],
                             on="language", how="inner")
        valid  = merged.dropna(subset=["compression_onset_tokens", "peak_accuracy"])

        ax.scatter(valid["compression_onset_tokens"], valid["peak_accuracy"],
                   s=90, zorder=3, color="steelblue", edgecolors="k", lw=0.5)
        for _, row in valid.iterrows():
            ax.annotate(row["language"],
                        (row["compression_onset_tokens"], row["peak_accuracy"]),
                        fontsize=8, xytext=(5, 3), textcoords="offset points")

        title_suffix = "Insufficient data"
        if len(valid) >= 2:
            try:
                slope, intercept, r, p, _ = stats.linregress(
                    valid["compression_onset_tokens"], valid["peak_accuracy"])
                x_line = np.linspace(valid["compression_onset_tokens"].min(),
                                      valid["compression_onset_tokens"].max(), 100)
                ax.plot(x_line, slope * x_line + intercept,
                        color="crimson", alpha=0.6, lw=1.5, ls="--",
                        label=f"r = {r:.2f}, p = {p:.3f}")
                ax.legend(fontsize=9)
                title_suffix = f"Pearson r={r:.2f}, p={p:.3f}  (n={len(valid)})"
            except ValueError:
                title_suffix = f"n={len(valid)}"

        ax.xaxis.set_major_formatter(FuncFormatter(format_tokens))
        ax.set_xlabel("Compression-seeking onset", fontsize=10)
        ax.set_ylabel("Peak downstream accuracy", fontsize=10)
        ax.set_title(f"{task.upper()}\n{title_suffix}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"[{model_label}] Compression onset → downstream accuracy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = Path(plots_dir) / "correlation_scatter.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")
