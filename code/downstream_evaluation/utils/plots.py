"""
Plotting functions for the downstream evaluation results notebook.

Per-model plots (trajectories, phase overlays, overlays):
  - plot_rankme_phases
  - plot_overlay
  - plot_alpha_phases
  - plot_alpha_overlay

Combined scatter plots (both models overlaid, one file per plot):
  - plot_predictor_scatter                (one predictor, 2×2: tasks × outcomes)
  - plot_alpha_rate_scatter_combined
  - plot_alpha_correlation_scatter_combined

All plots are saved to plots_dir and displayed inline.
"""

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

ALPHA_PHASE_COLORS = {"regularization_increasing": "dodgerblue", "regularization_decreasing": "darkorchid"}
ALPHA_PHASE_LABELS = {"regularization_increasing": "Reg.-increasing (α↓)", "regularization_decreasing": "Reg.-decreasing (α↑)"}

# Colors and markers used for each model in combined scatter plots.
_MODEL_PALETTE = [
    {"color": "steelblue",   "marker": "o"},
    {"color": "darkorange",  "marker": "s"},
    {"color": "seagreen",    "marker": "^"},
    {"color": "darkorchid",  "marker": "D"},
]


# ── Per-model plots ────────────────────────────────────────────────────────────

_COLOR_PRE  = "steelblue"
_COLOR_POST = "seagreen"
_COLOR_MIN  = "crimson"


def plot_rankme_phases(df_layer, checkpoints_all, token_counts,
                       langs_sorted, model_label, layer, aggregation, plots_dir,
                       model_key="", df_geometry=None) -> None:
    """
    Per-language RankMe trajectory split into Pre-minimum and Post-minimum phases.

    If df_geometry is provided (contains valley_tokens per language), each
    trajectory is coloured in two segments separated at the valley, and the
    minimum point is marked with a star. Otherwise falls back to a single colour.
    """
    ncols = 3
    nrows = -(-len(langs_sorted) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)

    tc = np.array(token_counts, dtype=float)
    valley_lookup = (
        dict(zip(df_geometry["language"], df_geometry["valley_tokens"]))
        if df_geometry is not None else {}
    )

    for idx, lang in enumerate(langs_sorted):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        rv  = np.array([sub.loc[c, "rankme"] if c in sub.index else np.nan
                        for c in checkpoints_all])

        valley_tok = valley_lookup.get(lang)
        if valley_tok is not None and not np.isnan(valley_tok):
            v_idx = int(np.argmin(np.abs(tc - float(valley_tok))))

            # Pre-minimum segment (first checkpoint … valley, inclusive)
            ax.plot(tc[:v_idx + 1], rv[:v_idx + 1],
                    marker="o", lw=2, color=_COLOR_PRE, ms=5)

            # Post-minimum segment (valley … last checkpoint, inclusive)
            if v_idx < len(tc) - 1:
                ax.plot(tc[v_idx:], rv[v_idx:],
                        marker="o", lw=2, color=_COLOR_POST, ms=5)

            # Valley marker
            ax.scatter([tc[v_idx]], [rv[v_idx]],
                       s=180, color=_COLOR_MIN, marker="*", zorder=5)
        else:
            ax.plot(tc, rv, marker="o", lw=2, color=_COLOR_PRE, ms=5)

        ax.set_title(lang, fontsize=10, fontweight="bold")
        apply_token_formatter(ax)
        ax.xaxis.label.set_size(8)
        ax.set_ylabel("RankMe", fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(langs_sorted), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Figure-level legend
    pre_patch  = mpatches.Patch(color=_COLOR_PRE,  label="Pre-minimum")
    post_patch = mpatches.Patch(color=_COLOR_POST, label="Post-minimum")
    min_handle = plt.Line2D([0], [0], marker="*", color="w",
                             markerfacecolor=_COLOR_MIN, markersize=13,
                             label="Minimum")
    fig.legend(handles=[pre_patch, post_patch, min_handle],
               loc="lower right", fontsize=10, ncol=3,
               framealpha=0.9, edgecolor="lightgray")

    fig.suptitle(f"[{model_label}] RankMe trajectories — {layer}, agg={aggregation}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    suffix = f"_{model_key}" if model_key else ""
    path = Path(plots_dir) / f"rankme_phases{suffix}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_overlay(df_eval, df_layer, df_grokking, task_languages, random_chance,
                 checkpoints_all, token_counts, langs_sorted, model_label, plots_dir,
                 model_key="", df_geometry=None) -> None:
    """Dual-axis RankMe + accuracy overlay per language, one figure per task."""
    tc = np.array(token_counts, dtype=float)
    valley_lookup = (
        dict(zip(df_geometry["language"], df_geometry["valley_tokens"]))
        if df_geometry is not None else {}
    )

    for task in task_languages:
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

            valley_tok = valley_lookup.get(lang)
            if valley_tok is not None and not np.isnan(valley_tok):
                v_idx = int(np.argmin(np.abs(tc - float(valley_tok))))
                ax.plot(tc[:v_idx + 1], rv[:v_idx + 1],
                        marker="o", lw=2, color=_COLOR_PRE, ms=5)
                if v_idx < len(tc) - 1:
                    ax.plot(tc[v_idx:], rv[v_idx:],
                            marker="o", lw=2, color=_COLOR_POST, ms=5)
                ax.scatter([tc[v_idx]], [rv[v_idx]],
                           s=180, color=_COLOR_MIN, marker="*", zorder=5)
            else:
                ax.plot(tc, rv, marker="o", lw=2, color=_COLOR_PRE, ms=5)

            ax.set_ylabel("RankMe", color=_COLOR_PRE, fontsize=8)
            ax.tick_params(axis="y", colors=_COLOR_PRE)

            sub_e = df_task[df_task["language"] == lang].sort_values(
                "checkpoint", key=lambda s: s.map(ckpt_to_tokens))
            etoks = [ckpt_to_tokens(c) for c in sub_e["checkpoint"]]
            ax2.plot(etoks, sub_e["accuracy"], marker="s", lw=2, color="crimson", ms=5, ls="--")
            ax2.axhline(random_chance[task], color="black", ls="--", alpha=0.9, lw=1.5)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Accuracy", color="crimson", fontsize=8)
            ax2.tick_params(axis="y", colors="crimson")

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

        # Figure-level legend
        pre_patch    = mpatches.Patch(color=_COLOR_PRE,  label="Pre-minimum (RankMe)")
        post_patch   = mpatches.Patch(color=_COLOR_POST, label="Post-minimum (RankMe)")
        min_handle   = plt.Line2D([0], [0], marker="*", color="w",
                                   markerfacecolor=_COLOR_MIN, markersize=13, label="Minimum")
        acc_handle   = plt.Line2D([0], [0], color="crimson", lw=2, ls="--",
                                   marker="s", markersize=6, label="Accuracy")
        grok_handle  = plt.Line2D([0], [0], color="purple", lw=1.5, ls="--",
                                   label="Grokking onset")
        fig.legend(handles=[pre_patch, post_patch, min_handle, acc_handle, grok_handle],
                   loc="lower right", fontsize=9, ncol=5,
                   framealpha=0.9, edgecolor="lightgray")

        task_label = task.replace("_", "-").upper()
        fig.suptitle(
            f"{model_label}  —  {task_label}\n"
            "RankMe trajectory vs. accuracy per language",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        suffix = f"_{model_key}" if model_key else ""
        path = Path(plots_dir) / f"overlay_{task}{suffix}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.show()
        print(f"Saved: {path}")


def plot_alpha_phases(df_layer, df_alpha_phases, checkpoints_all, token_counts,
                      langs_sorted, model_label, layer, aggregation, plots_dir,
                      model_key="") -> None:
    """Per-language AlphaReQ trajectory with regularization phase shading."""
    ncols = 3
    nrows = -(-len(langs_sorted) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)

    for idx, lang in enumerate(langs_sorted):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        av  = np.array([sub.loc[c, "alpha_req"] if c in sub.index else np.nan
                        for c in checkpoints_all])
        ax.plot(token_counts, av, marker="o", lw=2, color="steelblue", ms=5)

        lang_row = df_alpha_phases[df_alpha_phases["language"] == lang]
        if not lang_row.empty:
            for pname, pcolor in ALPHA_PHASE_COLORS.items():
                p = lang_row.iloc[0]["phases"].get(pname)
                if p is not None:
                    ax.axvspan(p[0], p[1], alpha=0.15, color=pcolor)

        ax.set_title(lang, fontsize=10, fontweight="bold")
        apply_token_formatter(ax)
        ax.xaxis.label.set_size(8)
        ax.set_ylabel("AlphaReQ (α)", fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(langs_sorted), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    blue_patch   = mpatches.Patch(color="dodgerblue",  alpha=0.5, label="Reg.-increasing (α↓)")
    purple_patch = mpatches.Patch(color="darkorchid",  alpha=0.5, label="Reg.-decreasing (α↑)")
    fig.legend(handles=[blue_patch, purple_patch], loc="lower right", fontsize=9)
    fig.suptitle(f"[{model_label}] AlphaReQ training phases — {layer}, agg={aggregation}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    suffix = f"_{model_key}" if model_key else ""
    path = Path(plots_dir) / f"alpha_phases{suffix}.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")


def plot_alpha_overlay(df_eval, df_layer, df_alpha_phases, df_grokking, task_languages,
                       random_chance, checkpoints_all, token_counts, langs_sorted,
                       model_label, plots_dir, model_key="") -> None:
    """Dual-axis AlphaReQ + accuracy overlay per language, with phase shading."""
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
            av  = np.array([sub.loc[c, "alpha_req"] if c in sub.index else np.nan
                            for c in checkpoints_all])
            ax.plot(token_counts, av, marker="o", lw=2, color="steelblue", ms=5)
            ax.set_ylabel("AlphaReQ (α)", color="steelblue", fontsize=8)
            ax.tick_params(axis="y", colors="steelblue")

            sub_e = df_task[df_task["language"] == lang].sort_values(
                "checkpoint", key=lambda s: s.map(ckpt_to_tokens))
            etoks = [ckpt_to_tokens(c) for c in sub_e["checkpoint"]]
            ax2.plot(etoks, sub_e["accuracy"], marker="s", lw=2, color="crimson", ms=5, ls="--")
            ax2.axhline(random_chance[task], color="black", ls="--", alpha=0.9, lw=1.5)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Accuracy", color="crimson", fontsize=8)
            ax2.tick_params(axis="y", colors="crimson")

            lang_row = df_alpha_phases[df_alpha_phases["language"] == lang]
            if not lang_row.empty:
                for pname, pcolor in ALPHA_PHASE_COLORS.items():
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

        fig.suptitle(f"[{model_label}] AlphaReQ vs {task.upper()} accuracy",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        suffix = f"_{model_key}" if model_key else ""
        path = Path(plots_dir) / f"alpha_overlay_{task}{suffix}.png"
        plt.savefig(path)
        plt.show()
        print(f"Saved: {path}")


# ── Combined scatter plots (both models overlaid) ─────────────────────────────
#
# Each function accepts `models_data`: a list of dicts with keys:
#   label        — model display name
#   df_grokking  — grokking DataFrame
#   df_geometry  — RankMe geometry DataFrame  (rankme scatter functions)
#   df_alpha_phases — AlphaReQ phases DataFrame  (alpha scatter functions)
#   color        — point/line color for this model
#   marker       — scatter marker style for this model

def _regression_label(valid, x_col, y_col):
    """Return (slope, intercept, title_str) or (None, None, reason_str)."""
    if len(valid) < 2:
        return None, None, f"n={len(valid)} (insufficient)"
    if valid[x_col].nunique() == 1:
        return None, None, f"n={len(valid)}, undefined (zero variance in x)"
    try:
        slope, intercept, pe_r, pe_p, _ = stats.linregress(valid[x_col], valid[y_col])
        sp_r, sp_p = stats.spearmanr(valid[x_col], valid[y_col])
        label = (f"Sp.r={sp_r:.2f}(p={sp_p:.3f}) | Pe.r={pe_r:.2f}(p={pe_p:.3f}) n={len(valid)}")
        return slope, intercept, label
    except ValueError:
        return None, None, f"n={len(valid)}"


def plot_predictor_scatter(models_data: list, x_col: str, x_label: str, plots_dir,
                           outcomes: list | None = None) -> None:
    """
    Scatter: one RankMe predictor vs selected outcomes × all tasks — both models overlaid.

    outcomes: list of (y_col, y_label) pairs to plot. Defaults to both grokking onset
              and peak accuracy. Pass a restricted list to exclude outcomes that are not
              methodologically valid for this predictor (e.g. exclude grokking onset for
              predictors that may use post-grokking data).
    """
    _TASK_LABELS = {"m_mmlu": "M-MMLU", "xcopa": "XCOPA",
                    "belebele": "Belebele", "m_arc": "M-ARC"}
    all_tasks = []
    for md in models_data:
        for t in (md["df_grokking"]["task"].unique() if not md["df_grokking"].empty else []):
            if t not in [x[0] for x in all_tasks]:
                all_tasks.append((t, _TASK_LABELS.get(t, t.upper())))
    if outcomes is None:
        outcomes = [("grokking_tokens", "Grokking onset (B)"), ("peak_accuracy", "Peak accuracy")]
    token_cols = {"valley_tokens", "grokking_tokens"}

    n_tasks   = len(all_tasks)
    n_outcomes = len(outcomes)
    fig, axes = plt.subplots(n_tasks, n_outcomes,
                             figsize=(7 * n_outcomes, 5 * n_tasks), squeeze=False)

    for row_idx, (task, task_label) in enumerate(all_tasks):
        for col_idx, (y_col, y_label) in enumerate(outcomes):
            ax = axes[row_idx, col_idx]
            title_lines = []
            has_data = False

            for i, md in enumerate(models_data):
                palette = _MODEL_PALETTE[i % len(_MODEL_PALETTE)]
                color  = md.get("color",  palette["color"])
                marker = md.get("marker", palette["marker"])
                label  = md["label"]

                df_t = md["df_grokking"]
                if df_t.empty:
                    continue
                df_t = df_t[df_t["task"] == task]
                if df_t.empty:
                    continue

                merged = df_t.merge(
                    md["df_geometry"][["language", x_col]],
                    on="language", how="inner")
                valid = merged.dropna(subset=[x_col, y_col])
                if valid.empty:
                    continue

                has_data = True
                ax.scatter(valid[x_col], valid[y_col],
                           s=90, zorder=3, color=color, edgecolors="k", lw=0.5,
                           marker=marker, label=label)
                for _, row in valid.iterrows():
                    ax.annotate(row["language"],
                                (row[x_col], row[y_col]),
                                fontsize=8, xytext=(5, 3), textcoords="offset points",
                                color=color)

                slope, intercept, stat_str = _regression_label(valid, x_col, y_col)
                if slope is not None:
                    x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
                    ax.plot(x_line, slope * x_line + intercept,
                            color=color, alpha=0.6, lw=1.5, ls="--")
                title_lines.append(f"{label}: {stat_str}")

            if not has_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")

            if x_col in token_cols:
                ax.xaxis.set_major_formatter(FuncFormatter(format_tokens))
            if y_col in token_cols:
                ax.yaxis.set_major_formatter(FuncFormatter(format_tokens))

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f"{task_label} — {y_label}\n" + "\n".join(title_lines), fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"{x_label} → downstream performance  [both models]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = Path(plots_dir) / f"scatter_{x_col}.png"
    plt.savefig(path, dpi=100)
    plt.show()
    print(f"Saved: {path}")


def plot_correlation_heatmap(models_data: list, plots_dir) -> None:
    """
    Two-panel heatmap: rows = 9 RankMe predictors, columns = 4 outcomes (task × measure).
    Color encodes Spearman ρ; * marks p < 0.05; gray = undefined (zero variance).

    Required key per model dict: label, df_correlations, color.
    """
    _SHORT = {
        "RankMe at first ckpt":              "RankMe first ckpt",
        "RankMe at valley":                  "RankMe at valley",
        "Initial RankMe drop rate (1/B)":    "Initial drop rate (1/B)",
        "RankMe at last ckpt":               "RankMe last ckpt",
        "Mean RankMe — first 25% of tokens": "Mean 0–25%",
        "Mean RankMe — first 50% of tokens": "Mean 0–50%",
        "Mean RankMe — last 50% of tokens":  "Mean 50–100%",
        "Mean RankMe — last 25% of tokens":  "Mean 75–100%",
        "Mean RankMe — before valley":       "Mean before valley",
        "Mean RankMe — after valley":        "Mean after valley",
    }
    _PRED_ORDER = list(_SHORT.keys())
    _COLS = [
        ("m_mmlu",   "Grokking onset (B)", "M-MMLU\nGrokking onset"),
        ("m_mmlu",   "Peak accuracy",      "M-MMLU\nPeak accuracy"),
        ("xcopa",    "Grokking onset (B)", "XCOPA\nGrokking onset"),
        ("xcopa",    "Peak accuracy",      "XCOPA\nPeak accuracy"),
        ("belebele", "Grokking onset (B)", "Belebele\nGrokking onset"),
        ("belebele", "Peak accuracy",      "Belebele\nPeak accuracy"),
        ("m_arc",    "Grokking onset (B)", "M-ARC\nGrokking onset"),
        ("m_arc",    "Peak accuracy",      "M-ARC\nPeak accuracy"),
    ]

    n    = len(models_data)
    fig, axes = plt.subplots(1, n, figsize=(max(6, len(_COLS) * 0.9) * n, 8))
    if n == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")
    im = None

    for ax, md in zip(axes, models_data):
        df = md.get("df_correlations", pd.DataFrame())

        mat_r = np.full((len(_PRED_ORDER), len(_COLS)), np.nan)
        mat_p = np.full((len(_PRED_ORDER), len(_COLS)), np.nan)

        if not df.empty:
            for ci, (task, outcome, _) in enumerate(_COLS):
                for ri, pred in enumerate(_PRED_ORDER):
                    row = df[(df["task"] == task) &
                             (df["predictor (x)"] == pred) &
                             (df["outcome (y)"] == outcome)]
                    if not row.empty:
                        mat_r[ri, ci] = float(row["spearman_r"].iloc[0])
                        mat_p[ri, ci] = float(row["spearman_p"].iloc[0])

        masked = np.ma.masked_invalid(mat_r)
        im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(np.arange(-0.5, len(_COLS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(_PRED_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xticks(range(len(_COLS)))
        ax.set_xticklabels([c[2] for c in _COLS], fontsize=9, rotation=30)
        ax.set_yticks(range(len(_PRED_ORDER)))
        ax.set_yticklabels([_SHORT[p] for p in _PRED_ORDER], fontsize=9)

        for ri in range(len(_PRED_ORDER)):
            for ci in range(len(_COLS)):
                r = mat_r[ri, ci]
                p = mat_p[ri, ci]
                if np.isnan(r):
                    ax.text(ci, ri, "—", ha="center", va="center",
                            fontsize=9, color="gray")
                    continue
                star   = "*" if (not np.isnan(p) and p < 0.05) else ""
                txt    = f"{r:.2f}{star}"
                color  = "white" if abs(r) > 0.55 else "black"
                weight = "bold" if star else "normal"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight)

        ax.set_title(md["label"], fontsize=11, pad=10)

    fig.suptitle(
        "Spearman ρ: RankMe geometry → downstream performance\n"
        "* = p < 0.05  |  gray = undefined (zero variance)",
        fontsize=12, fontweight="bold",
    )
    # tight_layout first, leaving a strip on the right for the colorbar
    plt.tight_layout(rect=[0, 0, 0.91, 0.93])
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.62])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label("Spearman ρ", fontsize=10)
    path = Path(plots_dir) / "correlation_heatmap.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_alpha_correlation_scatter_combined(models_data: list, plots_dir) -> None:
    """Scatter: AlphaReQ trough position vs peak accuracy — both models overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, task in enumerate(["m_mmlu", "xcopa"]):
        ax = axes[ax_idx]
        title_lines = []
        has_data = False

        for i, md in enumerate(models_data):
            palette = _MODEL_PALETTE[i % len(_MODEL_PALETTE)]
            color  = md.get("color",  palette["color"])
            marker = md.get("marker", palette["marker"])
            label  = md["label"]

            df_t = md["df_grokking"]
            if df_t.empty:
                continue
            df_t = df_t[df_t["task"] == task]
            if df_t.empty:
                continue

            merged = df_t.merge(
                md["df_alpha_phases"][["language", "reg_decreasing_onset_tokens"]],
                on="language", how="inner")
            valid  = merged.dropna(subset=["reg_decreasing_onset_tokens", "peak_accuracy"])
            if valid.empty:
                continue

            has_data = True
            ax.scatter(valid["reg_decreasing_onset_tokens"], valid["peak_accuracy"],
                       s=90, zorder=3, color=color, edgecolors="k", lw=0.5,
                       marker=marker, label=label)
            for _, row in valid.iterrows():
                ax.annotate(row["language"],
                            (row["reg_decreasing_onset_tokens"], row["peak_accuracy"]),
                            fontsize=7, xytext=(5, 3), textcoords="offset points",
                            color=color)

            slope, intercept, stat_str = _regression_label(
                valid, "reg_decreasing_onset_tokens", "peak_accuracy")
            if slope is not None:
                x_line = np.linspace(valid["reg_decreasing_onset_tokens"].min(),
                                     valid["reg_decreasing_onset_tokens"].max(), 100)
                ax.plot(x_line, slope * x_line + intercept,
                        color=color, alpha=0.6, lw=1.5, ls="--")
            title_lines.append(f"{label}: {stat_str}")

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)

        ax.xaxis.set_major_formatter(FuncFormatter(format_tokens))
        ax.set_xlabel("Reg.-decreasing onset (AlphaReQ trough)", fontsize=10)
        ax.set_ylabel("Peak downstream accuracy", fontsize=10)
        ax.set_title(f"{task.upper()}\n" + "\n".join(title_lines) if title_lines
                     else task.upper(), fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("AlphaReQ trough position → downstream accuracy  [both models]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = Path(plots_dir) / "alpha_correlation_scatter.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")


def plot_checkpoint_correlations(df_ckpt_corr: pd.DataFrame, model_label: str,
                                 plots_dir, model_key: str = "") -> None:
    """
    Plot per-checkpoint cross-language Spearman(RankMe, accuracy) for both tasks.

    At each training checkpoint the correlation is computed across all languages
    that have both RankMe and accuracy data for that task. Filled markers indicate
    p < 0.05; open markers indicate p ≥ 0.05.
    """
    _TASK_LABELS = {"m_mmlu": "M-MMLU", "xcopa": "XCOPA",
                    "belebele": "Belebele", "m_arc": "M-ARC"}
    _TASK_COLORS = ["steelblue", "darkorange", "seagreen", "darkorchid"]
    tasks = [(t, _TASK_LABELS.get(t, t.upper()))
             for t in df_ckpt_corr["task"].unique()]
    colors = {t: _TASK_COLORS[i % len(_TASK_COLORS)] for i, (t, _) in enumerate(tasks)}

    n_tasks = len(tasks)
    ncols = min(n_tasks, 2)
    nrows = -(-n_tasks // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharey=True,
                             squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    for ax, (task, task_label) in zip(axes_flat, tasks):
        df_t  = df_ckpt_corr[df_ckpt_corr["task"] == task].copy()
        color = colors[task]

        if df_t.empty or df_t["spearman_r"].isna().all():
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(task_label)
            continue

        tc  = df_t["token_count"].values
        rho = df_t["spearman_r"].values
        p   = df_t["spearman_p"].values

        valid = ~np.isnan(rho)
        ax.plot(tc[valid], rho[valid], color=color, lw=1.5, alpha=0.5)

        sig   = valid & (p < 0.05)
        insig = valid & ~sig
        if insig.any():
            ax.scatter(tc[insig], rho[insig], s=45, facecolors="none",
                       edgecolors=color, lw=1.5, zorder=4)
        if sig.any():
            ax.scatter(tc[sig], rho[sig], s=60, color=color,
                       edgecolors="k", lw=0.5, zorder=5)

        n_modal = int(df_t.loc[df_t["n_languages"] > 0, "n_languages"].mode().iloc[0])
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(f"{task_label}  (n = {n_modal} languages)",
                     fontsize=11, fontweight="bold")
        apply_token_formatter(ax)
        ax.set_xlabel("Training tokens", fontsize=9)
        ax.grid(True, alpha=0.3)

        filled_h = mpatches.Patch(color=color, label="p < 0.05")
        open_h   = plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="none", markeredgecolor=color,
                               markersize=7, label="p ≥ 0.05")
        ax.legend(handles=[filled_h, open_h], fontsize=8)

    axes_flat[0].set_ylabel("Spearman ρ  (RankMe vs accuracy, across languages)", fontsize=9)
    fig.suptitle(
        f"[{model_label}] Cross-language Spearman(RankMe, accuracy) per checkpoint",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    suffix = f"_{model_key}" if model_key else ""
    path   = Path(plots_dir) / f"checkpoint_correlations{suffix}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_alpha_rate_scatter_combined(models_data: list, plots_dir) -> None:
    """Scatter: rate of α decline vs grokking onset — both models overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, task in enumerate(["m_mmlu", "xcopa"]):
        ax = axes[ax_idx]
        title_lines = []
        has_data = False

        for i, md in enumerate(models_data):
            palette = _MODEL_PALETTE[i % len(_MODEL_PALETTE)]
            color  = md.get("color",  palette["color"])
            marker = md.get("marker", palette["marker"])
            label  = md["label"]

            df_t = md["df_grokking"]
            if df_t.empty:
                continue
            df_t = df_t[df_t["task"] == task]
            if df_t.empty:
                continue

            merged = df_t.merge(
                md["df_alpha_phases"][["language", "reg_increasing_rate"]],
                on="language", how="inner")
            valid  = merged.dropna(subset=["reg_increasing_rate", "grokking_tokens"])
            if valid.empty:
                continue

            has_data = True
            ax.scatter(valid["reg_increasing_rate"], valid["grokking_tokens"],
                       s=90, zorder=3, color=color, edgecolors="k", lw=0.5,
                       marker=marker, label=label)
            for _, row in valid.iterrows():
                ax.annotate(row["language"],
                            (row["reg_increasing_rate"], row["grokking_tokens"]),
                            fontsize=7, xytext=(5, 3), textcoords="offset points",
                            color=color)

            slope, intercept, stat_str = _regression_label(
                valid, "reg_increasing_rate", "grokking_tokens")
            if slope is not None:
                x_line = np.linspace(valid["reg_increasing_rate"].min(),
                                     valid["reg_increasing_rate"].max(), 100)
                ax.plot(x_line, slope * x_line + intercept,
                        color=color, alpha=0.6, lw=1.5, ls="--")
            title_lines.append(f"{label}: {stat_str}")

        if not has_data:
            ax.text(0.5, 0.5, "No grokking detected", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")

        ax.yaxis.set_major_formatter(FuncFormatter(format_tokens))
        ax.set_xlabel("Rate of α decline (α/B)", fontsize=10)
        ax.set_ylabel("Grokking onset (B)", fontsize=10)
        ax.set_title(f"{task.upper()}\n" + "\n".join(title_lines) if title_lines
                     else task.upper(), fontsize=9)
        if has_data:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Rate of α decline → grokking onset  [both models]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = Path(plots_dir) / "alpha_rate_scatter.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")

