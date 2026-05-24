"""
Visualization script for geometry analysis results.

Generates three plot types from a metrics CSV:
  - training_curves : metric vs checkpoint (one line per language)
  - layer_profiles  : metric vs layer     (one line per checkpoint)
  - heatmaps        : checkpoint x layer  (colored by metric)

Usage examples:
  # All plots, all data
  python geometry_analysis/visualize.py --csv results/fuxi.csv

  # Only training curves, RankMe only, last-token aggregation
  python geometry_analysis/visualize.py --csv results/fuxi.csv \
      --plot-types training_curves --metrics rankme --aggregations last

  # Heatmaps for two languages, all metrics
  python geometry_analysis/visualize.py --csv results/fuxi.csv \
      --plot-types heatmaps --languages English French
"""

import argparse
import json
import math
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

METRIC_LABELS = {
    "rankme": "RankMe",
    "pr": "Participation Ratio",
    "alpha_req": "AlphaReQ",
    "top_5_var": "Top-5 Variance",
    "top_10_var": "Top-10 Variance",
    "top_20_var": "Top-20 Variance",
    "top_50_var": "Top-50 Variance",
    "top_100_var": "Top-100 Variance",
    # Extra:
    "top_250_var": "Top-250 Variance",
    "top_500_var": "Top-500 Variance",
    "top_1000_var": "Top-1000 Variance",
}


# ── Checkpoint ordering ───────────────────────────────────────────────────────

def _auto_ticks(lo, hi, n_target=7):
    """Generate ~n_target nice round tick positions spanning [lo, hi]."""
    if hi <= lo:
        return [round(lo)]
    span = hi - lo
    raw_step = span / max(n_target - 1, 1)
    mag = 10 ** math.floor(math.log10(raw_step))
    step = mag
    for factor in [1, 2, 2.5, 5, 10]:
        step = mag * factor
        if span / step <= n_target + 1:
            break
    start = math.ceil(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t))
        t += step
    return ticks


def _tick_label(val):
    """Format a billion-token value: ≥1000B shown as T (e.g. 1000→'1T', 1500→'1.5T')."""
    if val >= 1000:
        t = val / 1000
        return f"{int(t)}T" if t == int(t) else f"{t:.1f}T"
    return str(int(val))


def _ckpt_key(name):
    if str(name).lower() == "main":
        return float("inf")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([BT]?)$", str(name), re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    # Apertus format: step{N}-tokens{M}[BT]
    m = re.match(r"step\d+-tokens(\d+)([BT])", str(name), re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    return float("inf") - 1


def sort_checkpoints(checkpoints):
    return sorted(checkpoints, key=_ckpt_key)


def layer_num(layer_name):
    m = re.search(r"(\d+)", str(layer_name))
    return int(m.group(1)) if m else 0


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _language_colors(languages):
    n = max(len(languages), 1)
    if n <= 10:
        palette = sns.color_palette("tab10", n_colors=n)
    elif n <= 20:
        palette = sns.color_palette("tab20", n_colors=n)
    else:
        palette = sns.color_palette("husl", n_colors=n)
    return {lang: palette[i] for i, lang in enumerate(languages)}


def _minmax(ys):
    arr = np.array(ys, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi == lo:
        return arr.tolist()
    return ((arr - lo) / (hi - lo)).tolist()


def _smooth_on_uniform_grid(xs, ys, sigma):
    from scipy.ndimage import gaussian_filter1d
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    nan_mask = np.isnan(ys_arr)
    if nan_mask.all() or sigma <= 0:
        return ys
    valid_xs = xs_arr[~nan_mask]
    valid_ys = ys_arr[~nan_mask]
    n = len(xs_arr)
    x_uniform = np.linspace(xs_arr[0], xs_arr[-1], n)
    y_uniform = np.interp(x_uniform, valid_xs, valid_ys)
    y_smoothed_uniform = gaussian_filter1d(y_uniform, sigma=sigma)
    y_smoothed = np.interp(xs_arr, x_uniform, y_smoothed_uniform)
    y_smoothed[nan_mask] = np.nan
    return y_smoothed.tolist()


def _checkpoint_colors(checkpoints):
    cmap = plt.get_cmap("viridis")
    n = max(len(checkpoints), 1)
    return {ckpt: cmap(i / (n - 1) if n > 1 else 0.5) for i, ckpt in enumerate(checkpoints)}


def _curve_turning_point(xs, ys, min_x=None):
    arr = np.array(ys, dtype=float)
    xs_arr = np.array(xs, dtype=float)
    finite = np.isfinite(arr)
    if min_x is not None:
        finite &= xs_arr >= min_x
    if finite.sum() < 2:
        return None

    finite_idx = np.nonzero(finite)[0]
    idx = int(finite_idx[np.nanargmax(arr[finite])])
    return xs_arr[idx], arr[idx]


def _global_turning_point(subset, checkpoints, layer_nums, metric):
    best_point = None
    best_value = None

    for ckpt in checkpoints:
        vals = (
            subset[subset["checkpoint"] == ckpt]
            .set_index("layer")
            .reindex(layer_nums)[metric]
            .to_numpy()
        )
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        idx = int(np.nanargmax(vals))
        value = float(vals[idx])
        if best_value is None or value > best_value:
            best_value = value
            best_point = (float(layer_num(layer_nums[idx])), ckpt, value)

    return best_point


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Plot type 1: training curves ─────────────────────────────────────────────

def plot_training_curves(df, metrics, languages, layers, aggregations, output_dir, model_label,
                         y_ranges=None, smoothing=0.0, normalize="none"):
    from collections import Counter
    checkpoints = sort_checkpoints(df["checkpoint"].unique())
    lang_colors = _language_colors(languages)

    numeric_ckpts = [c for c in checkpoints if str(c).lower() != "main"]
    numeric_xs = [_ckpt_key(c) for c in numeric_ckpts]
    min_numeric = min(numeric_xs, default=0)
    max_numeric = max(numeric_xs, default=0)
    main_x = max_numeric  # place "main" at the rightmost position

    def ckpt_x(c):
        return main_x if str(c).lower() == "main" else _ckpt_key(c)

    max_x = max_numeric
    tick_positions = _auto_ticks(min_numeric, max_x)
    tick_labels = [_tick_label(t) for t in tick_positions]
    xs_common = [ckpt_x(c) for c in checkpoints]

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        print(f"  {metric_label}...")
        for layer in layers:
            for agg in aggregations:
                subset = df[(df["layer"] == layer) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(12, 6))
                endpoints = []
                min_data_x = float("inf")
                smoothed_matrix = []
                raw_ys_per_lang = {}

                for lang in languages:
                    row = subset[subset["dataset"] == lang].set_index("checkpoint")
                    ys = [row.loc[c, metric] if c in row.index else np.nan for c in checkpoints]
                    raw_ys_per_lang[lang] = ys
                    valid_xs = [x for x, y in zip(xs_common, ys) if not np.isnan(y)]
                    if valid_xs:
                        min_data_x = min(min_data_x, valid_xs[0])
                    if smoothing > 0:
                        smoothed_matrix.append(_smooth_on_uniform_grid(xs_common, ys, smoothing))

                left_x = min_data_x if min_data_x < float("inf") else tick_positions[0]

                # change-point detection (only when smoothing is active)
                per_lang_cps = []
                if smoothing > 0 and smoothed_matrix:
                    xs_arr = np.array(xs_common)
                    for smoothed_ys in smoothed_matrix:
                        s = np.array(smoothed_ys, dtype=float)
                        valid_idx = [i for i, v in enumerate(s) if not np.isnan(v)]
                        if len(valid_idx) >= 2:
                            min_i = min(valid_idx, key=lambda i, s=s: s[i])
                            max_i = max(valid_idx, key=lambda i, s=s: s[i])

                            def _deriv_jump(idx):
                                lefts = [j for j in valid_idx if j < idx]
                                rights = [j for j in valid_idx if j > idx]
                                if not lefts or not rights:
                                    return 0.0
                                prev_i, next_i = lefts[-1], rights[0]
                                d_l = (s[idx] - s[prev_i]) / (xs_arr[idx] - xs_arr[prev_i])
                                d_r = (s[next_i] - s[idx]) / (xs_arr[next_i] - xs_arr[idx])
                                return abs(d_r - d_l)

                            ext_i = min_i if _deriv_jump(min_i) >= _deriv_jump(max_i) else max_i
                            lefts = [j for j in valid_idx if j < ext_i]
                            rights = [j for j in valid_idx if j > ext_i]
                            if lefts and rights:
                                prev_i, next_i = lefts[-1], rights[0]
                                d_left = (s[ext_i] - s[prev_i]) / (xs_arr[ext_i] - xs_arr[prev_i])
                                d_right = (s[next_i] - s[ext_i]) / (xs_arr[next_i] - xs_arr[ext_i])
                                denom = abs(d_left) + abs(d_right)
                                frac = abs(d_left) / denom if denom > 0 else 0.5
                                x_cross = xs_arr[ext_i] + frac * (xs_arr[next_i] - xs_arr[ext_i])
                                per_lang_cps.append(min(xs_common, key=lambda x, x_cross=x_cross: abs(x - x_cross)))
                            else:
                                per_lang_cps.append(xs_arr[ext_i])

                mode_cp = Counter(per_lang_cps).most_common(1)[0][0] if per_lang_cps else None

                def _apply_normalize(ys, xs):
                    if normalize == "global":
                        return _minmax(ys)
                    if normalize == "per-segment" and mode_cp is not None:
                        arr = np.array(ys, dtype=float)
                        cp_idx = xs.index(mode_cp)
                        pre, post = arr[:cp_idx + 1].copy(), arr[cp_idx:].copy()
                        result = np.full_like(arr, np.nan)
                        result[:cp_idx + 1] = _minmax(pre.tolist())
                        result[cp_idx:] = _minmax(post.tolist())
                        return result.tolist()
                    return ys

                for lang in languages:
                    ys = raw_ys_per_lang[lang]
                    plot_ys = _apply_normalize(ys, xs_common)
                    ax.plot(xs_common, plot_ys, color=lang_colors[lang], linewidth=1.5)
                    valid = [(x, y) for x, y in zip(xs_common, plot_ys) if not np.isnan(y)]
                    if valid:
                        endpoints.append((valid[-1][0], valid[-1][1], lang, lang_colors[lang]))

                # shaded regions around change point
                if mode_cp is not None:
                    ax.axvspan(left_x, mode_cp, color="steelblue", alpha=0.15, zorder=0)
                    ax.axvspan(mode_cp, max_x, color="tomato", alpha=0.15, zorder=0)
                    ax.axvline(mode_cp, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
                               label=f"Change point (mode): {mode_cp:.0f}B")
                    ax.legend(fontsize=11, loc="upper right")

                # right-side endpoint labels
                endpoints.sort(key=lambda t: t[1])
                label_ys = [t[1] for t in endpoints]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                min_gap = y_range * 0.05
                for _ in range(50):
                    changed = False
                    for i in range(1, len(label_ys)):
                        if label_ys[i] - label_ys[i - 1] < min_gap:
                            mid = (label_ys[i] + label_ys[i - 1]) / 2
                            label_ys[i - 1] = mid - min_gap / 2
                            label_ys[i] = mid + min_gap / 2
                            changed = True
                    if not changed:
                        break

                y_min, y_max = ax.get_ylim()
                for (lx, ly, lang, color), label_y in zip(endpoints, label_ys):
                    label_y_frac = (label_y - y_min) / (y_max - y_min)
                    ax.annotate(lang, xy=(lx, ly), xycoords="data",
                                xytext=(1.04, label_y_frac), textcoords="axes fraction",
                                color=color, fontsize=12, va="center", clip_on=False,
                                arrowprops={"arrowstyle": "-", "color": color,
                                            "lw": 1, "alpha": 0.5, "relpos": (0, 0.5)})

                ax.set_xlim(left=left_x, right=max_x)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, fontsize=12)
                ax.set_xlabel("Billion of tokens", fontsize=13)
                ax.set_ylabel(metric_label, fontsize=13)
                if y_ranges and (metric, agg) in y_ranges:
                    ax.set_ylim(y_ranges[(metric, agg)])
                title = f"{metric_label} over training — {layer} | agg={agg}"
                if model_label:
                    title = f"[{model_label}] " + title
                ax.set_title(title, fontsize=14)
                ax.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout()

                fname = f"training_curve_{metric}_{layer}_{agg}.png"
                _save(fig, os.path.join(output_dir, "training_curves", metric, fname))


# ── Plot type 2: layer profiles ───────────────────────────────────────────────

def plot_layer_profiles(df, metrics, languages, layers, aggregations, output_dir, model_label, y_ranges=None):
    checkpoints = sort_checkpoints(df["checkpoint"].unique())
    ckpt_colors = _checkpoint_colors(checkpoints)
    layer_nums = sorted(layers, key=layer_num)
    x = [layer_num(l) for l in layer_nums]

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        print(f"  {metric_label}...")
        for lang in languages:
            for agg in aggregations:
                subset = df[(df["dataset"] == lang) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(12, 6))
                for ckpt in checkpoints:
                    vals = (
                        subset[subset["checkpoint"] == ckpt]
                        .set_index("layer")
                        .reindex(layer_nums)[metric]
                    )
                    ax.plot(x, vals, marker="o", label=ckpt,
                            color=ckpt_colors[ckpt], linewidth=2, markersize=5)

                main_ckpt = next((ckpt for ckpt in checkpoints if str(ckpt).lower() == "main"), None)
                compressed_turning_point = None
                if main_ckpt is not None:
                    main_vals = (
                        subset[subset["checkpoint"] == main_ckpt]
                        .set_index("layer")
                        .reindex(layer_nums)[metric]
                        .to_numpy()
                    )
                    compressed_turning_point = _curve_turning_point(x, main_vals, min_x=3)

                early_turning_point = _global_turning_point(subset, checkpoints, layer_nums, metric)

                turning_point_handles = []
                early_line_color = "#0072B2"  # Professional blue
                compressed_line_color = "#E69F00"  # Professional orange

                ax.set_xlabel("Layer", fontsize=12)
                ax.set_ylabel(metric_label, fontsize=12)
                if y_ranges and (metric, agg) in y_ranges:
                    ax.set_ylim(y_ranges[(metric, agg)])
                title = f"{metric_label} by layer — {lang} | agg={agg}"
                if model_label:
                    title = f"[{model_label}] " + title
                ax.set_title(title, fontsize=13)
                if early_turning_point is not None:
                    early_x, early_ckpt, _ = early_turning_point
                    ax.axvline(early_x, color=early_line_color, linestyle=(0, (6, 4)), linewidth=1.4,
                               alpha=0.85, zorder=0)
                    turning_point_handles.append(
                        Line2D([0], [0], color=early_line_color, linestyle=(0, (6, 4)), linewidth=1.4,
                               label=f"Early training turning point ({early_ckpt})")
                    )
                if compressed_turning_point is not None:
                    compressed_x, _ = compressed_turning_point
                    ax.axvline(compressed_x, color=compressed_line_color, linestyle=(0, (6, 4)), linewidth=1.4,
                               alpha=0.85, zorder=0)
                    turning_point_handles.append(
                        Line2D([0], [0], color=compressed_line_color, linestyle=(0, (6, 4)), linewidth=1.4,
                               label="Compressed turning point")
                    )
                n_ckpts = len(checkpoints)
                ncol = min(8, max(3, (n_ckpts + 3) // 4))  # Adaptive columns based on checkpoint count
                checkpoint_legend = ax.legend(title="Checkpoint", fontsize=8, title_fontsize=9,
                                              ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                                              frameon=True, framealpha=0.95)
                if turning_point_handles:
                    tp_legend = ax.legend(handles=turning_point_handles, title="Turning points",
                                          fontsize=8, title_fontsize=9, loc="upper right",
                                          frameon=True, framealpha=0.95)
                    ax.add_artist(checkpoint_legend)
                    ax.add_artist(tp_legend)
                else:
                    ax.add_artist(checkpoint_legend)
                ax.grid(True, alpha=0.3)
                fig.tight_layout(rect=(0, 0.14, 1, 1))

                fname = f"layer_profile_{metric}_{lang}_{agg}.png"
                _save(fig, os.path.join(output_dir, "layer_profiles", metric, fname))

                plot_data = {
                    "metric": metric,
                    "language": lang,
                    "aggregation": agg,
                    "layers": [layer_num(l) for l in layer_nums],
                    "checkpoints": [str(c) for c in checkpoints],
                    "checkpoint_data": {},
                    "turning_points": {},
                }
                for ckpt in checkpoints:
                    vals = (
                        subset[subset["checkpoint"] == ckpt]
                        .set_index("layer")
                        .reindex(layer_nums)[metric]
                        .to_numpy()
                    )
                    plot_data["checkpoint_data"][str(ckpt)] = [float(v) if np.isfinite(v) else None for v in vals]

                if early_turning_point is not None:
                    early_x, early_ckpt, early_val = early_turning_point
                    plot_data["turning_points"]["early_training"] = {
                        "layer": float(early_x),
                        "checkpoint": str(early_ckpt),
                        "value": float(early_val),
                        "color": early_line_color,
                    }
                if compressed_turning_point is not None:
                    compressed_x, compressed_val = compressed_turning_point
                    plot_data["turning_points"]["compressed"] = {
                        "layer": float(compressed_x),
                        "checkpoint": str(main_ckpt) if main_ckpt else "unknown",
                        "value": float(compressed_val),
                        "color": compressed_line_color,
                    }

                json_fname = f"layer_profile_{metric}_{lang}_{agg}.json"
                _save_json(plot_data, os.path.join(output_dir, "layer_profiles", metric, json_fname))

        # ── All-languages summary plot (mean across checkpoints) ──────────────
        for agg in aggregations:
            fig, ax = plt.subplots(figsize=(12, 6))
            lang_colors = _language_colors(languages)
            for lang in languages:
                subset = df[(df["dataset"] == lang) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue
                mean_vals = (
                    subset.groupby("layer")[metric].mean()
                    .reindex(layer_nums)
                )
                ax.plot(x, mean_vals.values, marker="o", label=lang,
                        color=lang_colors[lang], linewidth=2, markersize=4)

            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            if y_ranges and (metric, agg) in y_ranges:
                ax.set_ylim(y_ranges[(metric, agg)])
            title = f"{metric_label} by layer — all languages (checkpoint mean) | agg={agg}"
            if model_label:
                title = f"[{model_label}] " + title
            ax.set_title(title, fontsize=13)
            n_langs = len(languages)
            ncol = min(7, max(3, (n_langs + 2) // 3))
            ax.legend(title="Language", fontsize=9, title_fontsize=10,
                     ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                     frameon=True, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            fig.tight_layout(rect=(0, 0.14, 1, 1))

            fname = f"layer_profile_{metric}_all_languages_{agg}.png"
            _save(fig, os.path.join(output_dir, "layer_profiles", metric, fname))


# ── Plot type 3: heatmaps ─────────────────────────────────────────────────────

def plot_heatmaps(df, metrics, languages, layers, aggregations, output_dir, model_label, y_ranges=None):
    checkpoints = sort_checkpoints(df["checkpoint"].unique())
    layer_nums_sorted = sorted(layers, key=layer_num)
    x_labels = [layer_num(l) for l in layer_nums_sorted]

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        print(f"  {metric_label}...")
        for lang in languages:
            for agg in aggregations:
                vmin = y_ranges[(metric, agg)][0] if y_ranges and (metric, agg) in y_ranges else None
                vmax = y_ranges[(metric, agg)][1] if y_ranges and (metric, agg) in y_ranges else None
                subset = df[(df["dataset"] == lang) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                matrix = np.full((len(checkpoints), len(layer_nums_sorted)), np.nan)
                for r, ckpt in enumerate(checkpoints):
                    for c, layer in enumerate(layer_nums_sorted):
                        row = subset[(subset["checkpoint"] == ckpt) & (subset["layer"] == layer)]
                        if not row.empty:
                            matrix[r, c] = row[metric].values[0]

                fig_w = max(8, len(layer_nums_sorted) * 0.8)
                fig_h = max(5, len(checkpoints) * 1.0)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                sns.heatmap(
                    matrix,
                    xticklabels=x_labels,
                    yticklabels=checkpoints,
                    cmap="viridis",
                    annot=False,
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kws={"label": metric_label},
                    linewidths=0.5,
                )
                ax.set_xlabel("Layer")
                ax.set_ylabel("Checkpoint")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
                title = f"{metric_label} heatmap — {lang} | agg={agg}"
                if model_label:
                    title = f"[{model_label}] " + title
                ax.set_title(title)
                plt.tight_layout()

                fname = f"heatmap_{metric}_{lang}_{agg}.png"
                _save(fig, os.path.join(output_dir, "heatmaps", metric, fname))


# ── GIF export ───────────────────────────────────────────────────────────────

def make_training_curve_gifs(output_dir, metrics, aggregations, layers, duration_s):
    """
    Build one animated GIF per (metric, aggregation) from the training curve PNGs,
    animating through layers in order (layer_0 → layer_N).

    GIF is limited to 256 colours per frame by the format spec. Quality is maximised
    by quantising each frame independently with the median-cut algorithm before saving.
    For true lossless animation export, use a video format instead.
    """
    layer_nums_sorted = sorted(layers, key=layer_num)

    for metric in metrics:
        for agg in aggregations:
            frames = []
            for layer in layer_nums_sorted:
                path = os.path.join(
                    output_dir, "training_curves", metric,
                    f"training_curve_{metric}_{layer}_{agg}.png",
                )
                if os.path.exists(path):
                    frames.append(Image.open(path).copy())

            if len(frames) < 2:
                print(f"  Skipping GIF for {metric}/{agg} — fewer than 2 frames found.")
                continue

            frame_ms = max(1, int(duration_s * 1000 / len(frames)))

            # Quantise each frame to 255 colours (median-cut) for best GIF quality.
            # 255 instead of 256 reserves one slot for the GIF transparency index.
            palette_frames = [
                f.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
                for f in frames
            ]

            gif_path = os.path.join(
                output_dir, "training_curves", metric,
                f"training_curve_{metric}_{agg}.gif",
            )
            palette_frames[0].save(
                gif_path,
                save_all=True,
                append_images=palette_frames[1:],
                duration=frame_ms,
                loop=0,
                optimize=False,
            )
            print(f"  Saved {gif_path}  ({len(frames)} frames × {frame_ms} ms)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate geometry analysis visualizations from a metrics CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv", required=True, help="Path to metrics CSV file.")
    p.add_argument("--output-dir", default="plots",
                   help="Root directory for output PNG files (default: plots/).")
    p.add_argument("--model", default="",
                   help="Model name shown in plot titles (e.g. FuxiTranyu-8B).")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Metrics to plot. Default: all columns in CSV.")
    p.add_argument("--languages", nargs="+", default=None,
                   help="Languages to include. Default: all in CSV.")
    p.add_argument("--layers", nargs="+", default=None,
                   help="Layers to include (e.g. layer_0 layer_12). Default: all.")
    p.add_argument("--aggregations", nargs="+", default=None,
                   help="Aggregations to include (last mean). Default: all.")
    p.add_argument(
        "--plot-types", nargs="+",
        default=["training_curves", "layer_profiles", "heatmaps"],
        choices=["training_curves", "layer_profiles", "heatmaps"],
        help="Which plot types to generate (default: all three).",
    )
    p.add_argument(
        "--shared-y-axis", action="store_true",
        help="Fix the y-axis range to the global min/max per metric across all plots. "
             "Makes it easy to compare plots for different languages or layers directly.",
    )
    p.add_argument(
        "--smoothing", type=float, default=0.0, metavar="SIGMA",
        help="Gaussian smoothing sigma (in data points) for training curves. "
             "Also enables change-point detection and shaded regions (default: 0 = off).",
    )
    p.add_argument(
        "--normalize", choices=["none", "global", "per-segment"], default="none",
        help="Normalization for training curves: none (default), global min-max per curve, "
             "or per-segment min-max around the detected change point.",
    )
    p.add_argument(
        "--training-curves-gif-duration", type=float, default=None, metavar="SECONDS",
        help="If set, produce an animated GIF for each (metric, aggregation) from the "
             "training curve PNGs, animating through layers. Total duration in seconds "
             "(e.g. --training-curves-gif-duration 10). Tip: combine with --shared-y-axis "
             "so all frames share the same scale.",
    )
    p.add_argument(
        "--limit-tokens", type=str, default=None, metavar="TOKENS",
        help="Limit plots to checkpoints up to this token count (e.g. 500B or 1T). "
             "Default: None (use all checkpoints).",
    )
    return p.parse_args()


def _parse_token_limit(limit_str):
    """Parse a token limit string like '500B' or '1T' and return value in billions."""
    if limit_str is None:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([BT]?)$", limit_str.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid token limit format: {limit_str}. Use e.g. '500B' or '1T'.")
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "T":
        val *= 1000
    return val


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Apply token limit filter if specified
    if args.limit_tokens:
        limit_b = _parse_token_limit(args.limit_tokens)
        df["checkpoint_tokens"] = df["checkpoint"].map(_ckpt_key)
        initial_rows = len(df)
        df = df[df["checkpoint_tokens"] <= limit_b]
        filtered_rows = len(df)
        print(f"Token limit: {args.limit_tokens} ({limit_b}B). Filtered from {initial_rows} to {filtered_rows} rows.")
        df = df.drop(columns=["checkpoint_tokens"])

    metric_cols = [c for c in df.columns
                   if c not in {"checkpoint", "dataset", "layer", "aggregation"}]
    metrics      = args.metrics      or metric_cols
    languages    = args.languages    or sorted(df["dataset"].unique())
    layers       = args.layers       or sorted(df["layer"].unique(), key=layer_num)
    aggregations = args.aggregations or sorted(df["aggregation"].unique())

    print(f"Metrics:      {metrics}")
    print(f"Languages:    {languages}")
    print(f"Layers:       {layers}")
    print(f"Aggregations: {aggregations}")
    print(f"Plot types:   {args.plot_types}")
    print(f"Shared y-axis: {args.shared_y_axis}")
    print(f"Smoothing:    {args.smoothing}")
    print(f"Normalize:    {args.normalize}")
    print(f"Output dir:   {args.output_dir}/\n")

    # Compute global y-axis range per (metric, aggregation) pair across all selected data.
    # Keyed as y_ranges[(metric, agg)] = (vmin, vmax).
    # A 5% margin is added so data points don't sit right at the axis edge.
    y_ranges = None
    if args.shared_y_axis:
        plot_df = df[
            df["dataset"].isin(languages) &
            df["layer"].isin(layers) &
            df["aggregation"].isin(aggregations)
        ]
        y_ranges = {}
        for metric in metrics:
            if metric not in plot_df.columns:
                continue
            for agg in aggregations:
                subset = plot_df[plot_df["aggregation"] == agg]
                if subset.empty:
                    continue
                vmin = subset[metric].min()
                vmax = subset[metric].max()
                pad = (vmax - vmin) * 0.05
                y_ranges[(metric, agg)] = (vmin - pad, vmax + pad)
        print(f"Y-axis ranges per (metric, aggregation):")
        for (m, a), (lo, hi) in y_ranges.items():
            print(f"  {m} / {a}: [{round(lo,2)}, {round(hi,2)}]")
        print()

    kw = {
        "metrics": metrics,
        "languages": languages,
        "layers": layers,
        "aggregations": aggregations,
        "output_dir": args.output_dir,
        "model_label": args.model,
        "y_ranges": y_ranges,
    }

    if "training_curves" in args.plot_types:
        print("[1/3] Training curves...")
        plot_training_curves(df, **kw, smoothing=args.smoothing, normalize=args.normalize)

    if "layer_profiles" in args.plot_types:
        print("\n[2/3] Layer profiles...")
        plot_layer_profiles(df, **kw)

    if "heatmaps" in args.plot_types:
        print("\n[3/3] Heatmaps...")
        plot_heatmaps(df, **kw)

    if args.training_curves_gif_duration is not None:
        if "training_curves" not in args.plot_types:
            print("\n[GIF] training_curves were not generated in this run — skipping GIF.")
        else:
            if not args.shared_y_axis:
                print("\n[GIF] Warning: --shared-y-axis not set. Frames will have different "
                      "y-scales, making the animation hard to read.")
            print(f"\n[GIF] Building training curve GIFs ({args.training_curves_gif_duration}s)...")
            make_training_curve_gifs(
                args.output_dir, metrics, aggregations, layers, args.training_curves_gif_duration,
            )

    print(f"\nDone — all plots saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
