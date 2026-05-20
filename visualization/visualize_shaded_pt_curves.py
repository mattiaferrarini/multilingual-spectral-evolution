import argparse
import math
import os
import re
from collections import Counter

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
}

def _compute_ticks(xs_numeric, tick_step=None):
    """Return (tick_positions, tick_labels) spanning the data range.

    Always includes the first and last value; intermediate ticks are spaced
    by tick_step (auto-chosen to give ~6 ticks when None).
    xs_numeric: sorted list of finite x values (in billions of tokens).
    """
    xs = sorted(xs_numeric)
    if not xs:
        return [0], ["0"]
    first, last = xs[0], xs[-1]
    if first == last:
        return [first], [str(int(first))]

    if tick_step is None:
        raw = (last - first) / 5
        mag = 10 ** math.floor(math.log10(max(raw, 1)))
        tick_step = max(round(raw / mag) * mag, 1)

    start = math.ceil(first / tick_step) * tick_step
    intermediates, t = [], start
    while t < last:
        if t > first:
            intermediates.append(float(t))
        t += tick_step

    positions = sorted(set([first] + intermediates + [last]))
    labels = [str(int(p)) if p == int(p) else str(p) for p in positions]
    return positions, labels


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


def _sort_checkpoints(checkpoints):
    return sorted(checkpoints, key=_ckpt_key)


def _layer_num(layer_name):
    m = re.search(r"(\d+)", str(layer_name))
    return int(m.group(1)) if m else 0


def _language_colors(languages):
    n = max(len(languages), 1)
    if n <= 10:
        palette = sns.color_palette("tab10", n_colors=n)
    elif n <= 20:
        palette = sns.color_palette("tab20", n_colors=n)
    else:
        palette = sns.color_palette("husl", n_colors=n)
    return {lang: palette[i] for i, lang in enumerate(languages)}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _minmax(ys):
    arr = np.array(ys, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi == lo:
        return arr.tolist()
    return ((arr - lo) / (hi - lo)).tolist()


def _smooth_on_uniform_grid(xs, ys, sigma):
    """Smooth in x-space by interpolating to a uniform grid before applying the Gaussian."""
    from scipy.ndimage import gaussian_filter1d
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    nan_mask = np.isnan(ys_arr)
    if nan_mask.all() or sigma <= 0:
        return ys
    valid_xs = xs_arr[~nan_mask]
    valid_ys = ys_arr[~nan_mask]
    # uniform grid spanning the data range with the same number of points
    n = len(xs_arr)
    x_uniform = np.linspace(xs_arr[0], xs_arr[-1], n)
    y_uniform = np.interp(x_uniform, valid_xs, valid_ys)
    y_smoothed_uniform = gaussian_filter1d(y_uniform, sigma=sigma)
    # map back to original x positions
    y_smoothed = np.interp(xs_arr, x_uniform, y_smoothed_uniform)
    y_smoothed[nan_mask] = np.nan
    return y_smoothed.tolist()


def _collect_curve_data(subset, languages, checkpoints, xs_common, metric, smoothing):
    """Return raw y-values per language, smoothed matrix, and leftmost data x."""
    raw_ys_per_lang = {}
    smoothed_matrix = []
    min_data_x = float("inf")
    for lang in languages:
        row = subset[subset["dataset"] == lang].set_index("checkpoint")
        ys = [row.loc[c, metric] if c in row.index else np.nan for c in checkpoints]
        raw_ys_per_lang[lang] = ys
        valid_xs = [x for x, y in zip(xs_common, ys) if not np.isnan(y)]
        if valid_xs:
            min_data_x = min(min_data_x, valid_xs[0])
        if smoothing > 0:
            smoothed_matrix.append(_smooth_on_uniform_grid(xs_common, ys, smoothing))
    return raw_ys_per_lang, smoothed_matrix, min_data_x


def _find_consensus_breakpoints(smoothed_matrix, xs_common, pen=None):
    """Find trend reversals as prominent local extrema of the mean smoothed curve.

    Uses scipy peak prominence to distinguish real reversals from noise-induced
    wiggles.  *pen* is the minimum required prominence in the same units as the
    metric (default: 15 % of the curve's value range).  Raise it to get fewer
    breakpoints; lower it to get more.
    """
    from scipy.signal import find_peaks

    if not smoothed_matrix:
        return []

    xs_arr = np.array(xs_common)
    mean_curve = np.nanmean(np.array(smoothed_matrix, dtype=float), axis=0)
    valid_mask = ~np.isnan(mean_curve)
    if valid_mask.sum() < 5:
        return []

    valid_xs = xs_arr[valid_mask]
    valid_s = mean_curve[valid_mask]
    n = len(valid_s)

    prominence = pen if pen is not None else np.ptp(valid_s) * 0.15
    min_dist = max(2, n // 10)

    peaks_max, _ = find_peaks(valid_s, prominence=prominence, distance=min_dist)
    peaks_min, _ = find_peaks(-valid_s, prominence=prominence, distance=min_dist)
    extrema_idx = np.sort(np.concatenate([peaks_max, peaks_min]))

    return [min(xs_common, key=lambda x: abs(x - valid_xs[i])) for i in extrema_idx]


def _normalize_ys(ys, xs, normalize, consensus_bps):
    if normalize == "global":
        return _minmax(ys)
    if normalize == "per-segment" and consensus_bps:
        arr = np.array(ys, dtype=float)
        bp_indices = [xs.index(bp) for bp in consensus_bps if bp in xs]
        boundaries = [0] + bp_indices + [len(xs)]
        result = np.full(len(arr), np.nan, dtype=float)
        for j in range(len(boundaries) - 1):
            lo, hi = boundaries[j], boundaries[j + 1]
            result[lo:hi] = _minmax(arr[lo:hi].tolist())
        return result.tolist()
    return ys


def _plot_lines(ax, languages, raw_ys_per_lang, xs_common, lang_colors, normalize, consensus_bps):
    """Draw one line per language; return list of (x, y, lang, color) endpoints."""
    endpoints = []
    for lang in languages:
        ys = raw_ys_per_lang[lang]
        plot_ys = _normalize_ys(ys, xs_common, normalize, consensus_bps)
        ax.plot(xs_common, plot_ys, color=lang_colors[lang], linewidth=1.5)
        valid = [(x, y) for x, y in zip(xs_common, plot_ys) if not np.isnan(y)]
        if valid:
            endpoints.append((valid[-1][0], valid[-1][1], lang, lang_colors[lang]))
    return endpoints


def _segment_is_increasing(seg, use_derivative=False):
    """Return True if segment is trending upward.

    use_derivative=True: use the mean of first differences (more robust for
    noisy or non-monotone segments).  use_derivative=False: compare endpoints.
    """
    if len(seg) < 2:
        return True
    if use_derivative:
        return float(np.nanmean(np.diff(seg.astype(float)))) >= 0
    return seg[-1] >= seg[0]


def _shade_segments(ax, consensus_bps, smoothed_matrix, xs_common, left_x, tick_positions, use_derivative=False):
    """Shade trend segments blue (increasing) / red (decreasing) and add a legend."""
    if not consensus_bps:
        return
    mean_smoothed = np.nanmean(np.array(smoothed_matrix, dtype=float), axis=0)
    boundaries_x = [left_x] + list(consensus_bps) + [tick_positions[-1]]
    xs_arr = np.array(xs_common)
    for x_lo, x_hi in zip(boundaries_x[:-1], boundaries_x[1:]):
        idx_lo = int(np.argmin(np.abs(xs_arr - x_lo)))
        idx_hi = int(np.argmin(np.abs(xs_arr - x_hi)))
        seg = mean_smoothed[idx_lo:idx_hi + 1]
        seg = seg[~np.isnan(seg)]
        increasing = _segment_is_increasing(seg, use_derivative=use_derivative)
        ax.axvspan(x_lo, x_hi, color="steelblue" if increasing else "tomato", alpha=0.15, zorder=0)
    for bp in consensus_bps:
        ax.axvline(bp, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    from matplotlib.lines import Line2D
    cp_labels = ", ".join(
        str(int(bp)) if bp == int(bp) else str(bp) for bp in consensus_bps
    )
    ax.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.3, label="Entropy seeking"),
        Patch(facecolor="tomato", alpha=0.3, label="Compression seeking"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Changepoints: {cp_labels} B"),
    ], fontsize=11, loc="upper right")


def _annotate_endpoints(ax, endpoints):
    """Place language labels at curve endpoints, spreading them to avoid overlap."""
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
                    arrowprops=dict(arrowstyle="-", color=color, lw=1, alpha=0.5, relpos=(0, 0.5)))


def _setup_axes(ax, tick_positions, tick_labels, left_x, metric_label, layer, agg, model_label):
    ax.set_xlim(left=left_x, right=tick_positions[-1])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=12)
    ax.set_xlabel("Billion of tokens", fontsize=13)
    ax.set_ylabel(metric_label, fontsize=13)
    layer_n = _layer_num(layer)
    title = f"{metric_label} over pretraining - Layer {layer_n} | {agg.capitalize()} aggregation"
    if model_label:
        title = f"[{model_label}] " + title
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)


def plot_training_curves(df, metrics, languages, layers, aggregations, output_dir, model_label, smoothing=0.0, normalize="none", changepoint_pen=None, tick_step=None, main_value=None):
    all_checkpoints = _sort_checkpoints(df["checkpoint"].unique())
    lang_colors = _language_colors(languages)

    numeric_ckpts = [c for c in all_checkpoints if str(c).lower() != "main"]
    xs_numeric = sorted(_ckpt_key(c) for c in numeric_ckpts)

    if main_value is not None:
        main_x = float(main_value)
        checkpoints = all_checkpoints
        xs_for_ticks = sorted(xs_numeric + [main_x])
    else:
        checkpoints = numeric_ckpts
        main_x = None
        xs_for_ticks = xs_numeric

    tick_positions, tick_labels = _compute_ticks(xs_for_ticks, tick_step)

    def ckpt_x(c):
        return main_x if str(c).lower() == "main" else _ckpt_key(c)

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        print(f"  {metric_label}...")
        for layer in layers:
            for agg in aggregations:
                subset = df[(df["layer"] == layer) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                xs_common = [ckpt_x(c) for c in checkpoints]
                raw_ys_per_lang, smoothed_matrix, min_data_x = _collect_curve_data(
                    subset, languages, checkpoints, xs_common, metric, smoothing)
                left_x = min_data_x if min_data_x < float("inf") else tick_positions[0]
                consensus_bps = _find_consensus_breakpoints(smoothed_matrix, xs_common, pen=changepoint_pen)

                fig, ax = plt.subplots(figsize=(12, 6))
                endpoints = _plot_lines(ax, languages, raw_ys_per_lang, xs_common,
                                        lang_colors, normalize, consensus_bps)
                _shade_segments(ax, consensus_bps, smoothed_matrix, xs_common, left_x, tick_positions)
                _annotate_endpoints(ax, endpoints)
                _setup_axes(ax, tick_positions, tick_labels, left_x, metric_label, layer, agg, model_label)
                plt.tight_layout()

                fname = f"training_curve_{metric}_{layer}_{agg}.png"
                _save(fig, os.path.join(output_dir, metric, fname))


def _resolve_changepoints(changepoints, xs_common):
    """Snap manually supplied changepoint x-values to the nearest value in xs_common."""
    xs_arr = np.array(xs_common)
    snapped = []
    for cp in changepoints:
        nearest = xs_common[int(np.argmin(np.abs(xs_arr - cp)))]
        snapped.append(nearest)
    return sorted(set(snapped))


def show_training_curves(
    df,
    metrics=None,
    languages=None,
    layers=None,
    aggregations=None,
    model_label="",
    smoothing=0.0,
    normalize="none",
    changepoint_pen=None,
    changepoints=None,
    tick_step=None,
    main_value=None,
):
    """Display training-curve plots inline (designed for Jupyter notebooks).

    All list parameters default to every value present in *df*, so you can
    narrow down by passing e.g. layers=["layer_5"], metrics=["rankme"].
    Multiple values produce multiple plots, one per (metric, layer, aggregation).
    tick_step: spacing between intermediate x-ticks in billions of tokens (auto if None).
    main_value: x-position (in billions of tokens) for the "main" checkpoint. If None,
        "main" is excluded from the plot.
    changepoints: explicit list of x-positions (in billions of tokens) to use as
        changepoints instead of auto-detecting them.  When provided, each segment's
        direction is determined by the mean derivative within that segment rather
        than endpoint comparison.
    """
    all_checkpoints = _sort_checkpoints(df["checkpoint"].unique())
    numeric_ckpts = [c for c in all_checkpoints if str(c).lower() != "main"]
    xs_numeric = sorted(_ckpt_key(c) for c in numeric_ckpts)

    if main_value is not None:
        main_x = float(main_value)
        checkpoints = all_checkpoints
        xs_for_ticks = sorted(xs_numeric + [main_x])
    else:
        checkpoints = numeric_ckpts
        main_x = None
        xs_for_ticks = xs_numeric

    tick_positions, tick_labels = _compute_ticks(xs_for_ticks, tick_step)

    def ckpt_x(c):
        return main_x if str(c).lower() == "main" else _ckpt_key(c)

    metric_cols = [c for c in df.columns if c not in {"checkpoint", "dataset", "layer", "aggregation"}]
    _metrics = metrics or metric_cols
    _languages = languages or sorted(df["dataset"].unique())
    _layers = layers or sorted(df["layer"].unique(), key=_layer_num)
    _aggregations = aggregations or sorted(df["aggregation"].unique())
    lang_colors = _language_colors(_languages)

    manual_changepoints = changepoints is not None

    for metric in _metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        for layer in _layers:
            for agg in _aggregations:
                subset = df[(df["layer"] == layer) & (df["aggregation"] == agg)]
                if subset.empty:
                    continue

                xs_common = [ckpt_x(c) for c in checkpoints]
                raw_ys_per_lang, smoothed_matrix, min_data_x = _collect_curve_data(
                    subset, _languages, checkpoints, xs_common, metric, smoothing)
                left_x = min_data_x if min_data_x < float("inf") else tick_positions[0]

                if manual_changepoints:
                    consensus_bps = _resolve_changepoints(changepoints, xs_common)
                else:
                    consensus_bps = _find_consensus_breakpoints(smoothed_matrix, xs_common, pen=changepoint_pen)

                _, ax = plt.subplots(figsize=(12, 6))
                endpoints = _plot_lines(ax, _languages, raw_ys_per_lang, xs_common,
                                        lang_colors, normalize, consensus_bps)
                _shade_segments(ax, consensus_bps, smoothed_matrix, xs_common, left_x, tick_positions,
                                use_derivative=manual_changepoints)
                _annotate_endpoints(ax, endpoints)
                _setup_axes(ax, tick_positions, tick_labels, left_x, metric_label, layer, agg, model_label)
                plt.tight_layout()
                plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot training curves from a metrics CSV.")
    p.add_argument("--csv", required=True, help="Path to metrics CSV file.")
    p.add_argument("--output-dir", default="plots",
                   help="Root directory for output PNG files (default: plots/).")
    p.add_argument("--model", default="", help="Model name shown in plot titles.")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Metrics to plot. Default: all columns in CSV.")
    p.add_argument("--languages", nargs="+", default=None,
                   help="Languages to include. Default: all in CSV.")
    p.add_argument("--layers", nargs="+", default=None,
                   help="Layers to include. Default: all.")
    p.add_argument("--aggregations", nargs="+", default=None,
                   help="Aggregations to include. Default: all.")
    p.add_argument("--smoothing", type=float, default=0.0, metavar="SIGMA",
                   help="Gaussian smoothing sigma in number of data points (default: 0 = no smoothing).")
    p.add_argument("--normalize", choices=["none", "global", "per-segment"], default="none",
                   help="Normalization mode: none (default), global min-max per curve, or per-segment min-max around the change point.")
    p.add_argument("--changepoint-pen", type=float, default=None, metavar="PEN",
                   help="PELT penalty for changepoint detection (default: log(n)). Higher = fewer breakpoints.")
    p.add_argument("--tick-step", type=float, default=None, metavar="STEP",
                   help="Spacing between intermediate x-ticks in billions of tokens (default: auto ~6 ticks).")
    p.add_argument("--main-value", type=float, default=None, metavar="B",
                   help="X-position (billions of tokens) for the 'main' checkpoint. Omit to hide it.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    metric_cols = [c for c in df.columns
                   if c not in {"checkpoint", "dataset", "layer", "aggregation"}]
    metrics      = args.metrics      or metric_cols
    languages    = args.languages    or sorted(df["dataset"].unique())
    layers       = args.layers       or sorted(df["layer"].unique(), key=_layer_num)
    aggregations = args.aggregations or sorted(df["aggregation"].unique())

    print(f"Metrics:      {metrics}")
    print(f"Languages:    {languages}")
    print(f"Layers:       {layers}")
    print(f"Aggregations: {aggregations}")
    print(f"Output dir:   {args.output_dir}/\n")

    plot_training_curves(
        df, metrics=metrics, languages=languages, layers=layers,
        aggregations=aggregations, output_dir=args.output_dir,
        model_label=args.model, smoothing=args.smoothing, normalize=args.normalize,
        changepoint_pen=args.changepoint_pen, tick_step=args.tick_step,
        main_value=args.main_value,
    )

    print(f"\nDone — plots saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
