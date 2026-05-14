import argparse
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
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

_CURVE_TICKS = [10, 100, 200, 300, 400, 500, 600]
_MAIN_X = 600


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


def plot_training_curves(df, metrics, languages, layers, aggregations, output_dir, model_label, smoothing=0.0, normalize="none"):
    checkpoints = _sort_checkpoints(df["checkpoint"].unique())
    lang_colors = _language_colors(languages)

    has_main = any(str(c).lower() == "main" for c in checkpoints)
    numeric_ckpts = [c for c in checkpoints if str(c).lower() != "main"]
    max_numeric = max((_ckpt_key(c) for c in numeric_ckpts), default=0)

    def ckpt_x(c):
        return _MAIN_X if str(c).lower() == "main" else _ckpt_key(c)

    max_x = max(max_numeric, _MAIN_X if has_main else 0)
    tick_positions = [t for t in _CURVE_TICKS if t <= max_x + 1]
    tick_labels = [str(t) for t in tick_positions]

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
                xs_common = [ckpt_x(c) for c in checkpoints]
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

                # compute change point before plotting so per-segment normalization can use it
                per_lang_cps = []
                per_lang_is_min = []
                if smoothing > 0 and smoothed_matrix:
                    xs_arr = np.array(xs_common)
                    for smoothed_ys in smoothed_matrix:
                        s = np.array(smoothed_ys, dtype=float)
                        valid_idx = [i for i, v in enumerate(s) if not np.isnan(v)]
                        if len(valid_idx) >= 2:
                            min_i = min(valid_idx, key=lambda i: s[i])
                            max_i = max(valid_idx, key=lambda i: s[i])

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
                            is_min = ext_i == min_i

                            lefts = [j for j in valid_idx if j < ext_i]
                            rights = [j for j in valid_idx if j > ext_i]
                            if lefts and rights:
                                prev_i, next_i = lefts[-1], rights[0]
                                d_left = (s[ext_i] - s[prev_i]) / (xs_arr[ext_i] - xs_arr[prev_i])
                                d_right = (s[next_i] - s[ext_i]) / (xs_arr[next_i] - xs_arr[ext_i])
                                denom = abs(d_left) + abs(d_right)
                                frac = abs(d_left) / denom if denom > 0 else 0.5
                                x_cross = xs_arr[ext_i] + frac * (xs_arr[next_i] - xs_arr[ext_i])
                                nearest_cp = min(xs_common, key=lambda x: abs(x - x_cross))
                            else:
                                nearest_cp = xs_arr[ext_i]
                            per_lang_cps.append(nearest_cp)
                            per_lang_is_min.append(is_min)

                from collections import Counter
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

                if mode_cp is not None:
                    ax.axvspan(left_x, mode_cp, color="steelblue", alpha=0.15, zorder=0)
                    ax.axvspan(mode_cp, tick_positions[-1], color="tomato", alpha=0.15, zorder=0)
                    ax.axvline(mode_cp, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
                               label=f"Change point (mode): {mode_cp:.0f}B")
                    ax.legend(fontsize=11, loc="upper right")

                # Spread labels vertically to avoid overlap
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
                                arrowprops=dict(arrowstyle="-", color=color,
                                                lw=1, alpha=0.5, relpos=(0, 0.5)))

                ax.set_xlim(left=left_x, right=tick_positions[-1])
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, fontsize=12)
                ax.set_xlabel("Billion of tokens", fontsize=13)
                ax.set_ylabel(metric_label, fontsize=13)
                title = f"{metric_label} over training — {layer} | agg={agg}"
                if model_label:
                    title = f"[{model_label}] " + title
                ax.set_title(title, fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                fname = f"training_curve_{metric}_{layer}_{agg}.png"
                _save(fig, os.path.join(output_dir, metric, fname))


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
    )

    print(f"\nDone — plots saved under {args.output_dir}/")


if __name__ == "__main__":
    main()