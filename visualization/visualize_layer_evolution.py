"""
Q: Are RankMe training curves layer-dependent and language-dependent?
How do the curves evolve through layers?

Produces, for each (model, metric, aggregation):

  (A) Per-language overlays: one figure per language with all-layer training
      curves stacked on a log-token x-axis, coloured by depth (viridis).
      -> Shows directly how the curve shape changes with layer for a
         single language.

  (B) Phase-timing heatmaps over (layer x language):
        - peak_t  : token count at which RankMe reaches its maximum
                    (end of the entropy-seeking phase)
        - trough_t: token count at the minimum AFTER the peak
                    (bottom of compression-seeking)
        - amplitude: peak - trough
        - last_v   : final-checkpoint RankMe
      -> If peak_t shifts with layer at fixed language, curves are
         layer-dependent. If peak_t shifts with language at fixed layer,
         they are language-dependent. The heatmap shows both at once.

  (C) Small-multiple grid of training curves: a few layers x a few
      languages, so you can compare shapes by eye.

Usage:
  python visualization/visualize_layer_evolution.py \
      --csv results/fuxi_fine_wiki.csv --experiment fuxi_fine_wiki \
      --model "FuxiTranyu-8B"
"""

import argparse
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)


def _ckpt_key(name):
    s = str(name)
    if s.lower() == "main":
        return float("inf")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([BT]?)$", s, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    m = re.match(r"step\d+-tokens(\d+)([BT])", s, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2).upper() == "T" else val
    return float("inf") - 1


def _layer_num(layer_name):
    m = re.search(r"(\d+)", str(layer_name))
    return int(m.group(1)) if m else 0


def _prepare(df, metric, aggregation):
    sub = df[df["aggregation"] == aggregation].copy()
    sub["t_b"] = sub["checkpoint"].apply(_ckpt_key)
    sub = sub[sub["t_b"] < float("inf")]  # drop "main"
    return sub


def plot_layer_overlay_per_language(df, metric, aggregation, output_dir, experiment, model_label):
    sub = _prepare(df, metric, aggregation)
    layers = sorted(sub["layer"].unique(), key=_layer_num)
    layer_nums = [_layer_num(l) for l in layers]
    languages = sorted(sub["dataset"].unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(layer_nums), vmax=max(layer_nums))

    base_dir = os.path.join(output_dir, experiment, "layer_evolution",
                            f"per_language_{metric}_{aggregation}")
    os.makedirs(base_dir, exist_ok=True)

    for lang in languages:
        lang_df = sub[sub["dataset"] == lang]
        if lang_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for layer, ln in zip(layers, layer_nums):
            data = lang_df[lang_df["layer"] == layer].sort_values("t_b")
            ax.plot(data["t_b"], data[metric], color=cmap(norm(ln)),
                    linewidth=1.3, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Tokens (B)")
        ax.set_ylabel(metric)
        title = f"{lang} — per-layer {metric} curves | agg={aggregation}"
        if model_label:
            title = f"[{model_label}] " + title
        ax.set_title(title)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Layer")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"layer_overlay_{metric}_{lang}_{aggregation}.png"
        fig.savefig(os.path.join(base_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  per-language overlays -> {base_dir}")


def compute_phase_timing(df, metric, aggregation):
    sub = _prepare(df, metric, aggregation).sort_values(["dataset", "layer", "t_b"])
    rows = []
    for (lang, layer), grp in sub.groupby(["dataset", "layer"]):
        ts = grp["t_b"].values
        vs = grp[metric].values
        if len(vs) < 3:
            continue
        peak_idx = int(np.argmax(vs))
        peak_t, peak_v = ts[peak_idx], vs[peak_idx]
        post = vs[peak_idx:]
        post_t = ts[peak_idx:]
        if len(post) >= 2:
            trough_idx = int(np.argmin(post))
            trough_t, trough_v = post_t[trough_idx], post[trough_idx]
        else:
            trough_t, trough_v = peak_t, peak_v
        rows.append({
            "dataset": lang,
            "layer": layer,
            "layer_num": _layer_num(layer),
            "peak_t": peak_t,
            "peak_v": peak_v,
            "trough_t": trough_t,
            "trough_v": trough_v,
            "first_v": vs[0],
            "last_v": vs[-1],
            "amplitude": peak_v - trough_v,
            "rel_drop": (peak_v - trough_v) / max(peak_v, 1e-9),
        })
    return pd.DataFrame(rows)


def plot_phase_timing_heatmaps(shapes, metric, aggregation, output_dir, experiment, model_label):
    languages = sorted(shapes["dataset"].unique())
    layers = sorted(shapes["layer"].unique(), key=_layer_num)
    layer_idx = {l: i for i, l in enumerate(layers)}
    lang_idx = {l: i for i, l in enumerate(languages)}

    base_dir = os.path.join(output_dir, experiment, "layer_evolution")
    os.makedirs(base_dir, exist_ok=True)

    for field, label, cmap_name, fmt in [
        ("peak_t",    "Token at RankMe peak (B)",                "viridis", ".0f"),
        ("trough_t",  "Token at post-peak trough (B)",           "plasma",  ".0f"),
        ("amplitude", "Peak - trough amplitude (RankMe units)",  "magma",   ".0f"),
        ("rel_drop",  "Relative drop (peak - trough)/peak",      "magma",   ".2f"),
        ("last_v",    "Final-checkpoint RankMe",                 "viridis", ".0f"),
    ]:
        mat = np.full((len(layers), len(languages)), np.nan)
        for _, r in shapes.iterrows():
            mat[layer_idx[r["layer"]], lang_idx[r["dataset"]]] = r[field]
        fig, ax = plt.subplots(figsize=(1.4 + 0.7 * len(languages), 0.28 * len(layers) + 2))
        sns.heatmap(
            mat, xticklabels=languages,
            yticklabels=[_layer_num(l) for l in layers],
            cmap=cmap_name, ax=ax,
            cbar_kws={"label": label},
            annot=True, fmt=fmt, annot_kws={"fontsize": 6},
        )
        ax.set_xlabel("Language")
        ax.set_ylabel("Layer")
        title = f"{label} | agg={aggregation}"
        if model_label:
            title = f"[{model_label}] " + title
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(base_dir, f"phase_{field}_{metric}_{aggregation}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


def plot_grid(df, metric, aggregation, output_dir, experiment, model_label,
              languages_subset, layers_subset):
    sub = _prepare(df, metric, aggregation)
    languages_subset = [l for l in languages_subset if l in set(sub["dataset"])]
    layers_subset = [l for l in layers_subset if l in set(sub["layer"])]
    fig, axes = plt.subplots(len(layers_subset), len(languages_subset),
                              figsize=(2.3 * len(languages_subset), 1.7 * len(layers_subset)),
                              sharex=True, squeeze=False)
    for i, layer in enumerate(layers_subset):
        for j, lang in enumerate(languages_subset):
            ax = axes[i, j]
            data = sub[(sub["dataset"] == lang) & (sub["layer"] == layer)].sort_values("t_b")
            ax.plot(data["t_b"], data[metric], color="steelblue", linewidth=1.4)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(lang, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"L{_layer_num(layer)}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"{metric} training curves — layers x languages | agg={aggregation}"
                 + (f"  [{model_label}]" if model_label else ""), fontsize=11)
    fig.supxlabel("Tokens (B)", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fname = f"grid_{metric}_{aggregation}.png"
    out_path = os.path.join(output_dir, experiment, "layer_evolution", fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved grid: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--metric", default="rankme")
    p.add_argument("--aggregation", default="last")
    p.add_argument("--output-dir", default="plots")
    p.add_argument("--model", default="")
    p.add_argument("--grid-languages", nargs="+",
                   default=["English", "German", "Chinese", "Japanese", "Swahili", "Hindi"])
    p.add_argument("--grid-layers", nargs="+", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")

    plot_layer_overlay_per_language(df, args.metric, args.aggregation,
                                     args.output_dir, args.experiment, args.model)

    shapes = compute_phase_timing(df, args.metric, args.aggregation)
    plot_phase_timing_heatmaps(shapes, args.metric, args.aggregation,
                                args.output_dir, args.experiment, args.model)

    if args.grid_layers is None:
        all_layers = sorted(df["layer"].unique(), key=_layer_num)
        n = len(all_layers)
        idxs = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
        layers_subset = [all_layers[i] for i in idxs]
    else:
        layers_subset = [f"layer_{i}" for i in args.grid_layers]
    plot_grid(df, args.metric, args.aggregation, args.output_dir, args.experiment,
              args.model, args.grid_languages, layers_subset)

    print("\nMedian shape characteristics per layer (across languages):")
    summary = (shapes.groupby("layer_num")[["peak_t", "trough_t", "amplitude", "rel_drop"]]
               .median().round(2))
    print(summary.to_string())


if __name__ == "__main__":
    main()
