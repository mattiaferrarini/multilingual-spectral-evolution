"""
Generate per-target score heatmaps for ECLeKTic cross-lingual transfer evaluation.

Reads {output_dir}/{short_model}_judgments_per_pair.csv and writes one PNG per target
to {plotting.output_dir}/:

  accuracy_majority.png
  overall_score.png
  transfer_score.png
  accuracy_majority_row_norm.png  (÷ acc(src, src))
  accuracy_majority_col_norm.png  (÷ acc(tgt, tgt))

Usage:
    python eclektic_heatmap.py --config configs/eclektic_apertus.yaml
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from eclektic_targets import TARGETS


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ECLeKTic model config YAML")
    return parser.parse_args()


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _reindex(matrix, lang_order):
    return matrix.reindex(index=lang_order, columns=lang_order)


def _normalize_rows(matrix):
    """Divide each row by acc(src, src): how well src transfers out."""
    diag = pd.Series(np.diag(matrix.values), index=matrix.index)
    return matrix.div(diag, axis=0)


def _normalize_cols(matrix):
    """Divide each col by acc(tgt, tgt): how well tgt is served by foreign context."""
    diag = pd.Series(np.diag(matrix.values), index=matrix.columns)
    return matrix.div(diag, axis=1)


def _plot_heatmap(matrix, lang_order, lang_names, title, cbar_label, output_path):
    data = matrix.values.astype(float)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    n = len(lang_order)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    full_names = [lang_names[l] for l in lang_order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(full_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(full_names, rotation=0, fontsize=10)
    fig.suptitle(title, fontsize=13, y=0.98)
    ax.set_xlabel("Target language", fontsize=11)
    ax.set_ylabel("Source language", fontsize=11)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _find_repo_root(start_dir):
    d = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return start_dir
        d = parent


def _load_eclektic_matrix(cfg, metric, normalization):
    """Return (matrix, lang_order, lang_names, short_model)."""
    config_dir = None
    if isinstance(cfg, str):
        config_dir = os.path.dirname(os.path.abspath(cfg))
        cfg = _load_config(cfg)

    short_model = cfg["model"]["name"].split("/")[-1]
    output_dir = cfg["output_dir"]
    if config_dir and not os.path.isabs(output_dir):
        output_dir = os.path.join(_find_repo_root(config_dir), output_dir)
    lang_order = list(cfg["languages"].keys())
    lang_names = cfg["languages"]

    csv_path = os.path.join(output_dir, f"{short_model}_judgments_per_pair.csv")
    df = pd.read_csv(csv_path)
    df = df[df["original_lang"].isin(lang_order) & df["lang"].isin(lang_order)]
    df = df.rename(columns={"original_lang": "src_lang", "lang": "tgt_lang"})

    pivot = _reindex(df.pivot(index="src_lang", columns="tgt_lang", values=metric), lang_order)

    if normalization == "row":
        matrix = _normalize_rows(pivot)
    elif normalization == "col":
        matrix = _normalize_cols(pivot)
    else:
        matrix = pivot

    return matrix, lang_order, lang_names, short_model


def plot_models_comparison(configs, metric="accuracy_majority", normalization="raw", shared_scale=True, output_path=None):
    """
    Plot one heatmap per model side by side in a single figure.

    Parameters
    ----------
    configs : list[dict | str]
        Config dicts or paths to YAML config files, one per model.
    metric : str
        Column to plot (e.g. "accuracy_majority", "overall_score", "transfer_score").
    normalization : {"raw", "row", "col"}
        "raw" — metric value as-is; "row" — divided by diag(src,src); "col" — divided by diag(tgt,tgt).
    shared_scale : bool
        Whether all subplots share the same vmin/vmax colorbar range.
    output_path : str | None
        If given, save the figure to this path.
    """
    n = len(configs)
    records = [_load_eclektic_matrix(cfg, metric, normalization) for cfg in configs]

    if shared_scale:
        all_vals = np.concatenate([r[0].values.ravel() for r in records])
        global_vmin, global_vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    _metric_label = metric.replace("_", " ").title()
    _cbar_labels = {
        "raw": _metric_label,
        "row": f"{_metric_label} (÷ diag(src, src))",
        "col": f"{_metric_label} (÷ diag(tgt, tgt))",
    }
    cbar_label = _cbar_labels.get(normalization, "Value")

    fig, axes = plt.subplots(1, n, figsize=(12 * n, 10))
    if n == 1:
        axes = [axes]

    for ax, (matrix, lang_order, lang_names, short_model) in zip(axes, records):
        vmin = global_vmin if shared_scale else float(np.nanmin(matrix.values))
        vmax = global_vmax if shared_scale else float(np.nanmax(matrix.values))

        annot = np.empty(matrix.shape, dtype=object)
        for i, row in enumerate(lang_order):
            for j, col in enumerate(lang_order):
                val = matrix.loc[row, col]
                annot[i, j] = f"{val:.2f}" if pd.notna(val) else ""

        sns.heatmap(
            matrix,
            annot=annot,
            fmt="",
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": cbar_label},
        )
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

        full_names = [lang_names[l] for l in lang_order]
        ax.set_xticklabels(full_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(full_names, rotation=0, fontsize=10)
        ax.set_title(short_model, fontsize=11)
        ax.set_xlabel("Target language", fontsize=11)
        ax.set_ylabel("Source language", fontsize=11)

    norm_label = {"raw": "raw", "row": "row-normalized", "col": "col-normalized"}.get(normalization, normalization)
    fig.suptitle(f"{metric}  |  {norm_label}", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        dirpath = os.path.dirname(output_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fmt = "svg" if output_path.lower().endswith(".svg") else None
        fig.savefig(output_path, dpi=150, bbox_inches="tight", format=fmt)

    plt.show()


def main():
    args = _parse_args()
    cfg = _load_config(args.config)

    short_model = cfg["model"]["name"].split("/")[-1]
    output_dir = cfg["output_dir"]
    plot_dir = cfg["plotting"]["output_dir"]
    lang_order = list(cfg["languages"].keys())
    lang_names = cfg["languages"]

    csv_path = os.path.join(output_dir, f"{short_model}_judgments_per_pair.csv")
    df = pd.read_csv(csv_path)
    df = df[df["original_lang"].isin(lang_order) & df["lang"].isin(lang_order)]
    df = df.rename(columns={"original_lang": "src_lang", "lang": "tgt_lang"})

    for target in TARGETS:
        matrix = _reindex(
            df.pivot(index="src_lang", columns="tgt_lang", values=target),
            lang_order,
        )
        label = target.replace("_", " ").title()
        out = os.path.join(plot_dir, f"{target}.png")
        _plot_heatmap(
            matrix, lang_order, lang_names,
            title=f"{short_model}  |  {target}",
            cbar_label=label,
            output_path=out,
        )
        print(f"{target} → {out}")

    # Row- and col-normalized accuracy (diagonal exists for accuracy_majority)
    acc = _reindex(
        df.pivot(index="src_lang", columns="tgt_lang", values="accuracy_majority"),
        lang_order,
    )

    row_norm = _normalize_rows(acc)
    out = os.path.join(plot_dir, "accuracy_majority_row_norm.png")
    _plot_heatmap(
        row_norm, lang_order, lang_names,
        title=f"{short_model}  |  accuracy_majority  |  ÷ acc(src, src)",
        cbar_label="Transfer ratio (row-normalized)",
        output_path=out,
    )
    print(f"accuracy_majority_row_norm → {out}")

    col_norm = _normalize_cols(acc)
    out = os.path.join(plot_dir, "accuracy_majority_col_norm.png")
    _plot_heatmap(
        col_norm, lang_order, lang_names,
        title=f"{short_model}  |  accuracy_majority  |  ÷ acc(tgt, tgt)",
        cbar_label="Transfer ratio (col-normalized)",
        output_path=out,
    )
    print(f"accuracy_majority_col_norm → {out}")

    print("Done.")


if __name__ == "__main__":
    main()
