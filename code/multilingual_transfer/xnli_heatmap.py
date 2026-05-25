"""
Generate per-checkpoint accuracy heatmaps for XNLI cross-lingual transfer evaluation.

Produces three plot types per (checkpoint, k):
  - raw/           : mean accuracy with ±SE annotations
  - row_normalized/: acc(src, tgt) / acc(src, src)  — how well src transfers out
  - col_normalized/: acc(src, tgt) / acc(tgt, tgt)  — how well tgt is served by foreign context

Usage:
    python xnli_heatmap.py --config configs/xnli_apertus.yaml
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

_APERTUS_BRANCH_RE = re.compile(r'^step(\d+)-tokens(\d+)([BT])$')


def _checkpoint_sort_key(name):
    m = _APERTUS_BRANCH_RE.match(name)
    if m:
        return float(m.group(2)) * (1000 if m.group(3) == "T" else 1)
    m = re.match(r'^(\d+(?:\.\d+)?)', name)
    return float(m.group(1)) if m else float("inf")


def ckpt_label(ckpt):
    return ckpt if ckpt is not None else "default"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to XNLI config YAML")
    return parser.parse_args()


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _find_summary_files(output_dir, short_model):
    pattern = os.path.join(output_dir, f"{short_model}_*_summary.csv")
    return glob.glob(pattern)


def _parse_checkpoint(filename, short_model):
    basename = os.path.basename(filename)
    prefix = f"{short_model}_"
    suffix = "_summary.csv"
    return basename[len(prefix):-len(suffix)]


def _reindex(matrix, lang_order):
    return matrix.reindex(index=lang_order, columns=lang_order)


def _normalize_rows(accuracy_matrix):
    """Divide each row by acc(src, src): measures how well src transfers out."""
    diag = pd.Series(np.diag(accuracy_matrix.values), index=accuracy_matrix.index)
    return accuracy_matrix.div(diag, axis=0)


def _normalize_cols(accuracy_matrix):
    """Divide each column by acc(tgt, tgt): measures how well tgt is served by foreign context."""
    diag = pd.Series(np.diag(accuracy_matrix.values), index=accuracy_matrix.columns)
    return accuracy_matrix.div(diag, axis=1)


def _build_raw_annot(accuracy_matrix, lang_order):
    annot = np.empty(accuracy_matrix.shape, dtype=object)
    for i, row in enumerate(lang_order):
        for j, col in enumerate(lang_order):
            val = accuracy_matrix.loc[row, col]
            annot[i, j] = f"{val:.2f}" if pd.notna(val) else ""
    return annot


def _mean_se(std_matrix, n_matrix):
    se = std_matrix.values / np.sqrt(n_matrix.values)
    return np.nanmean(se)


def _build_ratio_annot(matrix, lang_order):
    annot = np.empty(matrix.shape, dtype=object)
    for i, row in enumerate(lang_order):
        for j, col in enumerate(lang_order):
            val = matrix.loc[row, col]
            annot[i, j] = f"{val:.2f}" if pd.notna(val) else ""
    return annot


def _plot_heatmap(matrix, annot, lang_order, lang_names, title, cbar_label, output_path, subtitle=None):
    vmin = matrix.values.min()
    vmax = matrix.values.max()

    fig, ax = plt.subplots(figsize=(12, 10))
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
    fig.suptitle(title, fontsize=13, y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, style="italic", color="gray", pad=6)
    ax.set_xlabel("Test language", fontsize=11)
    ax.set_ylabel("Context language", fontsize=11)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = _parse_args()
    cfg = _load_config(args.config)

    short_model = cfg["model"]["name"].split("/")[-1]
    output_dir = cfg["output_dir"]
    plot_dir = cfg["plotting"]["output_dir"]
    lang_order = list(cfg["languages"].keys())
    lang_names = cfg["languages"]

    summary_files = _find_summary_files(output_dir, short_model)
    if not summary_files:
        raise FileNotFoundError(f"No summary CSVs found for {short_model} in {output_dir}")

    checkpoints = sorted(
        [_parse_checkpoint(f, short_model) for f in summary_files],
        key=_checkpoint_sort_key,
    )
    file_map = {_parse_checkpoint(f, short_model): f for f in summary_files}

    k_values = cfg["icl"]["k"]
    if isinstance(k_values, int):
        k_values = [k_values]

    print(f"Generating heatmaps for {short_model}: {len(checkpoints)} checkpoints × {len(k_values)} k values")

    for ckpt in checkpoints:
        df = pd.read_csv(file_map[ckpt])
        label = ckpt_label(ckpt)
        for k in k_values:
            subset = df[df["k"] == k]
            acc = _reindex(subset.pivot(index="src_lang", columns="tgt_lang", values="mean_accuracy"), lang_order)
            std = _reindex(subset.pivot(index="src_lang", columns="tgt_lang", values="std_accuracy"), lang_order)
            n   = _reindex(subset.pivot(index="src_lang", columns="tgt_lang", values="n"), lang_order)

            stem = f"{short_model}_{label}_k{k}.png"
            base_title = f"{short_model}  |  {label}  |  k={k}"

            # Raw accuracy
            se = _mean_se(std, n)
            _plot_heatmap(
                acc, _build_raw_annot(acc, lang_order),
                lang_order, lang_names,
                title=base_title,
                cbar_label="Accuracy",
                output_path=os.path.join(plot_dir, "raw", stem),
                subtitle=f"mean SE = {se:.3f}  (n = {int(n.values.mean())} per cell)",
            )

            # Row-normalized: acc(src, tgt) / acc(src, src)
            row_norm = _normalize_rows(acc)
            _plot_heatmap(
                row_norm, _build_ratio_annot(row_norm, lang_order),
                lang_order, lang_names,
                title=f"{base_title}  |  ÷ acc(src, src)",
                cbar_label="Transfer ratio (row-normalized)",
                output_path=os.path.join(plot_dir, "row_normalized", stem),
            )

            # Col-normalized: acc(src, tgt) / acc(tgt, tgt)
            col_norm = _normalize_cols(acc)
            _plot_heatmap(
                col_norm, _build_ratio_annot(col_norm, lang_order),
                lang_order, lang_names,
                title=f"{base_title}  |  ÷ acc(tgt, tgt)",
                cbar_label="Transfer ratio (col-normalized)",
                output_path=os.path.join(plot_dir, "col_normalized", stem),
            )

            print(f"  [{label}] k={k} → raw / row_normalized / col_normalized")

    print("Done.")


if __name__ == "__main__":
    main()
