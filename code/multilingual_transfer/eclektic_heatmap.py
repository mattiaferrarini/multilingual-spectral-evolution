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
