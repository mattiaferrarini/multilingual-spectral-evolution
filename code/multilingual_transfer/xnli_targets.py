"""
XNLI-specific target loading.

Public API:
  NORMALIZATIONS                        -- list of target column names
  find_xnli_files(output_dir, short_model)
  load_xnli_targets(xnli_files, common_sorted, k_values, lang_codes, t)
"""

import glob
import os

import numpy as np
import pandas as pd

NORMALIZATIONS = ["row_norm", "col_norm"]


def find_xnli_files(output_dir, short_model):
    """Return {checkpoint_label: filepath} for all summary CSVs of this model."""
    pattern = os.path.join(output_dir, f"{short_model}_*_summary.csv")
    files = glob.glob(pattern)
    result = {}
    for f in files:
        basename = os.path.basename(f)
        label = basename[len(f"{short_model}_"):-len("_summary.csv")]
        result[label] = f
    return result


def load_xnli_targets(xnli_files, common_sorted, k_values, lang_codes, t):
    """
    Load XNLI transfer outcomes for each (perf_checkpoint, k, src_lang, tgt_lang) pair.

    For each index i in common_sorted, perf_checkpoint = common_sorted[i + t].
    Returns a DataFrame with columns:
      perf_checkpoint, k, src_lang, tgt_lang, mean_accuracy, row_norm, col_norm
    """
    rows = []
    for i in range(len(common_sorted) - t):
        perf_ckpt = common_sorted[i + t]

        df = pd.read_csv(xnli_files[perf_ckpt])
        df = df[df["k"].isin(k_values)]
        df = df[df["src_lang"].isin(lang_codes) & df["tgt_lang"].isin(lang_codes)]

        diag = (
            df[df["src_lang"] == df["tgt_lang"]]
            .set_index(["k", "src_lang"])["mean_accuracy"]
        )

        for _, row in df[df["src_lang"] != df["tgt_lang"]].iterrows():
            k = row["k"]
            src, tgt = row["src_lang"], row["tgt_lang"]
            acc = row["mean_accuracy"]

            acc_ss = diag.get((k, src))
            acc_tt = diag.get((k, tgt))
            row_norm = acc / acc_ss if (acc_ss is not None and acc_ss > 0) else np.nan
            col_norm = acc / acc_tt if (acc_tt is not None and acc_tt > 0) else np.nan

            rows.append({
                "perf_checkpoint": perf_ckpt,
                "k": k,
                "src_lang": src,
                "tgt_lang": tgt,
                "mean_accuracy": acc,
                "row_norm": row_norm,
                "col_norm": col_norm,
            })

    return pd.DataFrame(rows)
