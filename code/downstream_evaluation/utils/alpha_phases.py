"""
AlphaReQ geometry metrics from training trajectories.

For each language, computes non-overlapping quartile means and point-in-time
values of AlphaReQ (α) across training, parallel to the RankMe analysis.

The main entry point is compute_alpha_phases(), which returns a DataFrame with
one row per language.
"""

import numpy as np
import pandas as pd


def _nanmean_slice(av: np.ndarray, mask: np.ndarray) -> float:
    vals = av[mask]
    valid = vals[~np.isnan(vals)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


def compute_alpha_phases(df_layer: pd.DataFrame, checkpoints_all: list,
                         token_counts: list) -> pd.DataFrame:
    """
    Compute AlphaReQ geometry predictors per language.

    Returns df_alpha_phases with columns:
      alpha_first        — AlphaReQ at the first checkpoint
      alpha_ckpt10       — AlphaReQ at the 10th checkpoint
      alpha_first10_mean — mean AlphaReQ over the first 10 checkpoints
      alpha_q2_mean      — mean AlphaReQ in the second quartile (25–50%)
      alpha_q3_mean      — mean AlphaReQ in the third quartile (50–75%)
      alpha_late_mean    — mean AlphaReQ in the fourth quartile (75–100%)
      alpha_last         — AlphaReQ at the last checkpoint
    """
    tc = np.array(token_counts, dtype=float)
    t_max = tc[-1]
    records = []
    for lang in sorted(df_layer["dataset"].unique()):
        sub = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        av  = np.array([sub.loc[c, "alpha_req"] if c in sub.index else np.nan
                        for c in checkpoints_all], dtype=float)

        alpha_first = float(av[0])  if not np.isnan(av[0])  else float("nan")
        alpha_last  = float(av[-1]) if not np.isnan(av[-1]) else float("nan")
        alpha_ckpt10 = float(av[9]) if len(av) > 9 and not np.isnan(av[9]) else float("nan")
        valid10 = av[:10][~np.isnan(av[:10])]
        alpha_first10_mean = float(np.mean(valid10)) if len(valid10) > 0 else float("nan")

        alpha_q2_mean   = _nanmean_slice(av, (tc >  0.25 * t_max) & (tc <= 0.50 * t_max))
        alpha_q3_mean   = _nanmean_slice(av, (tc >  0.50 * t_max) & (tc <= 0.75 * t_max))
        alpha_late_mean = _nanmean_slice(av, tc >= 0.75 * t_max)

        records.append({
            "language":            lang,
            "alpha_first":         alpha_first,
            "alpha_ckpt10":        alpha_ckpt10,
            "alpha_first10_mean":  alpha_first10_mean,
            "alpha_q2_mean":       alpha_q2_mean,
            "alpha_q3_mean":       alpha_q3_mean,
            "alpha_late_mean":     alpha_late_mean,
            "alpha_last":          alpha_last,
        })
    return pd.DataFrame(records)


_ALPHA_DISPLAY_COLS = {
    "language":            "language",
    "alpha_first":         "AlphaReQ first checkpoint",
    "alpha_ckpt10":        "AlphaReQ checkpoint 10",
    "alpha_first10_mean":  "mean first 10 checkpoints",
    "alpha_q2_mean":       "mean Q2 (25–50%)",
    "alpha_q3_mean":       "mean Q3 (50–75%)",
    "alpha_late_mean":     "mean Q4 (75–100%)",
    "alpha_last":          "AlphaReQ last checkpoint",
}


def compute_and_show_alpha_phases(model_keys: list, data: dict) -> None:
    """
    Compute AlphaReQ geometry for every model, store in data[m]["df_alpha_phases"],
    and print the summary table.

    After this call, data[m] gains the key: df_alpha_phases.
    """
    for m in model_keys:
        d   = data[m]
        cfg = d["cfg"]
        df_alpha_phases     = compute_alpha_phases(d["df_layer"], d["checkpoints_all"],
                                                   d["token_counts"])
        d["df_alpha_phases"] = df_alpha_phases
        print(f"\n── {cfg['model_label']} ────────────────────────────────────────")
        print(df_alpha_phases[list(_ALPHA_DISPLAY_COLS)]
              .rename(columns=_ALPHA_DISPLAY_COLS)
              .to_string(index=False))
