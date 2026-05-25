"""
Correlation analysis between compression phase geometry and downstream accuracy.

Tests whether the onset or duration of the compression-seeking phase predicts
downstream performance (grokking onset, peak accuracy) using Spearman and
Pearson correlations. Computed separately for m-MMLU and XCOPA — never pooled.

The main entry point is compute_correlations_table(), which returns a tidy
DataFrame with correlation coefficients and p-values for each predictor-outcome
pair and task.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _correlate(x, y, min_pairs: int = 4) -> dict | None:
    x, y  = np.array(x, dtype=float), np.array(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < min_pairs:
        return None
    xv, yv     = x[valid], y[valid]
    sp_r, sp_p = stats.spearmanr(xv, yv)
    pe_r, pe_p = stats.pearsonr(xv, yv)
    return {"n_languages": int(valid.sum()),
            "spearman_r": round(float(sp_r), 4), "spearman_p": round(float(sp_p), 4),
            "pearson_r":  round(float(pe_r), 4), "pearson_p":  round(float(pe_p), 4)}


def compute_alpha_correlations_table(df_grokking: pd.DataFrame,
                                      df_alpha_phases: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman + Pearson correlations: AlphaReQ phase geometry → downstream accuracy.

    Predictors: regularization-decreasing onset (trough position) and
    rate of α decline (drop in α per billion tokens during the regularization-increasing phase).
    Returns df_alpha_correlations.
    """
    records = []
    for task in ["m_mmlu", "xcopa"]:
        df_t = df_grokking[df_grokking["task"] == task]
        if df_t.empty:
            continue
        merged = df_t.merge(
            df_alpha_phases[["language", "reg_decreasing_onset_tokens",
                              "reg_increasing_rate"]],
            on="language", how="inner")
        for x_col, x_label in [
            ("reg_decreasing_onset_tokens", "Reg.-decreasing onset (B)"),
            ("reg_increasing_rate",         "Rate of α decline (α/B)"),
        ]:
            for y_col, y_label in [
                ("grokking_tokens", "Grokking onset (B)"),
                ("peak_accuracy",   "Peak accuracy"),
            ]:
                corr = _correlate(merged[x_col], merged[y_col])
                row  = {"task": task, "predictor (x)": x_label, "outcome (y)": y_label}
                row.update(corr if corr else {"n_languages": 0, "spearman_r": None,
                                               "spearman_p": None, "pearson_r": None,
                                               "pearson_p": None})
                records.append(row)
    return pd.DataFrame(records)


def compute_correlations_table(df_grokking: pd.DataFrame,
                                df_geometry: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman + Pearson correlations: RankMe phase geometry → downstream accuracy.

    Predictors:
      - Compression onset / duration (timing-based — may have zero variance if all
        languages peak at the same checkpoint, as in FuxiTranyu-8B)
      - RankMe at first checkpoint (initial representation richness per language)
      - RankMe at last checkpoint  (final representation richness per language)
      - Valley of first descent     (token count at bottom of initial drop)
      - Initial drop rate           (compression speed during first descent, Δ RankMe / B tokens)

    Returns df_correlations.
    """
    phase_cols = ["language", "valley_tokens",
                  "rankme_first", "rankme_last", "rankme_initial_drop_rate",
                  "rankme_early_mean", "rankme_medium_mean", "rankme_late_mean",
                  "rankme_before_valley", "rankme_after_valley"]
    records = []
    for task in ["m_mmlu", "xcopa"]:
        df_t = df_grokking[df_grokking["task"] == task]
        if df_t.empty:
            continue
        merged = df_t.merge(df_geometry[phase_cols], on="language", how="inner")
        for x_col, x_label in [
            ("rankme_first",             "RankMe at first ckpt"),
            ("valley_tokens",            "Valley of first descent (B)"),
            ("rankme_initial_drop_rate", "Initial RankMe drop rate (1/B)"),
            ("rankme_last",              "RankMe at last ckpt"),
            ("rankme_early_mean",        "Mean RankMe — first 25% of tokens"),
            ("rankme_medium_mean",       "Mean RankMe — first 50% of tokens"),
            ("rankme_late_mean",         "Mean RankMe — last 25% of tokens"),
            ("rankme_before_valley",     "Mean RankMe — before valley"),
            ("rankme_after_valley",      "Mean RankMe — after valley"),
        ]:
            for y_col, y_label in [("grokking_tokens", "Grokking onset (B)"),
                                    ("peak_accuracy",   "Peak accuracy")]:
                corr = _correlate(merged[x_col], merged[y_col])
                row  = {"task": task, "predictor (x)": x_label, "outcome (y)": y_label}
                row.update(corr if corr else {"n_languages": 0, "spearman_r": None, "spearman_p": None,
                                               "pearson_r": None, "pearson_p": None})
                records.append(row)
    return pd.DataFrame(records)


def compute_checkpoint_correlations(df_layer: pd.DataFrame, df_eval: pd.DataFrame,
                                    checkpoints_all: list, token_counts: list,
                                    task_languages: dict) -> pd.DataFrame:
    """
    For each (checkpoint, task) pair, compute the cross-language Spearman correlation
    between RankMe and downstream accuracy at that checkpoint.

    Returns a DataFrame with columns:
      checkpoint, token_count, task, spearman_r, spearman_p, n_languages
    """
    layer_by_ckpt = {
        ckpt: grp.set_index("dataset")["rankme"]
        for ckpt, grp in df_layer.groupby("checkpoint")
    }
    eval_by_ckpt_task = {
        (ckpt, task): grp.set_index("language")["accuracy"]
        for (ckpt, task), grp in df_eval.groupby(["checkpoint", "task"])
    }

    records = []
    for ckpt, tc in zip(checkpoints_all, token_counts):
        rankme_map = layer_by_ckpt.get(ckpt, pd.Series(dtype=float))
        for task, langs in task_languages.items():
            acc_map = eval_by_ckpt_task.get((ckpt, task), pd.Series(dtype=float))
            overlap = [l for l in langs
                       if l in rankme_map.index and l in acc_map.index]
            rv    = np.array([float(rankme_map[l]) for l in overlap])
            av    = np.array([float(acc_map[l])    for l in overlap])
            valid = ~(np.isnan(rv) | np.isnan(av))
            n     = int(valid.sum())
            if n < 3:
                records.append({"checkpoint": ckpt, "token_count": tc, "task": task,
                                 "spearman_r": np.nan, "spearman_p": np.nan,
                                 "n_languages": n})
                continue
            sp_r, sp_p = stats.spearmanr(rv[valid], av[valid])
            records.append({"checkpoint": ckpt, "token_count": tc, "task": task,
                             "spearman_r": round(float(sp_r), 4),
                             "spearman_p": round(float(sp_p), 4),
                             "n_languages": n})
    return pd.DataFrame(records)
