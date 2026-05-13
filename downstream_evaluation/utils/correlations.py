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
    return {"n":          int(valid.sum()),
            "spearman_r": round(float(sp_r), 4), "spearman_p": round(float(sp_p), 4),
            "pearson_r":  round(float(pe_r), 4), "pearson_p":  round(float(pe_p), 4)}


def compute_correlations_table(df_grokking: pd.DataFrame,
                                df_phases: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman + Pearson correlations: compression phase → downstream accuracy.

    Returns df_correlations.
    """
    records = []
    for task in ["m_mmlu", "xcopa"]:
        df_t = df_grokking[df_grokking["task"] == task]
        if df_t.empty:
            continue
        merged = df_t.merge(
            df_phases[["language", "compression_onset_tokens", "compression_duration_tokens"]],
            on="language", how="inner")
        for x_col, x_label in [("compression_onset_tokens",    "Compression onset (B)"),
                                 ("compression_duration_tokens", "Compression duration (B)")]:
            for y_col, y_label in [("grokking_tokens", "Grokking onset (B)"),
                                    ("peak_accuracy",   "Peak accuracy")]:
                corr = _correlate(merged[x_col], merged[y_col])
                row  = {"task": task, "predictor (x)": x_label, "outcome (y)": y_label}
                row.update(corr if corr else {"n": 0, "spearman_r": None, "spearman_p": None,
                                               "pearson_r": None, "pearson_p": None})
                records.append(row)
    return pd.DataFrame(records)
