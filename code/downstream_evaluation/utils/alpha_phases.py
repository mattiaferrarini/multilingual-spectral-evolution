"""
Training phase identification from AlphaReQ trajectories.

Detects two phases in the AlphaReQ (α) curve for each language:
  - Regularization-increasing: α falls from the first checkpoint to its global
    minimum (the model gains self-regularization). Only present if the trough
    occurs after the first observed checkpoint (e.g. FuxiTranyu-8B).
  - Regularization-decreasing: α rises from the trough to the end of training
    (self-regularization weakens). Present in all models.

Since the regularization-increasing phase is not observable for Apertus-8B-2509
(the trough falls at or before the first checkpoint), all correlation predictors
are computed from the post-minimum (regularization-decreasing) phase only, making
them comparable across both models.

The main entry point is compute_alpha_phases(), which returns a DataFrame with
phase metadata and post-minimum geometry predictors for correlation analysis.
"""

import numpy as np
import pandas as pd


def _identify_alpha_phases_single(alpha_values, token_counts: list) -> dict:
    av = np.array(alpha_values, dtype=float)
    tc = np.array(token_counts, dtype=float)

    if len(av) < 2 or np.all(np.isnan(av)):
        return {"regularization_increasing": None, "regularization_decreasing": None,
                "trough_idx": 0, "trough_tokens": float(tc[0]) if len(tc) else float("nan")}

    trough_idx     = int(np.nanargmin(av))
    reg_increasing = (float(tc[0]), float(tc[trough_idx])) if trough_idx > 0 else None

    last_valid     = av[~np.isnan(av)][-1]
    has_decreasing = (trough_idx < len(av) - 1) and (last_valid >= av[trough_idx])
    reg_decreasing = (float(tc[trough_idx]), float(tc[-1])) if has_decreasing else None

    return {
        "regularization_increasing": reg_increasing,
        "regularization_decreasing": reg_decreasing,
        "trough_idx":    trough_idx,
        "trough_tokens": float(tc[trough_idx]),
    }


def _phase_onset(phases: dict, phase_name: str) -> float:
    p = phases.get(phase_name)
    return float(p[0]) if p is not None else float("nan")


def _phase_duration(phases: dict, phase_name: str) -> float:
    p = phases.get(phase_name)
    return float(p[1] - p[0]) if p is not None else float("nan")


def _nanmean_postmin(av: np.ndarray, tc: np.ndarray,
                     trough_idx: int, start_frac: float, end_frac: float) -> float:
    """
    Mean of av over [t_min + start_frac * post_range, t_min + end_frac * post_range].
    Returns NaN if no valid values in the window.
    """
    t_min      = tc[trough_idx]
    post_range = tc[-1] - t_min
    t_start    = t_min + start_frac * post_range
    t_end      = t_min + end_frac   * post_range
    mask       = (tc >= t_start) & (tc <= t_end)
    vals       = av[mask]
    valid      = vals[~np.isnan(vals)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


def _postmin_rate(av: np.ndarray, tc: np.ndarray, trough_idx: int) -> float:
    """
    Initial AlphaReQ increase rate: slope from the trough to the checkpoint
    closest to the 25% mark of the post-minimum token range.

    rate = (α[first_25] − α[trough]) / (tokens[first_25] − tokens[trough])

    Returns NaN if the post-minimum window is too narrow or values are missing.
    """
    t_min      = tc[trough_idx]
    post_range = tc[-1] - t_min
    if post_range <= 0:
        return float("nan")

    t_target  = t_min + 0.25 * post_range
    post_tc   = tc[trough_idx:]
    post_av   = av[trough_idx:]
    pt_idx    = int(np.nanargmin(np.abs(post_tc - t_target)))

    alpha_pt  = float(post_av[pt_idx])
    tokens_pt = float(post_tc[pt_idx])
    alpha_min = float(av[trough_idx])

    if np.isnan(alpha_pt) or np.isnan(alpha_min) or (tokens_pt - t_min) <= 0:
        return float("nan")
    return (alpha_pt - alpha_min) / (tokens_pt - t_min)


def compute_alpha_phases(df_layer: pd.DataFrame, checkpoints_all: list,
                         token_counts: list) -> pd.DataFrame:
    """
    Identify AlphaReQ phases and compute post-minimum geometry predictors per language.

    Returns df_alpha_phases with columns:

    Phase metadata (for display):
      trough_tokens, reg_increasing_onset_tokens, reg_increasing_duration_tokens,
      reg_increasing_rate, reg_decreasing_onset_tokens, reg_decreasing_duration_tokens

    Post-minimum predictors for correlation analysis:
      alpha_first                — AlphaReQ at the first checkpoint
                                   (safe for grokking onset; equals alpha_trough for Apertus)
      alpha_trough               — AlphaReQ at the global minimum
      alpha_last                 — AlphaReQ at the last checkpoint
      alpha_postmin_rate         — initial increase rate from trough to first 25% of post-min range
      alpha_postmin_early_mean   — mean over first 25% of post-min token range
      alpha_postmin_medium_mean  — mean over first 50% of post-min token range
      alpha_postmin_late_half_mean — mean over last 50% of post-min token range
      alpha_postmin_late_mean    — mean over last 25% of post-min token range
      alpha_postmin_mean         — mean over full post-min token range
    """
    tc = np.array(token_counts, dtype=float)
    records = []
    for lang in sorted(df_layer["dataset"].unique()):
        sub    = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        av     = np.array([sub.loc[c, "alpha_req"] if c in sub.index else np.nan
                           for c in checkpoints_all], dtype=float)
        phases = _identify_alpha_phases_single(av, token_counts)

        trough_idx  = phases["trough_idx"]
        duration_b  = _phase_duration(phases, "regularization_increasing")
        alpha_first = float(av[0])          if not np.isnan(av[0])          else float("nan")
        alpha_trough = float(av[trough_idx]) if not np.isnan(av[trough_idx]) else float("nan")
        alpha_last  = float(av[-1])         if not np.isnan(av[-1])         else float("nan")

        reg_increasing_rate = (
            (alpha_first - alpha_trough) / duration_b
            if duration_b > 0 and not np.isnan(alpha_first) and not np.isnan(alpha_trough)
            else float("nan")
        )

        records.append({
            "language": lang,
            "phases":   phases,
            # Phase metadata
            "trough_tokens":                  phases["trough_tokens"],
            "reg_increasing_onset_tokens":    _phase_onset(phases,    "regularization_increasing"),
            "reg_increasing_duration_tokens": _phase_duration(phases, "regularization_increasing"),
            "reg_increasing_rate":            reg_increasing_rate,
            "reg_decreasing_onset_tokens":    _phase_onset(phases,    "regularization_decreasing"),
            "reg_decreasing_duration_tokens": _phase_duration(phases, "regularization_decreasing"),
            # Post-minimum predictors
            "alpha_first":                    alpha_first,
            "alpha_trough":                   alpha_trough,
            "alpha_last":                     alpha_last,
            "alpha_postmin_rate":             _postmin_rate(av, tc, trough_idx),
            "alpha_postmin_early_mean":       _nanmean_postmin(av, tc, trough_idx, 0.00, 0.25),
            "alpha_postmin_medium_mean":      _nanmean_postmin(av, tc, trough_idx, 0.00, 0.50),
            "alpha_postmin_late_half_mean":   _nanmean_postmin(av, tc, trough_idx, 0.50, 1.00),
            "alpha_postmin_late_mean":        _nanmean_postmin(av, tc, trough_idx, 0.75, 1.00),
            "alpha_postmin_mean":             _nanmean_postmin(av, tc, trough_idx, 0.00, 1.00),
        })
    return pd.DataFrame(records)


_ALPHA_DISPLAY_COLS = {
    "language":                     "language",
    "trough_tokens":                "trough (B)",
    "alpha_first":                  "AlphaReQ first ckpt",
    "alpha_trough":                 "AlphaReQ at minimum",
    "alpha_last":                   "AlphaReQ last ckpt",
    "alpha_postmin_rate":           "initial increase rate (1/B)",
    "alpha_postmin_early_mean":     "mean first 25% post-min",
    "alpha_postmin_medium_mean":    "mean first 50% post-min",
    "alpha_postmin_late_half_mean": "mean last 50% post-min",
    "alpha_postmin_late_mean":      "mean last 25% post-min",
    "alpha_postmin_mean":           "mean full post-min",
}


def compute_and_show_alpha_phases(model_keys: list, data: dict) -> None:
    """
    Compute AlphaReQ phases for every model, store in data[m]["df_alpha_phases"],
    and print the summary table showing all phase metadata and predictors.

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
