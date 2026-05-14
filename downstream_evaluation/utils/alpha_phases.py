"""
Training phase identification from AlphaReQ trajectories.

Detects two phases in the AlphaReQ (α) curve for each language:
  - Regularization-increasing: α falls from the first checkpoint to its global
    minimum (the model gains self-regularization).
  - Regularization-decreasing: α rises from the trough to the end of training
    (self-regularization weakens slightly as the model adapts further).

The main entry point is compute_alpha_phases(), which returns a DataFrame with
onset and duration (in billions of tokens) for each phase and language.
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

    last_valid = av[~np.isnan(av)][-1]
    has_decreasing = (trough_idx < len(av) - 1) and (last_valid >= av[trough_idx])
    reg_decreasing = (float(tc[trough_idx]), float(tc[-1])) if has_decreasing else None

    return {
        "regularization_increasing": reg_increasing,
        "regularization_decreasing": reg_decreasing,
        "trough_idx":   trough_idx,
        "trough_tokens": float(tc[trough_idx]),
    }


def _phase_onset(phases: dict, phase_name: str) -> float:
    p = phases.get(phase_name)
    return float(p[0]) if p is not None else float("nan")


def _phase_duration(phases: dict, phase_name: str) -> float:
    p = phases.get(phase_name)
    return float(p[1] - p[0]) if p is not None else float("nan")


def compute_alpha_phases(df_layer: pd.DataFrame, checkpoints_all: list,
                         token_counts: list) -> pd.DataFrame:
    """
    Identify regularization-increasing and regularization-decreasing phases per language.

    Returns df_alpha_phases with trough position, onset, and duration columns.
    """
    records = []
    for lang in sorted(df_layer["dataset"].unique()):
        sub    = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        av     = [sub.loc[c, "alpha_req"] if c in sub.index else np.nan
                  for c in checkpoints_all]
        phases = _identify_alpha_phases_single(av, token_counts)
        records.append({
            "language": lang,
            "phases":   phases,
            "trough_tokens":                  phases["trough_tokens"],
            "reg_increasing_onset_tokens":    _phase_onset(phases,    "regularization_increasing"),
            "reg_increasing_duration_tokens": _phase_duration(phases, "regularization_increasing"),
            "reg_decreasing_onset_tokens":    _phase_onset(phases,    "regularization_decreasing"),
            "reg_decreasing_duration_tokens": _phase_duration(phases, "regularization_decreasing"),
        })
    return pd.DataFrame(records)
