"""
RankMe geometry metrics from training trajectories.

For each language, identifies the peak token count and computes magnitude-based
metrics: initial richness, valley of first descent, final richness,
and the initial drop rate.

The main entry point is compute_geometry(), which returns a DataFrame with one
row per language.
"""

import numpy as np
import pandas as pd


def _find_valley(rv: np.ndarray, tc: np.ndarray, peak_idx: int) -> tuple:
    """
    Find the bottom of the initial compression valley.

    Two-phase strategy:
    1. If the trajectory descends from the first checkpoint to a genuine minimum
       before the global peak (any pre-peak value is strictly less than rv[0]),
       use the pre-peak global minimum. This handles models like Apertus whose
       first checkpoint is already past an unobserved entropy peak, producing a
       visible U-shaped compression dip before a later global peak.
    2. Otherwise (first checkpoint is at or below all pre-peak values — e.g. the
       first checkpoint is itself a pre-entropy low), use the global minimum in
       the post-peak region. This handles Fuxi (peak at first checkpoint) and
       Vietnamese (low first checkpoint before a brief entropy rise).

    Returns (valley_idx, valley_tokens, valley_rankme).
    """
    # Case 1: genuine pre-peak descent
    if peak_idx > 0:
        pre_peak = rv[:peak_idx]
        valid_pre = ~np.isnan(pre_peak)
        if np.any(valid_pre) and float(np.nanmin(pre_peak)) < float(rv[0]):
            rel_idx = int(np.nanargmin(pre_peak))
            return rel_idx, float(tc[rel_idx]), float(rv[rel_idx])

    # Case 2: first checkpoint is at or below all pre-peak values — search post-peak
    post_rv = rv[peak_idx:]
    valid_post = ~np.isnan(post_rv)
    if not np.any(valid_post):
        return peak_idx, float(tc[peak_idx]), float(rv[peak_idx])
    rel_idx = int(np.nanargmin(post_rv))
    abs_idx = peak_idx + rel_idx
    return abs_idx, float(tc[abs_idx]), float(rv[abs_idx])


def _identify_phases_single(rankme_values, token_counts: list) -> dict:
    rv = np.array(rankme_values, dtype=float)
    tc = np.array(token_counts,  dtype=float)

    if len(rv) < 2 or np.all(np.isnan(rv)):
        return {"entropy_seeking": None, "compression_seeking": None,
                "peak_idx": 0, "peak_tokens": float(tc[0]) if len(tc) else float("nan")}

    peak_idx        = int(np.nanargmax(rv))
    entropy_seeking = (float(tc[0]), float(tc[peak_idx])) if peak_idx > 0 else None

    last_valid      = rv[~np.isnan(rv)][-1]
    has_compression = (peak_idx < len(rv) - 1) and (last_valid < rv[peak_idx])
    compression_seeking = (float(tc[peak_idx]), float(tc[-1])) if has_compression else None

    return {
        "entropy_seeking":     entropy_seeking,
        "compression_seeking": compression_seeking,
        "peak_idx":    peak_idx,
        "peak_tokens": float(tc[peak_idx]),
    }


def _phase_onset(phases: dict, phase_name: str) -> float:
    p = phases.get(phase_name)
    return float(p[0]) if p is not None else float("nan")


def _phase_duration(phases: dict, phase_name: str) -> float:
    p = phases.get(phase_name)
    return float(p[1] - p[0]) if p is not None else float("nan")


def _nanmean_slice(rv: np.ndarray, mask: np.ndarray) -> float:
    """Mean of rv[mask], ignoring NaN. Returns NaN if no valid values."""
    vals = rv[mask]
    valid = vals[~np.isnan(vals)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


_GEOMETRY_DISPLAY_COLS = {
    "language":                 "language",
    "peak_tokens":              "peak (B)",
    "rankme_peak":              "RankMe peak",
    "rankme_first":             "RankMe first ckpt",
    "rankme_valley":            "RankMe valley",
    "valley_tokens":            "valley (B)",
    "rankme_initial_drop_rate": "initial drop rate (1/B)",
    "rankme_last":              "RankMe last ckpt",
    "rankme_early_mean":        "mean early (0–25%)",
    "rankme_medium_mean":       "mean medium (0–50%)",
    "rankme_late_half_mean":    "mean late half (50–100%)",
    "rankme_late_mean":         "mean late (75–100%)",
    "rankme_before_valley":     "mean before valley",
    "rankme_after_valley":      "mean after valley",
}


def compute_and_show_geometry(model_keys: list, data: dict) -> None:
    """
    Compute geometry metrics for every model, store in data[m]["df_geometry"],
    and print the summary table.

    After this call, data[m] gains the key: df_geometry.
    """
    for m in model_keys:
        d   = data[m]
        cfg = d["cfg"]
        df_geometry     = compute_geometry(d["df_layer"], d["checkpoints_all"], d["token_counts"])
        d["df_geometry"] = df_geometry
        print(f"\n── {cfg['model_label']} ────────────────────────────────────────")
        print(df_geometry[list(_GEOMETRY_DISPLAY_COLS)]
              .rename(columns=_GEOMETRY_DISPLAY_COLS)
              .to_string(index=False))


def compute_geometry(df_layer: pd.DataFrame, checkpoints_all: list,
                     token_counts: list) -> pd.DataFrame:
    """
    Compute RankMe geometry metrics for every language.

    Returns df_geometry with one row per language and columns:
      rankme_first             — RankMe at the first checkpoint (initial richness)
      rankme_last              — RankMe at the last checkpoint (final richness)
      rankme_valley            — RankMe at the bottom of the first descent
      valley_tokens            — token count (B) at the valley of the first descent
      rankme_initial_drop_rate — (rankme_first - rankme_valley) / (valley_tokens - first_tokens)
      rankme_early_mean        — mean RankMe over checkpoints in the first 25% of token range
      rankme_medium_mean       — mean RankMe over checkpoints in the first 50% of token range
      rankme_late_half_mean    — mean RankMe over checkpoints in the last 50% of token range
      rankme_late_mean         — mean RankMe over checkpoints in the last 25% of token range
      rankme_before_valley     — mean RankMe from the first checkpoint up to and including the valley
      rankme_after_valley      — mean RankMe from the checkpoint after the valley to the last
    """
    tc = np.array(token_counts, dtype=float)
    t_max = tc[-1]

    records = []
    for lang in sorted(df_layer["dataset"].unique()):
        sub    = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        rv     = np.array([sub.loc[c, "rankme"] if c in sub.index else np.nan
                           for c in checkpoints_all], dtype=float)
        phases = _identify_phases_single(rv, token_counts)

        rankme_first = float(rv[0])               if not np.isnan(rv[0])  else float("nan")
        rankme_last  = float(rv[-1])              if not np.isnan(rv[-1]) else float("nan")
        rankme_peak  = float(np.nanmax(rv))       if np.any(~np.isnan(rv)) else float("nan")

        valley_idx, valley_tokens, rankme_valley = _find_valley(rv, tc, phases["peak_idx"])
        duration_to_valley = valley_tokens - tc[0]
        if (not np.isnan(rankme_first) and not np.isnan(rankme_valley)
                and duration_to_valley > 0):
            rankme_initial_drop_rate = (rankme_first - rankme_valley) / duration_to_valley
        else:
            rankme_initial_drop_rate = float("nan")

        rankme_early_mean     = _nanmean_slice(rv, tc <= 0.25 * t_max)
        rankme_medium_mean    = _nanmean_slice(rv, tc <= 0.50 * t_max)
        rankme_late_half_mean = _nanmean_slice(rv, tc >= 0.50 * t_max)
        rankme_late_mean      = _nanmean_slice(rv, tc >= 0.75 * t_max)
        rankme_before_valley  = _nanmean_slice(rv, np.arange(len(tc)) <= valley_idx)
        rankme_after_valley   = _nanmean_slice(rv, np.arange(len(tc)) >  valley_idx)

        records.append({
            "language":                 lang,
            "peak_tokens":              phases["peak_tokens"],
            "rankme_peak":              rankme_peak,
            "rankme_first":             rankme_first,
            "rankme_last":              rankme_last,
            "rankme_valley":            rankme_valley,
            "valley_tokens":            valley_tokens,
            "rankme_initial_drop_rate": rankme_initial_drop_rate,
            "rankme_early_mean":        rankme_early_mean,
            "rankme_medium_mean":       rankme_medium_mean,
            "rankme_late_half_mean":    rankme_late_half_mean,
            "rankme_late_mean":         rankme_late_mean,
            "rankme_before_valley":     rankme_before_valley,
            "rankme_after_valley":      rankme_after_valley,
        })
    return pd.DataFrame(records)
