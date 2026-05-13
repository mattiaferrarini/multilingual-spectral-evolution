import numpy as np
import pandas as pd


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


def compute_phases(df_layer: pd.DataFrame, checkpoints_all: list,
                   token_counts: list) -> pd.DataFrame:
    """
    Identify entropy-seeking and compression-seeking phases for every language.

    Returns df_phases with onset/duration columns per language.
    """
    records = []
    for lang in sorted(df_layer["dataset"].unique()):
        sub    = df_layer[df_layer["dataset"] == lang].set_index("checkpoint")
        rv     = [sub.loc[c, "rankme"] if c in sub.index else np.nan for c in checkpoints_all]
        phases = _identify_phases_single(rv, token_counts)
        records.append({
            "language": lang,
            "phases":   phases,
            "peak_tokens":                 phases["peak_tokens"],
            "compression_onset_tokens":    _phase_onset(phases,   "compression_seeking"),
            "compression_duration_tokens": _phase_duration(phases, "compression_seeking"),
            "entropy_onset_tokens":        _phase_onset(phases,   "entropy_seeking"),
            "entropy_duration_tokens":     _phase_duration(phases, "entropy_seeking"),
        })
    return pd.DataFrame(records)
