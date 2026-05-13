"""
Data loading utilities for the downstream evaluation pipeline.

Loads the RankMe geometry CSV (fuxi.csv / apertus.csv) and the merged
evaluation CSV produced by merge_results.py, returning clean DataFrames
ready for analysis.
"""

from pathlib import Path

import pandas as pd

from .checkpoints import ckpt_to_tokens, sort_checkpoints


def load_rankme_data(rankme_csv: Path, layer: str, aggregation: str) -> tuple:
    """
    Load RankMe CSV and return a filtered slice for the given layer/aggregation.

    Returns: (df_rankme, df_layer, checkpoints_all, token_counts, langs_sorted)
    """
    df_rankme = pd.read_csv(rankme_csv)

    # Drop checkpoints whose names cannot be parsed (e.g. "main" branch on HF Hub).
    known = [c for c in df_rankme["checkpoint"].unique()
             if ckpt_to_tokens(c) != float("inf")]
    dropped = set(df_rankme["checkpoint"].unique()) - set(known)
    if dropped:
        df_rankme = df_rankme[df_rankme["checkpoint"].isin(known)]

    checkpoints_all = sort_checkpoints(known)
    token_counts    = [ckpt_to_tokens(c) for c in checkpoints_all]

    df_layer     = df_rankme[(df_rankme["layer"] == layer) &
                              (df_rankme["aggregation"] == aggregation)].copy()
    langs_sorted = sorted(df_layer["dataset"].unique())

    print(f"Loaded RankMe: {len(df_rankme):,} rows — "
          f"{len(checkpoints_all)} checkpoints, {len(langs_sorted)} languages")
    print(f"Working slice : {layer}, agg={aggregation} → {len(df_layer)} rows")
    return df_rankme, df_layer, checkpoints_all, token_counts, langs_sorted


def load_eval_data(merged_csv: Path) -> tuple:
    """
    Load the merged evaluation CSV produced by merge_results.py.

    Returns: (df_eval, eval_available)
    """
    if not Path(merged_csv).exists():
        print(f"[INFO] {merged_csv} not found — run merge_results.py first (see README).")
        return pd.DataFrame(columns=["checkpoint", "language", "task", "accuracy"]), False

    df_eval = pd.read_csv(merged_csv)[["checkpoint", "language", "task", "accuracy"]].copy()
    print(f"Loaded eval: {len(df_eval)} records — "
          f"{df_eval['task'].nunique()} task(s), "
          f"{df_eval['checkpoint'].nunique()} checkpoint(s)")
    return df_eval, True
