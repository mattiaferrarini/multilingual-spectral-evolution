"""
Data loading utilities for the downstream evaluation pipeline.

Loads the RankMe geometry CSV (fuxi_fine_wiki.csv / apertus_fine_wiki.csv) and the merged
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

    print(f"Loaded geometry data: {len(df_rankme):,} rows — "
          f"{len(checkpoints_all)} checkpoints, {len(langs_sorted)} languages")
    print(f"Working slice : {layer}, agg={aggregation} → {len(df_layer)} rows")
    return df_rankme, df_layer, checkpoints_all, token_counts, langs_sorted


def load_model_data(model_keys: list, data: dict) -> None:
    """
    Load RankMe geometry and evaluation data for every model and update data in-place.

    After this call, data[m] gains the keys:
      df_rankme, df_layer, checkpoints_all, token_counts, langs_sorted,
      df_eval, eval_available
    """
    for m in model_keys:
        cfg = data[m]["cfg"]
        df_rankme, df_layer, checkpoints_all, token_counts, langs_sorted = load_rankme_data(
            cfg["rankme_csv"], cfg["layer"], cfg["aggregation"])
        df_eval, eval_available = load_eval_data(cfg["merged_csv"])
        data[m].update({
            "df_rankme":       df_rankme,
            "df_layer":        df_layer,
            "checkpoints_all": checkpoints_all,
            "token_counts":    token_counts,
            "langs_sorted":    langs_sorted,
            "df_eval":         df_eval,
            "eval_available":  eval_available,
        })
        print()


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
