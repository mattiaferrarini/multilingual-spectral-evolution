"""
Grokking detection for downstream evaluation results.

Grokking is defined as the first training checkpoint where accuracy exceeds
random chance by a threshold (default 15 pp) for at least a minimum number
of consecutive checkpoints (default 2). Both parameters are configurable.

The main entry point is compute_grokking(), which returns a DataFrame with the
grokking onset token count and peak accuracy per (task, language) pair.
"""

import numpy as np
import pandas as pd

from .checkpoints import ckpt_to_tokens


def _identify_grokking_single(accuracy_values, token_counts, random_chance,
                               threshold, min_consecutive) -> float:
    target = random_chance + threshold
    acc, tc = np.array(accuracy_values, dtype=float), np.array(token_counts, dtype=float)
    streak, streak_start = 0, float("nan")
    for a, t in zip(acc, tc):
        if not np.isnan(a) and a > target:
            if streak == 0:
                streak_start = t
            streak += 1
            if streak >= min_consecutive:
                return float(streak_start)
        else:
            streak, streak_start = 0, float("nan")
    return float("nan")


def _peak_acc_info(accuracy_values, token_counts) -> tuple:
    acc, tc = np.array(accuracy_values, dtype=float), np.array(token_counts, dtype=float)
    if np.all(np.isnan(acc)):
        return float("nan"), float("nan")
    idx = int(np.nanargmax(acc))
    return float(tc[idx]), float(acc[idx])


_GROKKING_DISPLAY_COLS = {
    "task":            "task",
    "language":        "language",
    "grokking_tokens": "grokking onset (B)",
    "peak_tokens":     "tokens at peak accuracy (B)",
    "peak_accuracy":   "peak accuracy",
    "random_chance":   "random chance baseline",
}


def compute_and_show_grokking(model_keys: list, data: dict) -> None:
    """
    Compute grokking for every model, store in data[m]["df_grokking"], and print the table.

    After this call, data[m] gains the key: df_grokking.
    """
    for m in model_keys:
        d   = data[m]
        cfg = d["cfg"]
        if d["eval_available"]:
            df_grokking     = compute_grokking(d["df_eval"], cfg["task_languages"],
                                               cfg["random_chance"],
                                               threshold=cfg["grokking_threshold"],
                                               min_consecutive=cfg["grokking_min_consec"])
            d["df_grokking"] = df_grokking
            print(f"\n── {cfg['model_label']} ────────────────────────────────────────")
            print(df_grokking[list(_GROKKING_DISPLAY_COLS)]
                  .rename(columns=_GROKKING_DISPLAY_COLS)
                  .to_string(index=False))
        else:
            d["df_grokking"] = pd.DataFrame()
            print(f"[{m}] No eval data — run submit_eval.sh + merge_results.py first.")


def compute_grokking(df_eval: pd.DataFrame, task_languages: dict, random_chance: dict,
                     threshold: float = 0.15, min_consecutive: int = 2) -> pd.DataFrame:
    """
    Apply grokking detection across all task-language pairs.

    Returns df_grokking with grokking onset and peak accuracy per (task, language).
    """
    records = []
    for task, langs in task_languages.items():
        rc, df_task = random_chance[task], df_eval[df_eval["task"] == task]
        for lang in langs:
            df_lang = df_task[df_task["language"] == lang].copy()
            if df_lang.empty:
                continue
            df_lang  = df_lang.sort_values("checkpoint", key=lambda s: s.map(ckpt_to_tokens))
            toks     = [ckpt_to_tokens(c) for c in df_lang["checkpoint"]]
            accs     = df_lang["accuracy"].tolist()
            peak_tok, peak_a = _peak_acc_info(accs, toks)
            records.append({
                "task": task, "language": lang,
                "grokking_tokens": _identify_grokking_single(accs, toks, rc, threshold, min_consecutive),
                "peak_tokens":     peak_tok,
                "peak_accuracy":   peak_a,
                "random_chance":   rc,
            })
    return pd.DataFrame(records)
