"""
ECLeKTic-specific target loading.

Public API:
  TARGETS        -- LLM-judge target column names
  STRING_TARGETS -- string-matching target column names (suffixed with _str)
  ALL_TARGETS    -- TARGETS + STRING_TARGETS
  get_judge_targets(output_dir, short_model)
  load_eclektic_targets(output_dir, short_model, lang_codes)
  load_string_targets(output_dir, short_model, lang_codes)
"""

import os

import pandas as pd

TARGETS = ["accuracy_majority", "overall_score", "transfer_score"]
STRING_TARGETS = ["transfer_score_str", "overall_score_str", "transfer_margin_str", "overall_margin_str"]
ALL_TARGETS = TARGETS + STRING_TARGETS


def get_judge_targets(output_dir, short_model):
    """Return per-judge accuracy column names from the judgments CSV (excludes accuracy_majority)."""
    csv_path = os.path.join(output_dir, f"{short_model}_judgments_per_pair.csv")
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return [c for c in cols if c.startswith("accuracy_") and c != "accuracy_majority"]


def load_eclektic_targets(output_dir, short_model, lang_codes):
    """
    Load LLM-judge ECLeKTic scores for each (src_lang, tgt_lang) pair.

    Returns a DataFrame with columns:
      src_lang, tgt_lang, accuracy_majority, overall_score, transfer_score,
      accuracy_{judge} ... (one column per judge found in the CSV)
    Only cross-lingual pairs (src != tgt) whose both languages are in lang_codes are kept.
    """
    csv_path = os.path.join(output_dir, f"{short_model}_judgments_per_pair.csv")
    df = pd.read_csv(csv_path)

    df = df[df["original_lang"].isin(lang_codes) & df["lang"].isin(lang_codes)]
    df = df[df["original_lang"] != df["lang"]]

    judge_cols = [c for c in df.columns if c.startswith("accuracy_") and c != "accuracy_majority"]
    df = df[["original_lang", "lang"] + TARGETS + judge_cols].copy()
    df = df.rename(columns={"original_lang": "src_lang", "lang": "tgt_lang"})
    return df.reset_index(drop=True)


def load_string_targets(output_dir, short_model, lang_codes):
    """
    Load string-matching ECLeKTic scores for each (src_lang, tgt_lang) pair.

    Returns a DataFrame with columns:
      src_lang, tgt_lang, transfer_score_str, overall_score_str,
      transfer_margin_str, overall_margin_str
    Only pairs whose both languages are in lang_codes are kept.
    """
    csv_path = os.path.join(output_dir, f"{short_model}_string_scores_per_pair.csv")
    df = pd.read_csv(csv_path)

    df = df[df["original_language"].isin(lang_codes) & df["target_language"].isin(lang_codes)]
    df = df.rename(columns={
        "original_language": "src_lang",
        "target_language":   "tgt_lang",
        "transfer_score":    "transfer_score_str",
        "overall_score":     "overall_score_str",
        "transfer_margin":   "transfer_margin_str",
        "overall_margin":    "overall_margin_str",
    })
    return df[["src_lang", "tgt_lang"] + STRING_TARGETS].reset_index(drop=True)
