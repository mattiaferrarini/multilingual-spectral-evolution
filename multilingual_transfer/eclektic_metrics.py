"""
ECLeKTic Metrics Script
Loads judgment CSVs produced by eclektic_score.py and computes aggregated metrics.

Metrics (averaged over generation indices, std reported if multiple):
  - Overall score: QA success penalizing both missing knowledge and failed transfer.
  - Transfer score: among questions correct in the source language, fraction also
    correct in target languages.

See https://arxiv.org/abs/2502.21228 for further details.
"""

import os
import glob
import argparse
import yaml
import pandas as pd
import numpy as np


def _compute_metrics(data: pd.DataFrame) -> dict:
    correct_in_lang_qids = set(
        data[(data["correct"]) & (data["lang"] == data["original_lang"])]["q_id"].tolist()
    )
    scored_data = data[data["lang"] != data["original_lang"]]
    successes = (
        (scored_data["correct"]) & (scored_data["q_id"].isin(correct_in_lang_qids))
    ).tolist()

    overall_score = sum(successes) / len(scored_data)

    transfer_data = data[data["q_id"].isin(correct_in_lang_qids)]
    transfer_score = (
        ((transfer_data["correct"]) & (transfer_data["q_id"].isin(correct_in_lang_qids))).sum()
        / len(transfer_data)
        if len(transfer_data) > 0
        else 0.0
    )
    return {"overall_score": overall_score, "transfer_score": transfer_score}


def compute_model_metrics(judgments_path: str) -> dict:
    model_name = os.path.basename(judgments_path).replace("_judgments.csv", "")
    raw = pd.read_csv(judgments_path)
    n_judges = raw["judge"].nunique()

    # majority vote: collapse tall judge rows into one row per (q_id, lang, generation_idx)
    group_keys = ["q_id", "lang", "generation_idx", "original_lang"]
    data = (
        raw.groupby(group_keys)["correct"]
        .sum()
        .reset_index()
        .rename(columns={"correct": "n_yes"})
    )
    data["correct"] = data["n_yes"] >= (n_judges // 2 + 1)

    n_generations = data["generation_idx"].max() + 1

    per_gen = [_compute_metrics(data[data["generation_idx"] == g]) for g in range(n_generations)]
    overall_scores = [m["overall_score"] for m in per_gen]
    transfer_scores = [m["transfer_score"] for m in per_gen]

    overall_score = float(np.mean(overall_scores))
    overall_std = float(np.std(overall_scores, ddof=1)) if n_generations > 1 else None
    transfer_score = float(np.mean(transfer_scores))
    transfer_std = float(np.std(transfer_scores, ddof=1)) if n_generations > 1 else None

    print(f"\nResults for {model_name}:")
    if overall_std is not None:
        print(f"  Overall score:   {overall_score:.4f} ±{overall_std:.4f} (std over {n_generations} generations)")
        print(f"  Transfer score:  {transfer_score:.4f} ±{transfer_std:.4f}")
    else:
        print(f"  Overall score:   {overall_score:.4f}")
        print(f"  Transfer score:  {transfer_score:.4f}")

    return {
        "model": model_name,
        "overall_score": overall_score,
        "overall_std": overall_std,
        "transfer_score": transfer_score,
        "transfer_std": transfer_std,
    }


def main():
    parser = argparse.ArgumentParser(description="ECLeKTic metrics from judgment CSVs")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--judgments_dir",
        default=None,
        help="Directory containing *_judgments.csv files (defaults to config output_dir)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "results")
    judgments_dir = args.judgments_dir or output_dir

    judgment_files = sorted(glob.glob(os.path.join(judgments_dir, "*_judgments.csv")))
    if not judgment_files:
        raise FileNotFoundError(f"No *_judgments.csv files found in {judgments_dir}")

    print(f"Found {len(judgment_files)} judgment file(s):")
    for f in judgment_files:
        print(f"  {f}")

    summary_rows = [compute_model_metrics(f) for f in judgment_files]

    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
