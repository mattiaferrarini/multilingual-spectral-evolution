#!/usr/bin/env python3
"""
Merge downstream evaluation results with RankMe spectral metrics.

Parses all lm-eval JSON outputs produced by evaluate.py and joins them
with a RankMe CSV (e.g. results/fuxi_wiki.csv) by matching on
(checkpoint, language).

Output is a single merged CSV ready for the results.ipynb analysis.

Usage:
    python downstream_evaluation/merge_results.py \\
        --eval-dir  results/eval \\
        --rankme-csv results/fuxi_wiki.csv \\
        --output    results/merged.csv

    # Use a specific layer and aggregation for the RankMe side
    python downstream_evaluation/merge_results.py \\
        --eval-dir  results/eval \\
        --rankme-csv results/fuxi_wiki.csv \\
        --output    results/merged.csv \\
        --layer layer_29 --aggregation last
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Metric keys tried in priority order when parsing lm-eval JSON outputs
_METRIC_KEYS = ["acc,none", "acc_norm,none", "exact_match,none"]

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "benchmarks.yaml"

def _load_benchmark_meta(config_path: Path) -> dict:
    with open(config_path) as f:
        benchmarks = yaml.safe_load(f)["benchmarks"]
    return {
        task: {"num_fewshot": cfg["num_fewshot"], "random_chance": cfg["random_chance"]}
        for task, cfg in benchmarks.items()
    }


def _ckpt_to_tokens(name: str) -> float:
    """Convert checkpoint name to billions of tokens (for sorting)."""
    m = re.match(r"^step\d+-tokens(\d+)([BT])$", str(name))
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2) == "T" else val
    m = re.match(r"^(\d+(?:\.\d+)?)", str(name))
    return float(m.group(1)) if m else float("inf")


def parse_eval_directory(eval_dir: Path, benchmark_meta: dict | None = None) -> pd.DataFrame:
    """
    Recursively scan eval_dir for lm-eval JSON result files.

    Expected file layout (produced by evaluate.py):
        eval_dir/{checkpoint}/{task}__{language}.json

    Returns DataFrame with columns:
        checkpoint, language, task, lm_task_name,
        accuracy, num_fewshot, random_chance
    """
    json_files = sorted(eval_dir.rglob("*.json"))
    logger.info(f"Scanning {eval_dir}: found {len(json_files)} JSON file(s).")

    records = []
    for jf in json_files:
        stem = jf.stem
        if "__" not in stem:
            logger.debug(f"Skipping {jf.name} — unexpected name format.")
            continue

        task, language = stem.split("__", 1)
        checkpoint = jf.parent.name

        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {jf}: {e}")
            continue

        # Extract the first available accuracy metric
        accuracy = None
        lm_task_name = None
        for task_key, metrics in data.get("results", {}).items():
            lm_task_name = task_key
            for mk in _METRIC_KEYS:
                v = metrics.get(mk)
                if v is not None:
                    accuracy = float(v)
                    break
            if accuracy is not None:
                break

        if accuracy is None:
            logger.warning(f"No accuracy metric found in {jf}")
            continue

        meta = (benchmark_meta or {}).get(task, {})
        records.append({
            "checkpoint":    checkpoint,
            "language":      language,
            "task":          task,
            "lm_task_name":  lm_task_name,
            "accuracy":      accuracy,
            "num_fewshot":   meta.get("num_fewshot"),
            "random_chance": meta.get("random_chance"),
        })

    if not records:
        logger.warning("No valid evaluation results found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(
        ["task", "language", "checkpoint"],
        key=lambda s: s.map(_ckpt_to_tokens) if s.name == "checkpoint" else s,
    ).reset_index(drop=True)
    logger.info(f"Parsed {len(df)} evaluation records.")
    return df


def merge_with_rankme(
    df_eval: pd.DataFrame,
    rankme_csv: Path,
    layer: str = "layer_29",
    aggregation: str = "last",
) -> pd.DataFrame:
    """
    Left-join eval results onto RankMe data, matching on (checkpoint, language).

    Filters the RankMe CSV to the chosen layer and aggregation before merging.
    The 'dataset' column in the RankMe CSV is renamed to 'language' to match.
    """
    df_rankme = pd.read_csv(rankme_csv)
    logger.info(f"RankMe CSV: {len(df_rankme):,} rows loaded from {rankme_csv}")

    df_rankme = df_rankme[
        (df_rankme["layer"] == layer) &
        (df_rankme["aggregation"] == aggregation)
    ].copy()
    df_rankme = df_rankme.rename(columns={"dataset": "language"})
    df_rankme["rankme_layer"]       = layer
    df_rankme["rankme_aggregation"] = aggregation
    logger.info(f"After filtering ({layer}, agg={aggregation}): {len(df_rankme)} rows")

    rankme_cols = ["checkpoint", "language", "rankme", "pr", "alpha_req",
                   "rankme_layer", "rankme_aggregation"]
    merged = df_eval.merge(
        df_rankme[rankme_cols],
        on=["checkpoint", "language"],
        how="left",
    )

    n_matched = merged["rankme"].notna().sum()
    logger.info(
        f"Merge complete: {n_matched}/{len(merged)} eval rows matched a RankMe value "
        f"({len(merged) - n_matched} unmatched — checkpoint name mismatch or language not in RankMe CSV)."
    )
    return merged


def main():
    p = argparse.ArgumentParser(description="Merge lm-eval results with RankMe metrics.")
    p.add_argument("--eval-dir",    required=True, help="Directory with lm-eval JSON outputs.")
    p.add_argument("--rankme-csv",  required=True, help="Path to RankMe metrics CSV.")
    p.add_argument("--output",      required=True, help="Output path for the merged CSV.")
    p.add_argument("--layer",       default="layer_29",
                   help="Which layer's RankMe values to include (default: layer_29).")
    p.add_argument("--aggregation", default="last",
                   help="Which aggregation to use (default: last).")
    args = p.parse_args()

    eval_dir  = Path(args.eval_dir)
    rankme_csv = Path(args.rankme_csv)

    if not eval_dir.exists():
        logger.error(f"Eval directory does not exist: {eval_dir}")
        return
    if not rankme_csv.exists():
        logger.error(f"RankMe CSV does not exist: {rankme_csv}")
        return

    benchmark_meta = _load_benchmark_meta(_DEFAULT_CONFIG)
    df_eval = parse_eval_directory(eval_dir, benchmark_meta)
    if df_eval.empty:
        logger.error("No evaluation results to merge — run evaluate.py first.")
        return

    merged = merge_with_rankme(df_eval, rankme_csv, layer=args.layer, aggregation=args.aggregation)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    logger.info(f"Merged CSV saved → {out}  ({len(merged)} rows)")

    print("\n── Accuracy summary ──────────────────────────────────────────────")
    print(merged.groupby(["task", "language"])[["accuracy", "rankme"]].first().to_string())


if __name__ == "__main__":
    main()
