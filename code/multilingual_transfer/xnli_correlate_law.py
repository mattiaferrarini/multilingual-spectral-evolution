"""
Correlate scaling-law predictors with XNLI cross-lingual transfer performance.

Predictors come from build_law_predictors() — static per language-pair values derived
from scaling-law parameters (alpha, A) and observed RankMe curve shape. They don't
vary across checkpoints, so there is no time-delay or checkpoint-collapse dimension.

Usage:
    python xnli_correlate_law.py \
        --config configs/xnli_law_fuxi.yaml \
        --analysis-config configs/xnli_law_correlation_analysis.yaml
"""

import argparse
import os

import yaml

from checkpoints import _checkpoint_sort_key
from correlation_utils import _compute_correlations
from geometry_predictors import LAW_PREDICTORS, build_law_predictors
from xnli_targets import NORMALIZATIONS, find_xnli_files, load_xnli_targets


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to per-model XNLI law config YAML")
    parser.add_argument("--analysis-config", required=True, help="Path to shared correlation analysis config YAML")
    return parser.parse_args()


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(analysis_cfg, model):
    corr = analysis_cfg["correlation"]
    return {k: v.format(model=model) if isinstance(v, str) else v for k, v in corr.items()}


def _build_lang_map(cfg):
    return {v: k for k, v in cfg["languages"].items()}


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    analysis_cfg = _load_config(args.analysis_config)

    short_model = cfg["model"]["name"].split("/")[-1]
    paths = _resolve_paths(analysis_cfg, short_model)

    lang_map = _build_lang_map(cfg)
    lang_codes = set(cfg["languages"].keys())

    law_cfg = cfg["law"]
    geo_cfg = cfg["geometry"]
    aggregation = law_cfg.get("aggregation", analysis_cfg["geometry"]["aggregation"])
    layer = law_cfg["layer"]

    k_values = cfg["icl"]["k"]
    if isinstance(k_values, int):
        k_values = [k_values]

    filter_cfg = analysis_cfg.get("filter", {})
    active_predictors = filter_cfg.get("predictors", LAW_PREDICTORS)
    active_normalizations = filter_cfg.get("normalizations", NORMALIZATIONS)
    active_correlations = set(filter_cfg.get("correlations", ["spearman", "pearson", "kendall"]))
    if "k_values" in filter_cfg:
        k_values = [k for k in k_values if k in filter_cfg["k_values"]]

    n_perm = paths.get("n_perm", 0)

    print(f"Building law predictors (layer={layer}, aggregation={aggregation})")
    law_df = build_law_predictors(lang_map, law_cfg["csv"], geo_cfg["csv"], lang_map, layer, aggregation)
    print(f"Law predictors: {len(law_df)} language pairs")

    xnli_files = find_xnli_files(cfg["output_dir"], short_model)
    all_ckpts = sorted(xnli_files.keys(), key=_checkpoint_sort_key)
    print(f"Found {len(all_ckpts)} XNLI checkpoints")

    target_df = load_xnli_targets(xnli_files, all_ckpts, k_values, lang_codes, t=0)
    target_df = target_df.rename(columns={"perf_checkpoint": "checkpoint"})
    print(f"Loaded {len(target_df)} XNLI target rows")

    pairs_df = target_df.merge(law_df, on=["src_lang", "tgt_lang"])
    pairs_df["layer"] = layer
    print(f"Built pairs DataFrame: {len(pairs_df)} rows")

    corr_df = _compute_correlations(
        pairs_df, n_perm=n_perm,
        predictors=active_predictors,
        normalizations=active_normalizations,
        correlations=active_correlations,
    )
    corr_df["layer"] = layer

    pairs_csv = paths["pairs_csv"]
    os.makedirs(os.path.dirname(pairs_csv), exist_ok=True)
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"\nSaved pairs to {pairs_csv}")

    results_csv = paths["results_csv"]
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    corr_df.to_csv(results_csv, index=False)
    print(f"Saved correlation results to {results_csv}")

    pooled = corr_df[corr_df["scope"] == "pooled"].drop(columns=["scope", "checkpoint"])
    print("\nPooled correlations:\n" + pooled.to_string(index=False))


if __name__ == "__main__":
    main()
