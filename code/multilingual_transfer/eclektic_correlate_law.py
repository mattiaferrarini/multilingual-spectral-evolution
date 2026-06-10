"""
Correlate scaling-law predictors with ECLeKTic cross-lingual transfer performance.

Predictors come from build_law_predictors() — static per language-pair values derived
from scaling-law parameters (alpha, A) and observed RankMe curve shape. They don't
vary across checkpoints, so there is no time-delay or checkpoint-collapse dimension.

Usage:
    python eclektic_correlate_law.py \
        --config configs/eclektic_law_fuxi.yaml \
        --analysis-config configs/eclektic_law_correlation_analysis.yaml
"""

import argparse
import os

import yaml

from correlation_utils import _compute_correlations
from eclektic_targets import ALL_TARGETS, get_judge_targets, load_eclektic_targets, load_string_targets
from geometry_predictors import LAW_PREDICTORS, build_law_predictors


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to per-model ECLeKTic law config YAML")
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

    filter_cfg = analysis_cfg.get("filter", {})
    active_predictors = filter_cfg.get("predictors", LAW_PREDICTORS)
    active_correlations = set(filter_cfg.get("correlations", ["spearman", "pearson", "kendall"]))

    n_perm = paths.get("n_perm", 0)

    target_df = load_eclektic_targets(cfg["output_dir"], short_model, lang_codes)
    string_df = load_string_targets(cfg["output_dir"], short_model, lang_codes)
    target_df = target_df.merge(string_df, on=["src_lang", "tgt_lang"])
    print(f"Loaded {len(target_df)} ECLeKTic target pairs")

    judge_targets = get_judge_targets(cfg["output_dir"], short_model)
    all_targets = ALL_TARGETS + judge_targets
    active_targets = filter_cfg.get("normalizations", all_targets)

    print(f"Building law predictors (layer={layer}, aggregation={aggregation})")
    law_df = build_law_predictors(lang_map, law_cfg["csv"], geo_cfg["csv"], lang_map, layer, aggregation)
    print(f"Law predictors: {len(law_df)} language pairs")

    pairs_df = target_df.merge(law_df, on=["src_lang", "tgt_lang"])
    pairs_df["k"] = 0
    pairs_df["checkpoint"] = "all"
    pairs_df["layer"] = layer
    print(f"Built pairs DataFrame: {len(pairs_df)} rows")

    corr_df = _compute_correlations(
        pairs_df, n_perm=n_perm,
        predictors=active_predictors,
        normalizations=active_targets,
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
