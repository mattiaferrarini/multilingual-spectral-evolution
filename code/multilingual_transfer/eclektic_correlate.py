"""
Correlate RankMe geometry predictors with ECLeKTic transfer performance.

ECLeKTic provides a single set of downstream scores (no per-checkpoint performance),
so there is no time-delay dimension. For each geometry checkpoint C and language pair
(src, tgt), the four RankMe-based predictors are correlated against the fixed ECLeKTic
scores (accuracy_majority, overall_score, transfer_score).

Optional checkpoint collapse (set `ckpt_collapse` list in analysis config under `correlation:`):
  null              : use geometry from each checkpoint independently (default)
  average_rankme    : average RankMe values per language over all geometry checkpoints ≤ C,
                      then compute predictors from the averaged values.
  average_predictors: compute predictors at each geometry checkpoint ≤ C, then average them.

Predictors (from geometry at checkpoint C):
  abs_diff    : |RankMe(src) - RankMe(tgt)|
  signed_diff : RankMe(src) - RankMe(tgt)
  min_rankme  : min(RankMe(src), RankMe(tgt))
  norm_asym   : (RankMe(src) - RankMe(tgt)) / (RankMe(src) + RankMe(tgt))

Targets (fixed ECLeKTic scores per language pair):
  accuracy_majority : majority-vote LLM-judge accuracy
  overall_score     : overall ECLeKTic score
  transfer_score    : cross-lingual transfer score

Usage:
    python eclektic_correlate.py --config configs/eclektic_apertus.yaml \
                                  --analysis-config configs/eclektic_correlation_analysis.yaml
"""

import argparse
import os

import pandas as pd
import yaml

from checkpoints import _checkpoint_sort_key
from correlation_utils import FAST_PERM_DIVISOR, _compute_correlations
from eclektic_targets import ALL_TARGETS, TARGETS, get_judge_targets, load_eclektic_targets, load_string_targets
from geometry_predictors import (
    PREDICTORS,
    build_predictor_pairs,
    build_predictor_pairs_avg_checkpoints,
    build_rolled_geom_avg_rankme,
    discover_layers,
    load_geometry,
    select_layers,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to per-model ECLeKTic config YAML")
    parser.add_argument("--analysis-config", required=True, help="Path to shared correlation analysis config YAML")
    parser.add_argument("--fast", action="store_true",
                        help=f"Fast mode: divide n_perm by {FAST_PERM_DIVISOR}, skip Kendall tau")
    return parser.parse_args()


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(analysis_cfg, model):
    corr = analysis_cfg["correlation"]
    return {k: v.format(model=model) if isinstance(v, str) else v for k, v in corr.items()}


def _build_lang_map(cfg):
    """Return {full_name: code}, e.g. {'German': 'de'}."""
    return {v: k for k, v in cfg["languages"].items()}


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    analysis_cfg = _load_config(args.analysis_config)

    short_model = cfg["model"]["name"].split("/")[-1]
    paths = _resolve_paths(analysis_cfg, short_model)

    lang_map = _build_lang_map(cfg)
    lang_codes = set(cfg["languages"].keys())

    geo_cfg = cfg["geometry"]
    aggregation = analysis_cfg["geometry"]["aggregation"]

    layers_cfg = analysis_cfg["geometry"].get("layers", {"start": 1.0, "end": 1.0, "step": 1})
    all_layers = discover_layers(geo_cfg["csv"], aggregation)
    selected_layers = select_layers(all_layers, layers_cfg["start"], layers_cfg["end"], layers_cfg["step"])
    print(f"Selected {len(selected_layers)} layer(s): {selected_layers}")

    target_df = load_eclektic_targets(cfg["output_dir"], short_model, lang_codes)
    string_df = load_string_targets(cfg["output_dir"], short_model, lang_codes)
    target_df = target_df.merge(string_df, on=["src_lang", "tgt_lang"])
    print(f"Loaded {len(target_df)} ECLeKTic target pairs")

    judge_targets = get_judge_targets(cfg["output_dir"], short_model)
    all_targets = ALL_TARGETS + judge_targets

    filter_cfg = analysis_cfg.get("filter", {})
    active_predictors = filter_cfg.get("predictors", PREDICTORS)
    active_targets = filter_cfg.get("normalizations", all_targets)
    active_correlations = set(filter_cfg.get("correlations", ["spearman", "pearson", "kendall"]))

    n_perm = paths.get("n_perm", 1000)

    ckpt_collapse_methods = paths.get("ckpt_collapse", [None])
    if not ckpt_collapse_methods:
        ckpt_collapse_methods = [None]
    if not isinstance(ckpt_collapse_methods, list):
        ckpt_collapse_methods = [ckpt_collapse_methods]

    all_pairs, all_corr = [], []

    for ckpt_method in ckpt_collapse_methods:
        method_label = ckpt_method if ckpt_method is not None else "none"
        print(f"\n=== Checkpoint collapse: {method_label} ===")

        for layer in selected_layers:
            print(f"\n--- Layer: {layer} ---")
            geom = load_geometry(geo_cfg["csv"], lang_map, layer, aggregation)

            all_geom_ckpts = sorted({ckpt for (ckpt, _) in geom}, key=_checkpoint_sort_key)
            common_sorted = all_geom_ckpts  # no XNLI intersection needed
            print(f"Found {len(common_sorted)} geometry checkpoints")

            if ckpt_method is None:
                pred_df = build_predictor_pairs(geom, common_sorted, lang_codes, t=0)
            elif ckpt_method == "average_rankme":
                rolled_geom = build_rolled_geom_avg_rankme(geom, all_geom_ckpts, common_sorted)
                pred_df = build_predictor_pairs(rolled_geom, common_sorted, lang_codes, t=0)
            elif ckpt_method == "average_predictors":
                geom_by_ckpt = {}
                for (ckpt, lang), val in geom.items():
                    geom_by_ckpt.setdefault(ckpt, {})[lang] = val
                pred_df = build_predictor_pairs_avg_checkpoints(
                    geom_by_ckpt, all_geom_ckpts, common_sorted, lang_codes, t=0
                )
            else:
                raise ValueError(f"Unknown ckpt_collapse method: {ckpt_method!r}. "
                                 f"Valid options: null, 'average_rankme', 'average_predictors'")

            # Drop perf_checkpoint — targets are fixed, not per-checkpoint
            pred_df = pred_df.drop(columns=["perf_checkpoint"], errors="ignore")

            pairs_df = pred_df.merge(target_df, on=["src_lang", "tgt_lang"])
            pairs_df["k"] = 0  # dummy column required by _compute_correlations
            pairs_df["layer"] = layer
            pairs_df["ckpt_collapse"] = method_label
            print(f"Built pairs DataFrame: {len(pairs_df)} rows")

            corr_df = _compute_correlations(
                pairs_df, n_perm=n_perm, fast=args.fast,
                predictors=active_predictors,
                normalizations=active_targets,
                correlations=active_correlations,
            )
            corr_df["layer"] = layer
            corr_df["ckpt_collapse"] = method_label

            all_pairs.append(pairs_df)
            all_corr.append(corr_df)

    pairs_df = pd.concat(all_pairs, ignore_index=True)
    corr_df = pd.concat(all_corr, ignore_index=True)

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
