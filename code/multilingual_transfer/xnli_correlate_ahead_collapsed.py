"""
Correlate RankMe geometry at checkpoint C with XNLI transfer performance at checkpoint C+T,
collapsing all selected layers into a single predictor value before computing correlations.

Two layer collapse strategies (configured via geometry.layer_collapse list):
  average_rankme     : average RankMe values across layers per (checkpoint, lang_code) first,
                       then compute predictors from the averaged values.
  average_predictors : compute predictors per layer, then average predictor values across layers.

Optional checkpoint collapse (set `ckpt_collapse` list in analysis config under `correlation:`):
  null              : single checkpoint (default)
  average_rankme    : average RankMe values per language over all geometry checkpoints ≤ C,
                      then compute predictors from the averaged values.
  average_predictors: compute predictors at each geometry checkpoint ≤ C, then average them.

All 6 combinations of (layer_collapse × ckpt_collapse) are run and tagged in the output.

For each T in t_values, and for each (src, tgt, geom_checkpoint, k) triple (src != tgt),
computes four RankMe-based predictors from checkpoint C and correlates them against
row-normalized and col-normalized transfer scores from checkpoint C+T.
P-values use a permutation test that respects language-level non-independence.

Outputs: pairs CSV and correlation results CSV (both include `t`, `layer_collapse`, and `ckpt_collapse` columns).

Predictors (from geometry checkpoint C, collapsed across layers):
  abs_diff    : |RankMe(src) - RankMe(tgt)|
  signed_diff : RankMe(src) - RankMe(tgt)
  min_rankme  : min(RankMe(src), RankMe(tgt))
  norm_asym   : (RankMe(src) - RankMe(tgt)) / (RankMe(src) + RankMe(tgt))

Outcomes (from performance checkpoint C+T):
  row_norm : acc(src, tgt) / acc(src, src)  — how well src transfers out
  col_norm : acc(src, tgt) / acc(tgt, tgt)  — how well tgt is served by foreign context

Usage:
    python xnli_correlate_ahead_collapsed.py \\
        --config configs/xnli_apertus.yaml \\
        --analysis-config configs/xnli_correlation_ahead_collapsed_analysis.yaml
"""

import argparse
import os

import pandas as pd
import yaml

from checkpoints import _checkpoint_sort_key
from correlation_utils import FAST_PERM_DIVISOR, _compute_correlations
from geometry_predictors import (
    PREDICTORS,
    build_predictor_pairs,
    build_predictor_pairs_avg_checkpoints,
    build_rolled_geom_avg_rankme,
    collapse_geometry_avg_rankme,
    collapse_predictor_pairs_avg_layers,
    discover_layers,
    load_geometry,
    select_layers,
)
from xnli_targets import NORMALIZATIONS, find_xnli_files, load_xnli_targets

PAIRS_COLS = [
    "checkpoint", "perf_checkpoint", "k", "src_lang", "tgt_lang",
    "mean_accuracy",
    *PREDICTORS,
    *NORMALIZATIONS,
]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to per-model XNLI config YAML")
    parser.add_argument("--analysis-config", required=True, help="Path to shared correlation analysis config YAML")
    parser.add_argument("--fast", action="store_true",
                        help=f"Fast mode: divide n_perm by {FAST_PERM_DIVISOR}, skip Kendall tau")
    return parser.parse_args()


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(analysis_cfg, model):
    """Substitute {model} in all template paths and return a flat dict."""
    corr = analysis_cfg["correlation"]
    return {k: v.format(model=model) if isinstance(v, str) else v for k, v in corr.items()}


def _build_lang_map(cfg):
    """Return {full_name: code}, e.g. {'Arabic': 'ar'}."""
    return {v: k for k, v in cfg["languages"].items()}


def _build_pred_df_for_ckpt_method(ckpt_method, geom, all_geom_ckpts,
                                    common_sorted, lang_codes, t):
    """Return pred_df (predictor-only DataFrame) for the given checkpoint collapse method."""
    if ckpt_method is None:
        return build_predictor_pairs(geom, common_sorted, lang_codes, t)
    elif ckpt_method == "average_rankme":
        rolled_geom = build_rolled_geom_avg_rankme(geom, all_geom_ckpts, common_sorted)
        return build_predictor_pairs(rolled_geom, common_sorted, lang_codes, t)
    elif ckpt_method == "average_predictors":
        geom_by_ckpt = {}
        for (ckpt, lang), val in geom.items():
            geom_by_ckpt.setdefault(ckpt, {})[lang] = val
        return build_predictor_pairs_avg_checkpoints(
            geom_by_ckpt, all_geom_ckpts, common_sorted, lang_codes, t
        )
    else:
        raise ValueError(f"Unknown ckpt_collapse method: {ckpt_method!r}. "
                         f"Valid options: null, 'average_rankme', 'average_predictors'")


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

    layers_cfg = analysis_cfg["geometry"].get("layers", {"start": 0.0, "end": 1.0, "step": 1})
    all_layers = discover_layers(geo_cfg["csv"], aggregation)
    selected_layers = select_layers(all_layers, layers_cfg["start"], layers_cfg["end"], layers_cfg["step"])
    print(f"Selected {len(selected_layers)} layer(s) for collapsing: {selected_layers}")

    collapse_methods = analysis_cfg["geometry"]["layer_collapse"]
    if isinstance(collapse_methods, str):
        collapse_methods = [collapse_methods]

    xnli_files = find_xnli_files(cfg["output_dir"], short_model)

    k_values = cfg["icl"]["k"]
    if isinstance(k_values, int):
        k_values = [k_values]

    t_values = paths.get("t_values", [1])
    if isinstance(t_values, int):
        t_values = [t_values]

    filter_cfg = analysis_cfg.get("filter", {})
    active_predictors = filter_cfg.get("predictors", PREDICTORS)
    active_normalizations = filter_cfg.get("normalizations", NORMALIZATIONS)
    active_correlations = set(filter_cfg.get("correlations", ["spearman", "pearson", "kendall"]))
    if "k_values" in filter_cfg:
        k_values = [k for k in k_values if k in filter_cfg["k_values"]]

    n_perm = paths.get("n_perm", 1000)

    ckpt_collapse_methods = paths.get("ckpt_collapse", [None])
    if not ckpt_collapse_methods:
        ckpt_collapse_methods = [None]
    if not isinstance(ckpt_collapse_methods, list):
        ckpt_collapse_methods = [ckpt_collapse_methods]

    all_pairs, all_corr = [], []

    for layer_method in collapse_methods:
        print(f"\n=== Layer collapse: {layer_method} ===")

        if layer_method == "average_rankme":
            collapsed_geom = collapse_geometry_avg_rankme(
                geo_cfg["csv"], lang_map, selected_layers, aggregation
            )
            all_geom_ckpts = sorted({ckpt for (ckpt, _) in collapsed_geom}, key=_checkpoint_sort_key)
            xnli_ckpts = set(xnli_files)
            common = set(all_geom_ckpts) & xnli_ckpts

            only_geom = set(all_geom_ckpts) - xnli_ckpts
            only_xnli = xnli_ckpts - set(all_geom_ckpts)
            if only_geom:
                print(f"Warning: {len(only_geom)} geometry checkpoint(s) with no XNLI data — skipped")
            if only_xnli:
                print(f"Warning: {len(only_xnli)} XNLI checkpoint(s) with no geometry data — skipped")

            common_sorted = sorted(common, key=_checkpoint_sort_key)
            print(f"Found {len(common_sorted)} checkpoints with both geometry and XNLI data")

            for ckpt_method in ckpt_collapse_methods:
                ckpt_label = ckpt_method if ckpt_method is not None else "none"
                print(f"\n--- Checkpoint collapse: {ckpt_label} ---")

                for t in t_values:
                    n_geom_ckpts = len(common_sorted) - t
                    if n_geom_ckpts <= 0:
                        print(f"  t={t}: not enough checkpoints ({len(common_sorted)} available), skipping")
                        continue
                    print(f"\n  t={t}: using {n_geom_ckpts} geometry checkpoint(s)")

                    pred_df = _build_pred_df_for_ckpt_method(
                        ckpt_method, collapsed_geom, all_geom_ckpts,
                        common_sorted, lang_codes, t
                    )
                    target_df = load_xnli_targets(xnli_files, common_sorted, k_values, lang_codes, t)
                    pairs_df = target_df.merge(pred_df, on=["perf_checkpoint", "src_lang", "tgt_lang"])[PAIRS_COLS]
                    pairs_df["layer_collapse"] = layer_method
                    pairs_df["ckpt_collapse"] = ckpt_label
                    pairs_df["t"] = t
                    print(f"  Built pairs DataFrame: {len(pairs_df)} rows")

                    corr_df = _compute_correlations(
                        pairs_df, n_perm=n_perm, fast=args.fast,
                        predictors=active_predictors,
                        normalizations=active_normalizations,
                        correlations=active_correlations,
                    )
                    corr_df["layer_collapse"] = layer_method
                    corr_df["ckpt_collapse"] = ckpt_label
                    corr_df["t"] = t

                    all_pairs.append(pairs_df)
                    all_corr.append(corr_df)

        elif layer_method == "average_predictors":
            layer_geoms = {
                layer: load_geometry(geo_cfg["csv"], lang_map, layer, aggregation)
                for layer in selected_layers
            }
            all_geom_ckpts = sorted(
                {ckpt for (ckpt, _) in layer_geoms[selected_layers[0]]},
                key=_checkpoint_sort_key
            )
            xnli_ckpts = set(xnli_files)
            common = set(all_geom_ckpts) & xnli_ckpts

            only_geom = set(all_geom_ckpts) - xnli_ckpts
            only_xnli = xnli_ckpts - set(all_geom_ckpts)
            if only_geom:
                print(f"Warning: {len(only_geom)} geometry checkpoint(s) with no XNLI data — skipped")
            if only_xnli:
                print(f"Warning: {len(only_xnli)} XNLI checkpoint(s) with no geometry data — skipped")

            common_sorted = sorted(common, key=_checkpoint_sort_key)
            print(f"Found {len(common_sorted)} checkpoints with both geometry and XNLI data")

            for ckpt_method in ckpt_collapse_methods:
                ckpt_label = ckpt_method if ckpt_method is not None else "none"
                print(f"\n--- Checkpoint collapse: {ckpt_label} ---")

                for t in t_values:
                    n_geom_ckpts = len(common_sorted) - t
                    if n_geom_ckpts <= 0:
                        print(f"  t={t}: not enough checkpoints ({len(common_sorted)} available), skipping")
                        continue
                    print(f"\n  t={t}: using {n_geom_ckpts} geometry checkpoint(s)")

                    layer_preds = []
                    for layer in selected_layers:
                        layer_preds.append(_build_pred_df_for_ckpt_method(
                            ckpt_method, layer_geoms[layer], all_geom_ckpts,
                            common_sorted, lang_codes, t
                        ))

                    collapsed_pred = collapse_predictor_pairs_avg_layers(layer_preds)
                    target_df = load_xnli_targets(xnli_files, common_sorted, k_values, lang_codes, t)

                    # Sort matches the original groupby-sorted output of _collapse_predictors_avg_predictors
                    pairs_df = (
                        collapsed_pred.merge(target_df, on=["perf_checkpoint", "src_lang", "tgt_lang"])[PAIRS_COLS]
                        .sort_values(["checkpoint", "perf_checkpoint", "k", "src_lang", "tgt_lang"])
                        .reset_index(drop=True)
                    )
                    pairs_df["layer_collapse"] = layer_method
                    pairs_df["ckpt_collapse"] = ckpt_label
                    pairs_df["t"] = t
                    print(f"  Built pairs DataFrame: {len(pairs_df)} rows")

                    corr_df = _compute_correlations(
                        pairs_df, n_perm=n_perm, fast=args.fast,
                        predictors=active_predictors,
                        normalizations=active_normalizations,
                        correlations=active_correlations,
                    )
                    corr_df["layer_collapse"] = layer_method
                    corr_df["ckpt_collapse"] = ckpt_label
                    corr_df["t"] = t

                    all_pairs.append(pairs_df)
                    all_corr.append(corr_df)

        else:
            raise ValueError(f"Unknown layer_collapse method: {layer_method!r}. "
                             f"Valid options: 'average_rankme', 'average_predictors'")

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
