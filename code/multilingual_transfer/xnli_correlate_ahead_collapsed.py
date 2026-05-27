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
import glob
import os

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from checkpoints import _checkpoint_sort_key

PREDICTORS = ["abs_diff", "signed_diff", "min_rankme", "norm_asym"]
NORMALIZATIONS = ["row_norm", "col_norm"]
FAST_PERM_DIVISOR = 10


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


def _load_geometry(csv_path, lang_map, layer, aggregation):
    """
    Load geometry CSV, filter to given layer/aggregation and XNLI languages.
    Returns dict {(checkpoint, lang_code): rankme}.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["layer"] == layer) & (df["aggregation"] == aggregation)]
    df = df[df["dataset"].isin(lang_map)]
    result = {}
    for _, row in df.iterrows():
        code = lang_map[row["dataset"]]
        result[(row["checkpoint"], code)] = row["rankme"]
    return result


def _discover_layers(csv_path, aggregation):
    """Return layer names present in the geometry CSV, sorted numerically."""
    df = pd.read_csv(csv_path, usecols=["layer", "aggregation"])
    df = df[df["aggregation"] == aggregation]
    layers = sorted(df["layer"].unique(), key=lambda x: int(x.split("_")[1]))
    return layers


def _select_layers(all_layers, start, end, step):
    """Select a subset of layers using relative start/end (0–1) and absolute integer step."""
    n = len(all_layers)
    start_idx = round(start * (n - 1))
    end_idx = round(end * (n - 1))
    return all_layers[start_idx : end_idx + 1 : step]


def _find_xnli_files(output_dir, short_model):
    """Return {checkpoint_label: filepath} for all summary CSVs of this model."""
    pattern = os.path.join(output_dir, f"{short_model}_*_summary.csv")
    files = glob.glob(pattern)
    result = {}
    for f in files:
        basename = os.path.basename(f)
        label = basename[len(f"{short_model}_"):-len("_summary.csv")]
        result[label] = f
    return result


def _collapse_geometry_avg_rankme(csv_path, lang_map, selected_layers, aggregation):
    """Average RankMe values across layers per (checkpoint, lang_code)."""
    from collections import defaultdict
    sums = defaultdict(float)
    counts = defaultdict(int)
    for layer in selected_layers:
        geom = _load_geometry(csv_path, lang_map, layer, aggregation)
        for key, val in geom.items():
            sums[key] += val
            counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}


def _collapse_predictors_avg_predictors(pairs_dfs):
    """Average predictor columns across layers; outcomes are layer-invariant so take first."""
    key_cols = ["checkpoint", "perf_checkpoint", "k", "src_lang", "tgt_lang"]
    pred_cols = ["rankme_src", "rankme_tgt", "abs_diff", "signed_diff", "min_rankme", "norm_asym"]
    outcome_cols = ["mean_accuracy", "row_norm", "col_norm"]
    combined = pd.concat(pairs_dfs, ignore_index=True)
    pred_avg = combined.groupby(key_cols)[pred_cols].mean().reset_index()
    out_first = combined.groupby(key_cols)[outcome_cols].first().reset_index()
    return pred_avg.merge(out_first, on=key_cols)


def _build_pairs_df_ahead(xnli_files, geom, k_values, common_sorted, lang_codes, t):
    """
    Build a long DataFrame with one row per (geom_checkpoint, k, src_lang, tgt_lang), src != tgt.

    For each index i, geometry predictors come from common_sorted[i] and XNLI outcomes
    come from common_sorted[i + t]. The `checkpoint` column holds the geometry checkpoint
    (consistent with permutation test semantics). `perf_checkpoint` holds the outcome checkpoint.
    """
    rows = []
    for i in range(len(common_sorted) - t):
        geom_ckpt = common_sorted[i]
        perf_ckpt = common_sorted[i + t]

        df = pd.read_csv(xnli_files[perf_ckpt])
        df = df[df["k"].isin(k_values)]
        df = df[df["src_lang"].isin(lang_codes) & df["tgt_lang"].isin(lang_codes)]

        diag = (
            df[df["src_lang"] == df["tgt_lang"]]
            .set_index(["k", "src_lang"])["mean_accuracy"]
        )

        for _, row in df[df["src_lang"] != df["tgt_lang"]].iterrows():
            k = row["k"]
            src, tgt = row["src_lang"], row["tgt_lang"]
            acc = row["mean_accuracy"]

            rs = geom.get((geom_ckpt, src))
            rt = geom.get((geom_ckpt, tgt))
            if rs is None or rt is None:
                continue

            acc_ss = diag.get((k, src))
            acc_tt = diag.get((k, tgt))
            row_norm = acc / acc_ss if (acc_ss is not None and acc_ss > 0) else np.nan
            col_norm = acc / acc_tt if (acc_tt is not None and acc_tt > 0) else np.nan

            rows.append({
                "checkpoint": geom_ckpt,
                "perf_checkpoint": perf_ckpt,
                "k": k,
                "src_lang": src,
                "tgt_lang": tgt,
                "mean_accuracy": acc,
                "rankme_src": rs,
                "rankme_tgt": rt,
                "abs_diff": abs(rs - rt),
                "signed_diff": rs - rt,
                "min_rankme": min(rs, rt),
                "norm_asym": (rs - rt) / (rs + rt + 1e-12),
                "row_norm": row_norm,
                "col_norm": col_norm,
            })

    return pd.DataFrame(rows)


def _build_rolled_geom_avg_rankme(geom, all_geom_ckpts_sorted, common_sorted):
    """
    For each XNLI checkpoint in common_sorted, average RankMe per language over all geometry
    checkpoints (from the full CSV) up to and including that checkpoint.
    Returns {(xnli_ckpt, lang): avg_rankme} — same structure as the flat geom dict.
    """
    geom_by_ckpt = {}
    for (ckpt, lang), val in geom.items():
        geom_by_ckpt.setdefault(ckpt, {})[lang] = val

    result = {}
    for xnli_ckpt in common_sorted:
        xnli_key = _checkpoint_sort_key(xnli_ckpt)
        sums, counts = {}, {}
        for g in all_geom_ckpts_sorted:
            if _checkpoint_sort_key(g) > xnli_key:
                break
            for lang, val in geom_by_ckpt.get(g, {}).items():
                sums[lang] = sums.get(lang, 0.0) + val
                counts[lang] = counts.get(lang, 0) + 1
        for lang in sums:
            result[(xnli_ckpt, lang)] = sums[lang] / counts[lang]
    return result


def _build_pairs_df_avg_predictors(xnli_files, geom_by_ckpt, all_geom_ckpts_sorted,
                                    k_values, common_sorted, lang_codes, t):
    """
    Like _build_pairs_df_ahead but computes each predictor at every geometry checkpoint up to
    the reference XNLI checkpoint, then averages those predictor values.
    rankme_src/tgt stored are the averages of raw RankMe (used by the permutation test).
    """
    rows = []
    for i in range(len(common_sorted) - t):
        xnli_ckpt = common_sorted[i]
        perf_ckpt = common_sorted[i + t]
        xnli_key = _checkpoint_sort_key(xnli_ckpt)

        window = [g for g in all_geom_ckpts_sorted if _checkpoint_sort_key(g) <= xnli_key]
        if not window:
            continue

        df = pd.read_csv(xnli_files[perf_ckpt])
        df = df[df["k"].isin(k_values)]
        df = df[df["src_lang"].isin(lang_codes) & df["tgt_lang"].isin(lang_codes)]

        diag = (
            df[df["src_lang"] == df["tgt_lang"]]
            .set_index(["k", "src_lang"])["mean_accuracy"]
        )

        for _, row in df[df["src_lang"] != df["tgt_lang"]].iterrows():
            k = row["k"]
            src, tgt = row["src_lang"], row["tgt_lang"]
            acc = row["mean_accuracy"]

            rs_vals, rt_vals = [], []
            for g in window:
                rs = geom_by_ckpt.get(g, {}).get(src)
                rt = geom_by_ckpt.get(g, {}).get(tgt)
                if rs is not None and rt is not None:
                    rs_vals.append(rs)
                    rt_vals.append(rt)

            if not rs_vals:
                continue

            acc_ss = diag.get((k, src))
            acc_tt = diag.get((k, tgt))
            row_norm = acc / acc_ss if (acc_ss is not None and acc_ss > 0) else np.nan
            col_norm = acc / acc_tt if (acc_tt is not None and acc_tt > 0) else np.nan

            rows.append({
                "checkpoint": xnli_ckpt,
                "perf_checkpoint": perf_ckpt,
                "k": k,
                "src_lang": src,
                "tgt_lang": tgt,
                "mean_accuracy": acc,
                "rankme_src": float(np.mean(rs_vals)),
                "rankme_tgt": float(np.mean(rt_vals)),
                "abs_diff": float(np.mean([abs(rs - rt) for rs, rt in zip(rs_vals, rt_vals)])),
                "signed_diff": float(np.mean([rs - rt for rs, rt in zip(rs_vals, rt_vals)])),
                "min_rankme": float(np.mean([min(rs, rt) for rs, rt in zip(rs_vals, rt_vals)])),
                "norm_asym": float(np.mean([(rs - rt) / (rs + rt + 1e-12) for rs, rt in zip(rs_vals, rt_vals)])),
                "row_norm": row_norm,
                "col_norm": col_norm,
            })

    return pd.DataFrame(rows)


def _correlate(x, y, correlations, min_pairs=4):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < min_pairs:
        return None
    xv, yv = x[valid], y[valid]
    result = {"n": int(valid.sum())}
    if "spearman" in correlations:
        sp_r, sp_p = stats.spearmanr(xv, yv)
        result["spearman_r"] = round(float(sp_r), 4)
        result["spearman_p"] = round(float(sp_p), 4)
    if "pearson" in correlations:
        pe_r, pe_p = stats.pearsonr(xv, yv)
        result["pearson_r"] = round(float(pe_r), 4)
        result["pearson_p"] = round(float(pe_p), 4)
    if "kendall" in correlations:
        ke_r, ke_p = stats.kendalltau(xv, yv)
        result["kendall_r"] = round(float(ke_r), 4)
        result["kendall_p"] = round(float(ke_p), 4)
    return result


def _null_corr(correlations):
    result = {"n": 0}
    for coef in ("spearman", "pearson", "kendall"):
        if coef in correlations:
            result[f"{coef}_r"] = None
            result[f"{coef}_p"] = None
    return result


def _get_pred_values(rs, rt, pred_name):
    if pred_name == "abs_diff":    return np.abs(rs - rt)
    if pred_name == "signed_diff": return rs - rt
    if pred_name == "min_rankme":  return np.minimum(rs, rt)
    if pred_name == "norm_asym":   return (rs - rt) / (rs + rt + 1e-12)


def _batch_pearson(X, y):
    """X: (n_perm, n), y: (n,) — returns (n_perm,) Pearson r values."""
    Xc = X - X.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    num = Xc @ yc
    denom = np.linalg.norm(Xc, axis=1) * np.linalg.norm(yc)
    return np.where(denom > 0, num / denom, 0.0)


def _batch_spearman(X, y):
    """X: (n_perm, n), y: (n,) — returns (n_perm,) Spearman r values."""
    order = X.argsort(axis=1)
    ranks_X = np.empty_like(order, dtype=float)
    ranks_X[np.arange(len(X))[:, None], order] = np.arange(1, X.shape[1] + 1, dtype=float)
    ranks_y = y.argsort().argsort().astype(float) + 1
    return _batch_pearson(ranks_X, ranks_y)


def _permutation_p_for_checkpoint(ckpt_pairs, n_perm=1000, seed=42, fast=False,
                                   predictors=None, normalizations=None, correlations=None):
    """
    Permutation p-values for all (predictor, normalization) combinations at one checkpoint.
    Permutes RankMe values across languages — the true unit of randomisation.
    Returns {(pred, norm): {spearman_p, pearson_p, kendall_p}} (only requested coefficients).
    """
    if predictors is None:
        predictors = PREDICTORS
    if normalizations is None:
        normalizations = NORMALIZATIONS
    if correlations is None:
        correlations = {"spearman", "pearson", "kendall"}

    skip_kendall = fast or "kendall" not in correlations

    rng = np.random.default_rng(seed)

    lang_rankme = dict(zip(ckpt_pairs["src_lang"], ckpt_pairs["rankme_src"]))
    lang_rankme.update(zip(ckpt_pairs["tgt_lang"], ckpt_pairs["rankme_tgt"]))
    langs = sorted(lang_rankme)
    rankme = np.array([lang_rankme[l] for l in langs])
    lang_idx = {l: i for i, l in enumerate(langs)}

    src_idx = ckpt_pairs["src_lang"].map(lang_idx).values
    tgt_idx = ckpt_pairs["tgt_lang"].map(lang_idx).values

    y_data = {}
    for norm in normalizations:
        y_all = ckpt_pairs[norm].values.astype(float)
        mask = ~np.isnan(y_all)
        y_data[norm] = (mask, y_all[mask])

    obs = {}
    for pred in predictors:
        x_obs = _get_pred_values(rankme[src_idx], rankme[tgt_idx], pred)
        for norm in normalizations:
            mask, y = y_data[norm]
            x = x_obs[mask]
            if len(y) < 4:
                obs[(pred, norm)] = None
                continue
            obs[(pred, norm)] = (
                stats.spearmanr(x, y)[0] if "spearman" in correlations else None,
                stats.pearsonr(x, y)[0] if "pearson" in correlations else None,
                stats.kendalltau(x, y)[0] if not skip_kendall else None,
            )

    counts = {k: [0, 0, 0] for k, v in obs.items() if v is not None}

    rm_perms = np.stack([rng.permutation(rankme) for _ in range(n_perm)])

    for pred in predictors:
        rs_all = rm_perms[:, src_idx]
        rt_all = rm_perms[:, tgt_idx]
        X_all = _get_pred_values(rs_all, rt_all, pred)

        for norm in normalizations:
            if obs.get((pred, norm)) is None:
                continue
            mask, y = y_data[norm]
            X_valid = X_all[:, mask]
            r_obs = obs[(pred, norm)]
            c = counts[(pred, norm)]

            if "spearman" in correlations:
                c[0] = int((np.abs(_batch_spearman(X_valid, y)) >= abs(r_obs[0])).sum())
            if "pearson" in correlations:
                c[1] = int((np.abs(_batch_pearson(X_valid, y)) >= abs(r_obs[1])).sum())
            if not skip_kendall:
                for i in range(n_perm):
                    r_ke = stats.kendalltau(X_valid[i], y)[0]
                    c[2] += int(abs(r_ke) >= abs(r_obs[2]))

    result = {}
    for (pred, norm), v in obs.items():
        if v is None:
            r = {}
            for coef in ("spearman", "pearson", "kendall"):
                if coef in correlations:
                    r[f"{coef}_p"] = None
            result[(pred, norm)] = r
        else:
            c = counts[(pred, norm)]
            r = {}
            if "spearman" in correlations:
                r["spearman_p"] = round((c[0] + 1) / (n_perm + 1), 4)
            if "pearson" in correlations:
                r["pearson_p"] = round((c[1] + 1) / (n_perm + 1), 4)
            if "kendall" in correlations:
                r["kendall_p"] = None if skip_kendall else round((c[2] + 1) / (n_perm + 1), 4)
            result[(pred, norm)] = r
    return result


def _compute_correlations(pairs_df, n_perm=1000, fast=False,
                           predictors=None, normalizations=None, correlations=None):
    """Return tidy DataFrame: one row per (predictor, normalization, k, scope, checkpoint)."""
    if predictors is None:
        predictors = PREDICTORS
    if normalizations is None:
        normalizations = NORMALIZATIONS
    if correlations is None:
        correlations = {"spearman", "pearson", "kendall"}

    records = []
    k_values = sorted(pairs_df["k"].unique())
    checkpoints = sorted(pairs_df["checkpoint"].unique(), key=_checkpoint_sort_key)

    for pred in predictors:
        for norm in normalizations:
            for k in k_values:
                subset = pairs_df[pairs_df["k"] == k]
                base = {"predictor": pred, "normalization": norm, "k": k}
                corr = _correlate(subset[pred], subset[norm], correlations)
                records.append({**base, "scope": "pooled", "checkpoint": None, **(corr or _null_corr(correlations))})

    eff_n_perm = n_perm // FAST_PERM_DIVISOR if fast else n_perm
    print(f"Running permutation tests ({eff_n_perm} permutations × {len(checkpoints)} checkpoints × {len(k_values)} k values)"
          + (" [fast mode]" if fast else "") + "...")
    perm_pvals = {}
    for k in k_values:
        k_subset = pairs_df[pairs_df["k"] == k]
        for ckpt in checkpoints:
            ckpt_pairs = k_subset[k_subset["checkpoint"] == ckpt]
            perm_pvals[(ckpt, k)] = _permutation_p_for_checkpoint(
                ckpt_pairs, n_perm=eff_n_perm, fast=fast,
                predictors=predictors, normalizations=normalizations, correlations=correlations,
            )
            print(f"  done: k={k}  {ckpt}")

    for pred in predictors:
        for norm in normalizations:
            for k in k_values:
                subset = pairs_df[pairs_df["k"] == k]
                base = {"predictor": pred, "normalization": norm, "k": k}
                for ckpt in checkpoints:
                    s = subset[subset["checkpoint"] == ckpt]
                    corr = _correlate(s[pred], s[norm], correlations)
                    if corr is not None:
                        perm = perm_pvals[(ckpt, k)].get((pred, norm), {})
                        for coef in ("spearman", "pearson", "kendall"):
                            if coef in correlations:
                                corr[f"{coef}_p"] = perm.get(f"{coef}_p", corr.get(f"{coef}_p"))
                    records.append({**base, "scope": "per_ckpt", "checkpoint": ckpt, **(corr or _null_corr(correlations))})

    return pd.DataFrame(records)


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
    all_layers = _discover_layers(geo_cfg["csv"], aggregation)
    selected_layers = _select_layers(all_layers, layers_cfg["start"], layers_cfg["end"], layers_cfg["step"])
    print(f"Selected {len(selected_layers)} layer(s) for collapsing: {selected_layers}")

    collapse_methods = analysis_cfg["geometry"]["layer_collapse"]
    if isinstance(collapse_methods, str):
        collapse_methods = [collapse_methods]

    xnli_files = _find_xnli_files(cfg["output_dir"], short_model)

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
            collapsed_geom = _collapse_geometry_avg_rankme(
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

                    if ckpt_method is None:
                        pairs_df = _build_pairs_df_ahead(
                            xnli_files, collapsed_geom, k_values, common_sorted, lang_codes, t
                        )
                    elif ckpt_method == "average_rankme":
                        rolled_geom = _build_rolled_geom_avg_rankme(
                            collapsed_geom, all_geom_ckpts, common_sorted
                        )
                        pairs_df = _build_pairs_df_ahead(
                            xnli_files, rolled_geom, k_values, common_sorted, lang_codes, t
                        )
                    elif ckpt_method == "average_predictors":
                        geom_by_ckpt = {}
                        for (c, lang), val in collapsed_geom.items():
                            geom_by_ckpt.setdefault(c, {})[lang] = val
                        pairs_df = _build_pairs_df_avg_predictors(
                            xnli_files, geom_by_ckpt, all_geom_ckpts,
                            k_values, common_sorted, lang_codes, t
                        )
                    else:
                        raise ValueError(f"Unknown ckpt_collapse method: {ckpt_method!r}. "
                                         f"Valid options: null, 'average_rankme', 'average_predictors'")

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
                layer: _load_geometry(geo_cfg["csv"], lang_map, layer, aggregation)
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

                    layer_pairs = []
                    for layer in selected_layers:
                        geom = layer_geoms[layer]

                        if ckpt_method is None:
                            lp = _build_pairs_df_ahead(
                                xnli_files, geom, k_values, common_sorted, lang_codes, t
                            )
                        elif ckpt_method == "average_rankme":
                            rolled_geom = _build_rolled_geom_avg_rankme(
                                geom, all_geom_ckpts, common_sorted
                            )
                            lp = _build_pairs_df_ahead(
                                xnli_files, rolled_geom, k_values, common_sorted, lang_codes, t
                            )
                        elif ckpt_method == "average_predictors":
                            geom_by_ckpt = {}
                            for (c, lang), val in geom.items():
                                geom_by_ckpt.setdefault(c, {})[lang] = val
                            lp = _build_pairs_df_avg_predictors(
                                xnli_files, geom_by_ckpt, all_geom_ckpts,
                                k_values, common_sorted, lang_codes, t
                            )
                        else:
                            raise ValueError(f"Unknown ckpt_collapse method: {ckpt_method!r}. "
                                             f"Valid options: null, 'average_rankme', 'average_predictors'")

                        layer_pairs.append(lp)

                    pairs_df = _collapse_predictors_avg_predictors(layer_pairs)
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
