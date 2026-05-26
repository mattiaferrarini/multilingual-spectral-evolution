"""
Correlate RankMe geometry with cross-lingual XNLI transfer within each checkpoint.

For each (src, tgt, checkpoint, k) triple (src != tgt), computes four RankMe-based
predictors and correlates them against row-normalized and col-normalized transfer
scores. P-values use a permutation test that respects language-level non-independence.

Outputs: pairs CSV and correlation results CSV. For plots, run xnli_plot_within_ckpt.py.

Predictors:
  abs_diff    : |RankMe(src) - RankMe(tgt)|
  signed_diff : RankMe(src) - RankMe(tgt)
  min_rankme  : min(RankMe(src), RankMe(tgt))
  norm_asym   : (RankMe(src) - RankMe(tgt)) / (RankMe(src) + RankMe(tgt))

Outcomes:
  row_norm : acc(src, tgt) / acc(src, src)  — how well src transfers out
  col_norm : acc(src, tgt) / acc(tgt, tgt)  — how well tgt is served by foreign context

Usage:
    python xnli_correlate_within_ckpt.py --config configs/xnli_apertus.yaml \
                                          --analysis-config configs/xnli_correlation_analysis.yaml
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


def _build_pairs_df(xnli_files, geom, k_values, common_ckpts, lang_codes):
    """
    Build a long DataFrame with one row per (checkpoint, k, src_lang, tgt_lang), src != tgt.
    Adds predictor columns and row_norm / col_norm outcomes.
    """
    rows = []
    for ckpt in common_ckpts:
        df = pd.read_csv(xnli_files[ckpt])
        df = df[df["k"].isin(k_values)]
        df = df[df["src_lang"].isin(lang_codes) & df["tgt_lang"].isin(lang_codes)]

        # Diagonal accuracies for normalization denominators
        diag = (
            df[df["src_lang"] == df["tgt_lang"]]
            .set_index(["k", "src_lang"])["mean_accuracy"]
        )

        for _, row in df[df["src_lang"] != df["tgt_lang"]].iterrows():
            k = row["k"]
            src, tgt = row["src_lang"], row["tgt_lang"]
            acc = row["mean_accuracy"]

            rs = geom.get((ckpt, src))
            rt = geom.get((ckpt, tgt))
            if rs is None or rt is None:
                continue

            acc_ss = diag.get((k, src))
            acc_tt = diag.get((k, tgt))
            row_norm = acc / acc_ss if (acc_ss is not None and acc_ss > 0) else np.nan
            col_norm = acc / acc_tt if (acc_tt is not None and acc_tt > 0) else np.nan

            rows.append({
                "checkpoint": ckpt,
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


def _correlate(x, y, min_pairs=4):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < min_pairs:
        return None
    xv, yv = x[valid], y[valid]
    sp_r, sp_p = stats.spearmanr(xv, yv)
    pe_r, pe_p = stats.pearsonr(xv, yv)
    ke_r, ke_p = stats.kendalltau(xv, yv)
    return {
        "n": int(valid.sum()),
        "spearman_r": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 4),
        "pearson_r": round(float(pe_r), 4),
        "pearson_p": round(float(pe_p), 4),
        "kendall_r": round(float(ke_r), 4),
        "kendall_p": round(float(ke_p), 4),
    }


def _null_corr():
    return {"n": 0, "spearman_r": None, "spearman_p": None,
            "pearson_r": None, "pearson_p": None,
            "kendall_r": None, "kendall_p": None}


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


def _permutation_p_for_checkpoint(ckpt_pairs, n_perm=1000, seed=42, fast=False):
    """
    Permutation p-values for all (predictor, normalization) combinations at one checkpoint.
    Permutes RankMe values across languages — the true unit of randomisation.
    Returns {(pred, norm): {spearman_p, pearson_p, kendall_p}}.
    """
    rng = np.random.default_rng(seed)

    lang_rankme = dict(zip(ckpt_pairs["src_lang"], ckpt_pairs["rankme_src"]))
    lang_rankme.update(zip(ckpt_pairs["tgt_lang"], ckpt_pairs["rankme_tgt"]))
    langs = sorted(lang_rankme)
    rankme = np.array([lang_rankme[l] for l in langs])
    lang_idx = {l: i for i, l in enumerate(langs)}

    src_idx = ckpt_pairs["src_lang"].map(lang_idx).values
    tgt_idx = ckpt_pairs["tgt_lang"].map(lang_idx).values

    # Precompute y vectors and valid masks per norm
    y_data = {}
    for norm in NORMALIZATIONS:
        y_all = ckpt_pairs[norm].values.astype(float)
        mask = ~np.isnan(y_all)
        y_data[norm] = (mask, y_all[mask])

    # Observed correlations
    obs = {}
    for pred in PREDICTORS:
        x_obs = _get_pred_values(rankme[src_idx], rankme[tgt_idx], pred)
        for norm in NORMALIZATIONS:
            mask, y = y_data[norm]
            x = x_obs[mask]
            if len(y) < 4:
                obs[(pred, norm)] = None
                continue
            obs[(pred, norm)] = (
                stats.spearmanr(x, y)[0],
                stats.pearsonr(x, y)[0],
                stats.kendalltau(x, y)[0],
            )

    counts = {k: [0, 0, 0] for k, v in obs.items() if v is not None}

    rm_perms = np.stack([rng.permutation(rankme) for _ in range(n_perm)])

    for pred in PREDICTORS:
        rs_all = rm_perms[:, src_idx]
        rt_all = rm_perms[:, tgt_idx]
        X_all = _get_pred_values(rs_all, rt_all, pred)

        for norm in NORMALIZATIONS:
            if obs.get((pred, norm)) is None:
                continue
            mask, y = y_data[norm]
            X_valid = X_all[:, mask]
            r_obs = obs[(pred, norm)]
            c = counts[(pred, norm)]

            c[0] = int((np.abs(_batch_spearman(X_valid, y)) >= abs(r_obs[0])).sum())
            c[1] = int((np.abs(_batch_pearson(X_valid, y)) >= abs(r_obs[1])).sum())

            if not fast:
                for i in range(n_perm):
                    r_ke = stats.kendalltau(X_valid[i], y)[0]
                    c[2] += int(abs(r_ke) >= abs(r_obs[2]))

    result = {}
    for (pred, norm), v in obs.items():
        if v is None:
            result[(pred, norm)] = {"spearman_p": None, "pearson_p": None, "kendall_p": None}
        else:
            c = counts[(pred, norm)]
            result[(pred, norm)] = {
                "spearman_p": round((c[0] + 1) / (n_perm + 1), 4),
                "pearson_p":  round((c[1] + 1) / (n_perm + 1), 4),
                "kendall_p":  None if fast else round((c[2] + 1) / (n_perm + 1), 4),
            }
    return result


def _compute_correlations(pairs_df, n_perm=1000, fast=False):
    """Return tidy DataFrame: one row per (predictor, normalization, k, scope, checkpoint)."""
    records = []
    k_values = sorted(pairs_df["k"].unique())
    checkpoints = sorted(pairs_df["checkpoint"].unique(), key=_checkpoint_sort_key)

    # Pooled rows (standard p-values — pairs not independent, treat as approximate)
    for pred in PREDICTORS:
        for norm in NORMALIZATIONS:
            for k in k_values:
                subset = pairs_df[pairs_df["k"] == k]
                base = {"predictor": pred, "normalization": norm, "k": k}
                corr = _correlate(subset[pred], subset[norm])
                records.append({**base, "scope": "pooled", "checkpoint": None, **(corr or _null_corr())})

    # Per-checkpoint: compute permutation p-values once per (ckpt, k), reuse across predictors
    eff_n_perm = n_perm // FAST_PERM_DIVISOR if fast else n_perm
    print(f"Running permutation tests ({eff_n_perm} permutations × {len(checkpoints)} checkpoints × {len(k_values)} k values)"
          + (" [fast mode]" if fast else "") + "...")
    perm_pvals = {}
    for k in k_values:
        k_subset = pairs_df[pairs_df["k"] == k]
        for ckpt in checkpoints:
            ckpt_pairs = k_subset[k_subset["checkpoint"] == ckpt]
            perm_pvals[(ckpt, k)] = _permutation_p_for_checkpoint(ckpt_pairs, n_perm=eff_n_perm, fast=fast)
            print(f"  done: k={k}  {ckpt}")

    for pred in PREDICTORS:
        for norm in NORMALIZATIONS:
            for k in k_values:
                subset = pairs_df[pairs_df["k"] == k]
                base = {"predictor": pred, "normalization": norm, "k": k}
                for ckpt in checkpoints:
                    s = subset[subset["checkpoint"] == ckpt]
                    corr = _correlate(s[pred], s[norm])
                    if corr is not None:
                        perm = perm_pvals[(ckpt, k)].get((pred, norm), {})
                        corr["spearman_p"] = perm.get("spearman_p", corr["spearman_p"])
                        corr["pearson_p"]  = perm.get("pearson_p",  corr["pearson_p"])
                        corr["kendall_p"]  = perm.get("kendall_p",  corr["kendall_p"])
                    records.append({**base, "scope": "per_ckpt", "checkpoint": ckpt, **(corr or _null_corr())})

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

    layers_cfg = analysis_cfg["geometry"].get("layers", {"start": 1.0, "end": 1.0, "step": 1})
    all_layers = _discover_layers(geo_cfg["csv"], aggregation)
    selected_layers = _select_layers(all_layers, layers_cfg["start"], layers_cfg["end"], layers_cfg["step"])
    print(f"Selected {len(selected_layers)} layer(s): {selected_layers}")

    xnli_files = _find_xnli_files(cfg["output_dir"], short_model)

    k_values = cfg["icl"]["k"]
    if isinstance(k_values, int):
        k_values = [k_values]

    n_perm = paths.get("n_perm", 1000)
    all_pairs, all_corr = [], []

    for layer in selected_layers:
        print(f"\n--- Layer: {layer} ---")
        geom = _load_geometry(geo_cfg["csv"], lang_map, layer, aggregation)

        geom_ckpts = {ckpt for (ckpt, _) in geom}
        xnli_ckpts = set(xnli_files)
        common = geom_ckpts & xnli_ckpts

        only_geom = geom_ckpts - xnli_ckpts
        only_xnli = xnli_ckpts - geom_ckpts
        if only_geom:
            print(f"Warning: {len(only_geom)} geometry checkpoint(s) with no XNLI data — skipped")
        if only_xnli:
            print(f"Warning: {len(only_xnli)} XNLI checkpoint(s) with no geometry data — skipped")

        common_sorted = sorted(common, key=_checkpoint_sort_key)
        print(f"Processing {len(common_sorted)} checkpoints with both geometry and XNLI data")

        pairs_df = _build_pairs_df(xnli_files, geom, k_values, common_sorted, lang_codes)
        pairs_df["layer"] = layer
        print(f"Built pairs DataFrame: {len(pairs_df)} rows")

        corr_df = _compute_correlations(pairs_df, n_perm=n_perm, fast=args.fast)
        corr_df["layer"] = layer

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
