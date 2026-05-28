"""
Generic correlation analysis utilities (Pearson, Spearman, Kendall + permutation tests).

Public API:
  FAST_PERM_DIVISOR
  _correlate(x, y, correlations, min_pairs=4)
  _null_corr(correlations)
  _batch_pearson(X, y)
  _batch_spearman(X, y)
  _permutation_p_for_checkpoint(ckpt_pairs, ...)
  _compute_correlations(pairs_df, ...)
"""

import numpy as np
import pandas as pd
from scipy import stats

from checkpoints import _checkpoint_sort_key
from geometry_predictors import PREDICTORS, get_pred_values
from xnli_targets import NORMALIZATIONS

FAST_PERM_DIVISOR = 10


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
        x_obs = get_pred_values(rankme[src_idx], rankme[tgt_idx], pred)
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
        X_all = get_pred_values(rs_all, rt_all, pred)

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
