"""
Generic correlation analysis utilities (Pearson, Spearman, Kendall + permutation tests).

Public API:
  FAST_PERM_DIVISOR
  _correlate(x, y, correlations, min_pairs=4)
  _null_corr(correlations)
  _batch_pearson(X, y)
  _batch_spearman(X, y)
  _parse_law_pred(pred_name)
  _get_law_pred_values(ss, tt, suffix)
  _permutation_p_for_checkpoint(ckpt_pairs, ...)
  _permutation_p_for_law(ckpt_pairs, ...)
  _compute_correlations(pairs_df, ...)
"""

import numpy as np
import pandas as pd
from scipy import stats

from checkpoints import _checkpoint_sort_key
from geometry_predictors import (
    LAW_CURVE_PARAMS, LAW_PHASE1_PARAMS, LAW_PREDICTORS,
    PREDICTORS, get_pred_values,
)
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


_LAW_PRED_SET = frozenset(LAW_PREDICTORS)
_LAW_SUFFIXES = frozenset(["abs_diff", "signed_diff", "min", "max",
                            "norm_asym", "abs_ratio", "signed_ratio", "log_ratio"])


def _parse_law_pred(pred_name):
    """Split e.g. 'drop_to_min_abs_diff' → ('drop_to_min', 'abs_diff').

    Matches by suffix so multi-underscore base params (drop_to_min, drop_minus_recovery)
    are handled correctly.
    """
    for s in _LAW_SUFFIXES:
        if pred_name.endswith("_" + s):
            return pred_name[: -(len(s) + 1)], s
    raise ValueError(f"Cannot parse law predictor name: {pred_name!r}")


def _get_law_pred_values(ss, tt, suffix):
    """Apply a law-predictor suffix formula to source/target parameter arrays.

    ss, tt : ndarray of shape (n_pairs,) or (n_perm, n_pairs)
    suffix : one of the 8 known suffixes

    Works for both shapes via NumPy broadcasting, so the same function serves
    both the observed-statistic computation (1-D) and the batch permutation (2-D).
    Uses log(|x| + eps) for log_ratio because law params can be negative.
    """
    eps = 1e-12
    if suffix == "abs_diff":     return np.abs(ss - tt)
    if suffix == "signed_diff":  return ss - tt
    if suffix == "min":          return np.minimum(ss, tt)
    if suffix == "max":          return np.maximum(ss, tt)
    if suffix == "norm_asym":    return (ss - tt) / (ss + tt + eps)
    if suffix == "abs_ratio":    return np.maximum(ss, tt) / (np.minimum(ss, tt) + eps)
    if suffix == "signed_ratio": return ss / (tt + eps)
    if suffix == "log_ratio":    return np.log(np.abs(ss) + eps) - np.log(np.abs(tt) + eps)
    raise ValueError(f"Unknown law predictor suffix: {suffix!r}")


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


def _permutation_p_for_law(ckpt_pairs, n_perm=1000, seed=42, fast=False,
                            predictors=None, normalizations=None, correlations=None):
    """
    Permutation p-values for law predictors at one (checkpoint, k) slice.

    Permutes per-language parameter vectors across language labels — the correct
    unit of randomisation (mirrors _permutation_p_for_checkpoint for RankMe).
    Pairs sharing a language are kept structurally consistent across permutations.

    Requires {p}_src and {p}_tgt columns in ckpt_pairs for each base param p.
    Does NOT require rankme_src/rankme_tgt.

    Returns {(pred, norm): {spearman_p, pearson_p, kendall_p}} — same format as
    _permutation_p_for_checkpoint.
    """
    if predictors is None:
        predictors = LAW_PREDICTORS
    if normalizations is None:
        normalizations = NORMALIZATIONS
    if correlations is None:
        correlations = {"spearman", "pearson", "kendall"}

    skip_kendall = fast or "kendall" not in correlations
    rng = np.random.default_rng(seed)

    base_params = LAW_PHASE1_PARAMS + LAW_CURVE_PARAMS  # 6 params

    # Build per-language parameter matrix: shape (n_langs, n_base_params).
    # Law params are static per language; read from {p}_src columns.
    all_langs = sorted(set(ckpt_pairs["src_lang"]) | set(ckpt_pairs["tgt_lang"]))
    lang_idx = {l: i for i, l in enumerate(all_langs)}
    n_langs = len(all_langs)

    param_matrix = np.zeros((n_langs, len(base_params)), dtype=float)
    for i, lang in enumerate(all_langs):
        src_rows = ckpt_pairs[ckpt_pairs["src_lang"] == lang]
        if len(src_rows):
            for j, p in enumerate(base_params):
                param_matrix[i, j] = src_rows[f"{p}_src"].iloc[0]
        else:
            tgt_rows = ckpt_pairs[ckpt_pairs["tgt_lang"] == lang]
            for j, p in enumerate(base_params):
                param_matrix[i, j] = tgt_rows[f"{p}_tgt"].iloc[0]

    src_idx = ckpt_pairs["src_lang"].map(lang_idx).values  # (n_pairs,)
    tgt_idx = ckpt_pairs["tgt_lang"].map(lang_idx).values  # (n_pairs,)

    # Pre-process target columns.
    y_data = {}
    for norm in normalizations:
        y_all = ckpt_pairs[norm].values.astype(float)
        mask = ~np.isnan(y_all)
        y_data[norm] = (mask, y_all[mask])

    # Observed correlations.
    obs = {}
    for pred in predictors:
        base, suffix = _parse_law_pred(pred)
        param_col = base_params.index(base)
        x_obs = _get_law_pred_values(
            param_matrix[src_idx, param_col],
            param_matrix[tgt_idx, param_col],
            suffix,
        )
        for norm in normalizations:
            mask, y = y_data[norm]
            x = x_obs[mask]
            if len(y) < 4:
                obs[(pred, norm)] = None
                continue
            obs[(pred, norm)] = (
                stats.spearmanr(x, y)[0]  if "spearman" in correlations else None,
                stats.pearsonr(x, y)[0]   if "pearson"  in correlations else None,
                stats.kendalltau(x, y)[0] if not skip_kendall else None,
            )

    counts = {k: [0, 0, 0] for k, v in obs.items() if v is not None}

    # Generate permutation index matrix: (n_perm, n_langs).
    # Each row is a full permutation of language positions — all base params
    # for a language are shuffled together, preserving within-language structure.
    perm_indices = np.stack([rng.permutation(n_langs) for _ in range(n_perm)])

    # Run permutation test.
    for pred in predictors:
        base, suffix = _parse_law_pred(pred)
        param_col = base_params.index(base)

        # Advanced indexing: perm_indices[:, src_idx] → (n_perm, n_pairs)
        # selects the permuted language position for each pair's source language.
        ss_all = param_matrix[perm_indices[:, src_idx], param_col]  # (n_perm, n_pairs)
        tt_all = param_matrix[perm_indices[:, tgt_idx], param_col]  # (n_perm, n_pairs)
        X_all = _get_law_pred_values(ss_all, tt_all, suffix)         # (n_perm, n_pairs)

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
    use_law_perm = bool(predictors) and set(predictors).issubset(_LAW_PRED_SET)
    perm_fn = _permutation_p_for_law if use_law_perm else _permutation_p_for_checkpoint
    perm_pvals = {}
    if eff_n_perm > 0:
        print(f"Running {'law ' if use_law_perm else ''}permutation tests "
              f"({eff_n_perm} permutations × {len(checkpoints)} checkpoints × {len(k_values)} k values)"
              + (" [fast mode]" if fast else "") + "...")
        for k in k_values:
            k_subset = pairs_df[pairs_df["k"] == k]
            for ckpt in checkpoints:
                ckpt_pairs = k_subset[k_subset["checkpoint"] == ckpt]
                perm_pvals[(ckpt, k)] = perm_fn(
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
                        perm = perm_pvals.get((ckpt, k), {}).get((pred, norm), {})
                        for coef in ("spearman", "pearson", "kendall"):
                            if coef in correlations:
                                corr[f"{coef}_p"] = perm.get(f"{coef}_p", corr.get(f"{coef}_p"))
                    records.append({**base, "scope": "per_ckpt", "checkpoint": ckpt, **(corr or _null_corr(correlations))})

    return pd.DataFrame(records)
