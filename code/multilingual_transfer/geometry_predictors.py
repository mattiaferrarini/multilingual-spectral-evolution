"""
RankMe geometry loading and predictor computation.

Public API:
  PREDICTORS                            -- list of checkpoint-based predictor names
  LAW_PHASE1_PARAMS                     -- language-specific phase-1 parameters used
  LAW_CURVE_PARAMS                      -- curve-shape scalar names derived from geometry
  LAW_PREDICTORS                        -- list of static law-based predictor names
  load_geometry(csv_path, lang_map, layer, aggregation)
  discover_layers(csv_path, aggregation)
  select_layers(all_layers, start, end, step)
  get_pred_values(rs, rt, pred_name)
  build_rolled_geom_avg_rankme(geom, all_geom_ckpts_sorted, common_sorted)
  collapse_geometry_avg_rankme(csv_path, lang_map, selected_layers, aggregation)
  build_predictor_pairs(geom, common_sorted, lang_codes, t)
  build_predictor_pairs_avg_checkpoints(geom_by_ckpt, all_geom_ckpts_sorted, common_sorted, lang_codes, t)
  collapse_predictor_pairs_avg_layers(predictor_dfs)
  load_law_params(law_csv_path)
  build_law_predictors(law_lang_map, law_csv_path,
                       geom_csv_path, geom_lang_map, layer, aggregation="last")

Checkpoint-based predictors: abs_diff, signed_diff, min_rankme, norm_asym,
                              abs_ratio, signed_ratio, max_rankme, log_ratio
Static law-based predictors:  {alpha,A,drop_to_min,recovery,
                                drop_minus_recovery,drop_over_recovery}
                              × {abs_diff,signed_diff,min,max,
                                 norm_asym,abs_ratio,signed_ratio,log_ratio}
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from checkpoints import _checkpoint_sort_key

PREDICTORS = ["rankme_src", "rankme_tgt",
              "abs_diff", "signed_diff", "min_rankme", "norm_asym",
              "abs_ratio", "signed_ratio", "max_rankme", "log_ratio"]

PREDICTOR_LABELS = {
    "rankme_src":   "src",
    "rankme_tgt":   "tgt",
    "abs_diff":     "|src − tgt|",
    "signed_diff":  "src − tgt",
    "min_rankme":   "min(src, tgt)",
    "norm_asym":    "(src − tgt) / (src + tgt)",
    "abs_ratio":    "max(src, tgt) / min(src, tgt)",
    "signed_ratio": "src / tgt",
    "max_rankme":   "max(src, tgt)",
    "log_ratio":    "log(src / tgt)",
}

LAW_PHASE1_PARAMS = ["alpha", "A"]

LAW_CURVE_PARAMS = [
    "drop_to_min",
    "recovery",
    "drop_minus_recovery",
    "drop_over_recovery",
]

LAW_PREDICTORS = [
    f"{p}_{s}"
    for p in (LAW_PHASE1_PARAMS + LAW_CURVE_PARAMS)
    for s in ["abs_diff", "signed_diff", "min", "max",
              "norm_asym", "abs_ratio", "signed_ratio", "log_ratio"]
]


def load_geometry(csv_path, lang_map, layer, aggregation):
    """
    Load geometry CSV, filter to given layer/aggregation and languages in lang_map.
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


def discover_layers(csv_path, aggregation):
    """Return layer names present in the geometry CSV, sorted numerically."""
    df = pd.read_csv(csv_path, usecols=["layer", "aggregation"])
    df = df[df["aggregation"] == aggregation]
    return sorted(df["layer"].unique(), key=lambda x: int(x.split("_")[1]))


def select_layers(all_layers, start, end, step):
    """Select a subset of layers using relative start/end (0–1) and absolute integer step."""
    n = len(all_layers)
    start_idx = round(start * (n - 1))
    end_idx = round(end * (n - 1))
    return all_layers[start_idx : end_idx + 1 : step]


def get_pred_values(rs, rt, pred_name):
    """Compute a single predictor from (arrays of) RankMe values for src and tgt."""
    if pred_name == "rankme_src":   return rs
    if pred_name == "rankme_tgt":   return rt
    if pred_name == "abs_diff":     return np.abs(rs - rt)
    if pred_name == "signed_diff":  return rs - rt
    if pred_name == "min_rankme":   return np.minimum(rs, rt)
    if pred_name == "norm_asym":    return (rs - rt) / (rs + rt + 1e-12)
    if pred_name == "abs_ratio":    return np.maximum(rs, rt) / (np.minimum(rs, rt) + 1e-12)
    if pred_name == "signed_ratio": return rs / (rt + 1e-12)
    if pred_name == "max_rankme":   return np.maximum(rs, rt)
    if pred_name == "log_ratio":    return np.log(rs + 1e-12) - np.log(rt + 1e-12)


def build_rolled_geom_avg_rankme(geom, all_geom_ckpts_sorted, common_sorted):
    """
    For each checkpoint in common_sorted, average RankMe per language over all geometry
    checkpoints up to and including that checkpoint.
    Returns {(ckpt, lang): avg_rankme} — same structure as the flat geom dict.
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


def collapse_geometry_avg_rankme(csv_path, lang_map, selected_layers, aggregation):
    """Average RankMe values across layers per (checkpoint, lang_code)."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for layer in selected_layers:
        geom = load_geometry(csv_path, lang_map, layer, aggregation)
        for key, val in geom.items():
            sums[key] += val
            counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}


def build_predictor_pairs(geom, common_sorted, lang_codes, t):
    """
    Build a DataFrame with one row per (geom_checkpoint, src_lang, tgt_lang), src != tgt.

    For each index i, geometry predictors come from common_sorted[i]; perf_checkpoint is
    common_sorted[i + t]. No target values are included — join with load_xnli_targets() or
    equivalent on ["perf_checkpoint", "src_lang", "tgt_lang"].

    Columns: checkpoint, perf_checkpoint, src_lang, tgt_lang,
             rankme_src, rankme_tgt, abs_diff, signed_diff, min_rankme, norm_asym,
             abs_ratio, signed_ratio, max_rankme, log_ratio
    """
    rows = []
    langs = sorted(lang_codes)
    for i in range(len(common_sorted) - t):
        geom_ckpt = common_sorted[i]
        perf_ckpt = common_sorted[i + t]
        for src in langs:
            for tgt in langs:
                if src == tgt:
                    continue
                rs = geom.get((geom_ckpt, src))
                rt = geom.get((geom_ckpt, tgt))
                if rs is None or rt is None:
                    continue
                rows.append({
                    "checkpoint": geom_ckpt,
                    "perf_checkpoint": perf_ckpt,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "rankme_src": rs,
                    "rankme_tgt": rt,
                    "abs_diff": abs(rs - rt),
                    "signed_diff": rs - rt,
                    "min_rankme": min(rs, rt),
                    "norm_asym": (rs - rt) / (rs + rt + 1e-12),
                    "abs_ratio": max(rs, rt) / (min(rs, rt) + 1e-12),
                    "signed_ratio": rs / (rt + 1e-12),
                    "max_rankme": max(rs, rt),
                    "log_ratio": np.log(rs + 1e-12) - np.log(rt + 1e-12),
                })
    return pd.DataFrame(rows)


def build_predictor_pairs_avg_checkpoints(geom_by_ckpt, all_geom_ckpts_sorted,
                                           common_sorted, lang_codes, t):
    """
    Like build_predictor_pairs but computes each predictor at every geometry checkpoint up to
    the reference checkpoint, then averages those values.
    rankme_src/tgt stored are the averages of raw RankMe (used by permutation tests).

    Columns: checkpoint, perf_checkpoint, src_lang, tgt_lang,
             rankme_src, rankme_tgt, abs_diff, signed_diff, min_rankme, norm_asym,
             abs_ratio, signed_ratio, max_rankme, log_ratio
    """
    rows = []
    langs = sorted(lang_codes)
    for i in range(len(common_sorted) - t):
        ref_ckpt = common_sorted[i]
        perf_ckpt = common_sorted[i + t]
        ref_key = _checkpoint_sort_key(ref_ckpt)

        window = [g for g in all_geom_ckpts_sorted if _checkpoint_sort_key(g) <= ref_key]
        if not window:
            continue

        for src in langs:
            for tgt in langs:
                if src == tgt:
                    continue
                rs_vals, rt_vals = [], []
                for g in window:
                    rs = geom_by_ckpt.get(g, {}).get(src)
                    rt = geom_by_ckpt.get(g, {}).get(tgt)
                    if rs is not None and rt is not None:
                        rs_vals.append(rs)
                        rt_vals.append(rt)
                if not rs_vals:
                    continue
                rows.append({
                    "checkpoint": ref_ckpt,
                    "perf_checkpoint": perf_ckpt,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "rankme_src": float(np.mean(rs_vals)),
                    "rankme_tgt": float(np.mean(rt_vals)),
                    "abs_diff": float(np.mean([abs(rs - rt) for rs, rt in zip(rs_vals, rt_vals)])),
                    "signed_diff": float(np.mean([rs - rt for rs, rt in zip(rs_vals, rt_vals)])),
                    "min_rankme": float(np.mean([min(rs, rt) for rs, rt in zip(rs_vals, rt_vals)])),
                    "norm_asym": float(np.mean([(rs - rt) / (rs + rt + 1e-12) for rs, rt in zip(rs_vals, rt_vals)])),
                    "abs_ratio": float(np.mean([max(rs, rt) / (min(rs, rt) + 1e-12) for rs, rt in zip(rs_vals, rt_vals)])),
                    "signed_ratio": float(np.mean([rs / (rt + 1e-12) for rs, rt in zip(rs_vals, rt_vals)])),
                    "max_rankme": float(np.mean([max(rs, rt) for rs, rt in zip(rs_vals, rt_vals)])),
                    "log_ratio": float(np.mean([np.log(rs + 1e-12) - np.log(rt + 1e-12) for rs, rt in zip(rs_vals, rt_vals)])),
                })
    return pd.DataFrame(rows)


def collapse_predictor_pairs_avg_layers(predictor_dfs):
    """
    Average predictor columns across layers.
    predictor_dfs: list of DataFrames from build_predictor_pairs or
                   build_predictor_pairs_avg_checkpoints (no target columns).
    """
    key_cols = ["checkpoint", "perf_checkpoint", "src_lang", "tgt_lang"]
    pred_cols = ["rankme_src", "rankme_tgt", "abs_diff", "signed_diff", "min_rankme", "norm_asym",
                 "abs_ratio", "signed_ratio", "max_rankme", "log_ratio"]
    combined = pd.concat(predictor_dfs, ignore_index=True)
    return combined.groupby(key_cols)[pred_cols].mean().reset_index()


def load_law_params(law_csv_path):
    """
    Load all scaling law parameters from a fitted-parameter CSV.

    Returns {full_lang_name: {param: value}} for all columns in the CSV.
    """
    df = pd.read_csv(law_csv_path)
    return {row["language"]: dict(row) for _, row in df.iterrows()}


def _compute_geom_curve_scalars(geom, sorted_ckpts, lang_codes):
    """
    Compute per-language curve-shape scalars from observed RankMe values.

    geom:         {(checkpoint, lang_code): rankme}  from load_geometry
    sorted_ckpts: checkpoints ordered by _checkpoint_sort_key
    lang_codes:   iterable of lang codes to compute

    Returns {lang_code: {"drop_to_min": float, "recovery": float,
                          "drop_minus_recovery": float, "drop_over_recovery": float}}
    """
    eps = 1e-12
    result = {}
    for lang in lang_codes:
        vals = [geom[(ck, lang)] for ck in sorted_ckpts if (ck, lang) in geom]
        if len(vals) < 2:
            continue
        rankme_first = vals[0]
        min_val = min(vals)
        min_idx = vals.index(min_val)
        max_after_min = max(vals[min_idx:])
        drop = rankme_first - min_val
        rec  = max_after_min - min_val
        result[lang] = {
            "drop_to_min":         drop,
            "recovery":            rec,
            "drop_minus_recovery": drop - rec,
            "drop_over_recovery":  drop / (rec + eps),
        }
    return result


def build_law_predictors(law_lang_map, law_csv_path,
                         geom_csv_path, geom_lang_map,
                         layer, aggregation="last"):
    """
    Build a DataFrame of static predictors for each language pair.

    law_lang_map:  {full_lang_name: lang_code}  e.g. {"Arabic": "ar", ...}
                   Keys must match the "language" column in the fitted-parameter CSV.
    law_csv_path:  path to the fitted scaling-law parameters CSV.
    geom_csv_path: path to the geometry CSV (RankMe measurements).
    geom_lang_map: {dataset_name: lang_code}  e.g. {"Arabic": "ar", ...}
                   Keys must match the "dataset" column in the geometry CSV.
    layer:         geometry layer to use (e.g. "layer_29"); read from caller's config.

    Returns a DataFrame with one row per (src_lang, tgt_lang) where src_lang != tgt_lang.
    Columns: src_lang, tgt_lang,
             {p}_src, {p}_tgt for each p in LAW_PHASE1_PARAMS + LAW_CURVE_PARAMS,
             plus all LAW_PREDICTORS columns.
    """
    raw = load_law_params(law_csv_path)
    params = {law_lang_map[name]: vals for name, vals in raw.items() if name in law_lang_map}

    geom = load_geometry(geom_csv_path, geom_lang_map, layer, aggregation)
    sorted_ckpts = sorted({ck for ck, _ in geom}, key=_checkpoint_sort_key)
    curve_scalars = _compute_geom_curve_scalars(geom, sorted_ckpts, params.keys())

    lang_codes = sorted(k for k in params if k in curve_scalars)
    eps = 1e-12
    rows = []
    for src in lang_codes:
        for tgt in lang_codes:
            if src == tgt:
                continue
            row = {"src_lang": src, "tgt_lang": tgt}
            all_params = list(LAW_PHASE1_PARAMS) + list(LAW_CURVE_PARAMS)
            src_vals = {**{p: params[src][p] for p in LAW_PHASE1_PARAMS},
                        **curve_scalars[src]}
            tgt_vals = {**{p: params[tgt][p] for p in LAW_PHASE1_PARAMS},
                        **curve_scalars[tgt]}
            for p in all_params:
                s, t = src_vals[p], tgt_vals[p]
                row[f"{p}_src"] = s
                row[f"{p}_tgt"] = t
                row[f"{p}_abs_diff"]     = abs(s - t)
                row[f"{p}_signed_diff"]  = s - t
                row[f"{p}_min"]          = min(s, t)
                row[f"{p}_max"]          = max(s, t)
                row[f"{p}_norm_asym"]    = (s - t) / (s + t + eps)
                row[f"{p}_abs_ratio"]    = max(s, t) / (min(s, t) + eps)
                row[f"{p}_signed_ratio"] = s / (t + eps)
                row[f"{p}_log_ratio"]    = np.log(abs(s) + eps) - np.log(abs(t) + eps)
            rows.append(row)
    return pd.DataFrame(rows)
