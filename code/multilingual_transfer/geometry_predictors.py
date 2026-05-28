"""
RankMe geometry loading and predictor computation.

Public API:
  PREDICTORS                            -- list of predictor names
  load_geometry(csv_path, lang_map, layer, aggregation)
  discover_layers(csv_path, aggregation)
  select_layers(all_layers, start, end, step)
  get_pred_values(rs, rt, pred_name)
  build_rolled_geom_avg_rankme(geom, all_geom_ckpts_sorted, common_sorted)
  collapse_geometry_avg_rankme(csv_path, lang_map, selected_layers, aggregation)
  build_predictor_pairs(geom, common_sorted, lang_codes, t)
  build_predictor_pairs_avg_checkpoints(geom_by_ckpt, all_geom_ckpts_sorted, common_sorted, lang_codes, t)
  collapse_predictor_pairs_avg_layers(predictor_dfs)
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from checkpoints import _checkpoint_sort_key

PREDICTORS = ["abs_diff", "signed_diff", "min_rankme", "norm_asym"]


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
    if pred_name == "abs_diff":    return np.abs(rs - rt)
    if pred_name == "signed_diff": return rs - rt
    if pred_name == "min_rankme":  return np.minimum(rs, rt)
    if pred_name == "norm_asym":   return (rs - rt) / (rs + rt + 1e-12)


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
             rankme_src, rankme_tgt, abs_diff, signed_diff, min_rankme, norm_asym
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
                })
    return pd.DataFrame(rows)


def build_predictor_pairs_avg_checkpoints(geom_by_ckpt, all_geom_ckpts_sorted,
                                           common_sorted, lang_codes, t):
    """
    Like build_predictor_pairs but computes each predictor at every geometry checkpoint up to
    the reference checkpoint, then averages those values.
    rankme_src/tgt stored are the averages of raw RankMe (used by permutation tests).

    Columns: checkpoint, perf_checkpoint, src_lang, tgt_lang,
             rankme_src, rankme_tgt, abs_diff, signed_diff, min_rankme, norm_asym
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
                })
    return pd.DataFrame(rows)


def collapse_predictor_pairs_avg_layers(predictor_dfs):
    """
    Average predictor columns across layers.
    predictor_dfs: list of DataFrames from build_predictor_pairs or
                   build_predictor_pairs_avg_checkpoints (no target columns).
    """
    key_cols = ["checkpoint", "perf_checkpoint", "src_lang", "tgt_lang"]
    pred_cols = ["rankme_src", "rankme_tgt", "abs_diff", "signed_diff", "min_rankme", "norm_asym"]
    combined = pd.concat(predictor_dfs, ignore_index=True)
    return combined.groupby(key_cols)[pred_cols].mean().reset_index()
