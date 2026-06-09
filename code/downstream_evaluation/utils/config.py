"""
Configuration loader for joao_pinto_407597.ipynb.

Centralizes all paths, layer settings, grokking parameters, and benchmark
metadata derived from benchmarks.yaml. The notebook only needs to call
init_models() to get everything.
"""

from pathlib import Path

import yaml

_RESULTS_DIR     = Path(__file__).parent.parent.parent.parent / "results"
_BENCHMARKS_PATH = Path(__file__).parent.parent / "configs" / "benchmarks.yaml"

_MODEL_CONFIGS = {
    "fuxi": {
        "rankme_csv": _RESULTS_DIR / "fuxi_fine_wiki.csv",
        "merged_csv": _RESULTS_DIR / "fuxi_fine_wiki_mmlu_xcopa_belebele_marc_layer21_merged.csv",
        "layer":      "layer_21",
        "label":      "FuxiTranyu-8B",
    },
    "fuxi_intermediate": {
        "rankme_csv": _RESULTS_DIR / "fuxi_fine_wiki.csv",
        "merged_csv": _RESULTS_DIR / "fuxi_intermediate_merged.csv",
        "layer":      "layer_29",
        "label":      "FuxiTranyu-8B (intermediate)",
    },
    "apertus": {
        "rankme_csv": _RESULTS_DIR / "apertus_fine_wiki.csv",
        "merged_csv": _RESULTS_DIR / "apertus_fine_wiki_mmlu_xcopa_belebele_marc_layer19_merged.csv",
        "layer":      "layer_19",
        "label":      "Apertus-8B-2509",
    },
}

# Grokking detection parameters — adjust here for ablation studies.
_GROKKING_THRESHOLD  = 0.15  # accuracy must exceed random_chance + this value
_GROKKING_MIN_CONSEC = 2     # for at least this many consecutive checkpoints

_PALETTE = {
    "fuxi":    {"color": "steelblue",  "marker": "o"},
    "apertus": {"color": "darkorange", "marker": "s"},
}

_DEFAULT_MODELS = ["fuxi", "apertus"]


def load_config(model: str) -> dict:
    """
    Return all configuration variables for the given model as a dict.

    Usage in notebook:
        cfg = load_config(MODEL)

    Keys: rankme_csv, merged_csv, layer, aggregation, model_label,
          grokking_threshold, grokking_min_consec, random_chance,
          task_languages, plots_dir
    """
    if model not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model}'. Choose: {list(_MODEL_CONFIGS)}")

    _cfg = _MODEL_CONFIGS[model]

    with open(_BENCHMARKS_PATH) as f:
        _benchmarks = yaml.safe_load(f)["benchmarks"]

    plots_dir = Path("plots_joao_pinto_407597")
    plots_dir.mkdir(exist_ok=True)

    return {
        "rankme_csv":          _cfg["rankme_csv"],
        "merged_csv":          _cfg["merged_csv"],
        "layer":               _cfg["layer"],
        "aggregation":         "last",
        "model_label":         _cfg["label"],
        "grokking_threshold":  _GROKKING_THRESHOLD,
        "grokking_min_consec": _GROKKING_MIN_CONSEC,
        "random_chance":       {t: c["random_chance"]          for t, c in _benchmarks.items()},
        "task_languages":      {t: list(c["languages"].keys()) for t, c in _benchmarks.items()},
        "plots_dir":           plots_dir,
    }


def init_models(model_keys: list | None = None) -> tuple[list, dict]:
    """
    Initialise the per-model data dict used throughout the notebook.

    Returns (MODELS, data) where data[m] contains cfg, color, and marker
    for each model key m.  Extend data[m] with computed DataFrames in
    subsequent cells (df_rankme, df_layer, df_geometry, …).

    Usage in notebook:
        MODELS, data = init_models()
    """
    if model_keys is None:
        model_keys = _DEFAULT_MODELS
    data = {}
    for m in model_keys:
        cfg = load_config(m)
        palette = _PALETTE.get(m, {"color": "gray", "marker": "o"})
        data[m] = {
            "cfg":    cfg,
            "color":  palette["color"],
            "marker": palette["marker"],
        }
        print(f"[{m}] {cfg['model_label']}  |  Layer: {cfg['layer']}  |  Aggregation: {cfg['aggregation']}")
    return model_keys, data
