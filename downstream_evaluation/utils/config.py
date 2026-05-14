"""
Configuration loader for results.ipynb.

Centralizes all paths, layer settings, grokking parameters, and benchmark
metadata derived from benchmarks.yaml. The notebook only needs to set
MODEL = "fuxi" | "apertus" and call load_config(MODEL) to get everything.
"""

from pathlib import Path

import yaml

_MODEL_CONFIGS = {
    "fuxi": {
        "rankme_csv": Path("../results/fuxi.csv"),
        "merged_csv": Path("../results/fuxi_merged.csv"),
        "layer":      "layer_29",
        "label":      "FuxiTranyu-8B",
    },
    "fuxi_intermediate": {
        "rankme_csv": Path("../results/fuxi.csv"),
        "merged_csv": Path("../results/fuxi_intermediate_merged.csv"),
        "layer":      "layer_29",
        "label":      "FuxiTranyu-8B (intermediate)",
    },
    "apertus": {
        "rankme_csv": Path("../results/apertus.csv"),
        "merged_csv": Path("../results/apertus_merged.csv"),
        "layer":      "layer_31",
        "label":      "Apertus-8B-2509",
    },
}

_BENCHMARKS_PATH = Path("configs/benchmarks.yaml")

# Grokking detection parameters — adjust here for ablation studies.
_GROKKING_THRESHOLD  = 0.15  # accuracy must exceed random_chance + this value
_GROKKING_MIN_CONSEC = 2     # for at least this many consecutive checkpoints


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

    plots_dir = Path("plots")
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
