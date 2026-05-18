"""
Clustering utilities for language layer-profile analysis.

Profiles are loaded from the JSON files produced by visualize.py and compared
using Euclidean distance on raw (un-normalized) mean profiles so that both
scale differences (e.g. Finnish RankMe ~600 vs Swahili ~100) and shape
differences contribute to the distance — i.e. profiles that visually overlay
well are close, profiles that differ in amplitude or shape are far apart.
"""

import glob
import json

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering


def load_layer_profiles(base_dir, experiment, metric, agg):
    """
    Load all layer-profile JSON files for a given (metric, aggregation) pair.

    Returns
    -------
    dict  { language_name -> full JSON dict }
    """
    pattern = (
        f"{base_dir}/{experiment}/layer_profiles/{metric}/"
        f"layer_profile_{metric}_*_{agg}.json"
    )
    profiles = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        profiles[data["language"]] = data
    return profiles


def mean_profile(lang_data):
    """
    Average a language's layer values across all checkpoints.

    Parameters
    ----------
    lang_data : dict   One entry from load_layer_profiles()

    Returns
    -------
    np.ndarray  shape (n_layers,)
    """
    arrays = np.array(list(lang_data["checkpoint_data"].values()), dtype=float)
    return np.nanmean(arrays, axis=0)


def compute_distance_matrix(profiles_data):
    """
    Compute the pairwise Euclidean distance matrix between languages.

    Distance is measured on the raw mean profiles (no normalization) so that
    both scale and shape contribute equally — languages that overlay on a
    shared y-axis are close; those that differ in magnitude or curve shape
    are far apart.

    Parameters
    ----------
    profiles_data : dict   Output of load_layer_profiles()

    Returns
    -------
    languages_list : list[str]
    dist_matrix    : np.ndarray  shape (n, n), symmetric, zero diagonal
    """
    languages_list = sorted(profiles_data.keys())
    n = len(languages_list)
    dist_matrix = np.zeros((n, n))

    profiles = {lang: mean_profile(profiles_data[lang]) for lang in languages_list}

    for i, lang1 in enumerate(languages_list):
        for j, lang2 in enumerate(languages_list):
            if i == j:
                continue
            p1 = np.nan_to_num(profiles[lang1])
            p2 = np.nan_to_num(profiles[lang2])
            dist_matrix[i, j] = euclidean(p1, p2)

    return languages_list, dist_matrix


def cluster_languages(dist_matrix, k=5, linkage="ward"):
    """
    Agglomerative clustering on a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray  shape (n, n)
    k           : int         number of clusters
    linkage     : str         Ward's method by default

    Returns
    -------
    np.ndarray  shape (n,) — cluster label per language
    """
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric="euclidean")
    return model.fit_predict(dist_matrix)
