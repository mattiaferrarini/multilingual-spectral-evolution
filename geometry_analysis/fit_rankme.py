"""
Fit piecewise models to RankMe scores using a declarative approach.
Models provide basis functions; an engine handles non-linear optimization + OLS.
"""

import re
import numpy as np
import pandas as pd
from typing import Protocol, Callable
from scipy.optimize import differential_evolution, lsq_linear

T_CHANGE_FIXED = 241.0  # billions of tokens (default changepoint)
T_SCALE = 1000.0       # divide B -> T before fitting for numerical stability

COLOR_ENTROPY_SEEKING = "tomato"      # phase is growing (RankMe increases)
COLOR_COMPRESSION_SEEKING = "steelblue"  # phase is declining (RankMe decreases)


def parse_checkpoint(s: str) -> float | None:
    m = re.match(r"^(\d+(?:\.\d+)?)([BMT]?)$", str(s), re.IGNORECASE)
    if m:
        v, unit = float(m.group(1)), m.group(2).upper()
        if unit == "B": return v
        if unit == "M": return v / 1_000.0
        if unit == "T": return v * 1_000.0
        return v
    m = re.match(r"step\d+-tokens(\d+(?:\.\d+)?)([BT])", str(s), re.IGNORECASE)
    if m:
        v, unit = float(m.group(1)), m.group(2).upper()
        return v * 1_000.0 if unit == "T" else v
    return None


# ---- Declarative Model Interfaces ----

class PiecewiseModel(Protocol):
    nonlinear_param_names: list[str]
    linear_param_names: list[str]
    
    def get_bounds(self, t_min: float, t_max: float, changepoints: list[float]) -> list[tuple[float, float]]:
        ...
        
    def evaluate_basis(self, t: np.ndarray, nonlinear_params: list[float], changepoints: list[float]) -> list[np.ndarray]:
        ...
        
    def validate_params(self, nonlinear_params: list[float], changepoints: list[float]) -> bool:
        ...

    def format_params(self, row) -> str:
        ...


# ---- 2-Phase Models ----

class TwoPhaseSharedAC:
    nonlinear_param_names = ["A", "gamma", "C", "lam"]
    linear_param_names = ["alpha", "beta"]

    def get_bounds(self, t_min: float, t_max: float, changepoints: list[float]) -> list[tuple[float, float]]:
        lam_max = max(t_max - changepoints[0], 0.1)
        return [
            (1e-6, 1e10),     # A
            (0.01, 5.0),      # gamma
            (-1e10, 1e10),    # C
            (0.01, lam_max),  # lam
        ]

    def validate_params(self, nonlinear_params: list[float], changepoints: list[float]) -> bool:
        A, gamma, C, lam = nonlinear_params
        return A > 0.0 and gamma > 0.0 and lam > 0.0

    def evaluate_basis(self, t: np.ndarray, nonlinear_params: list[float], changepoints: list[float]) -> list[np.ndarray]:
        A, gamma, C, lam = nonlinear_params
        t_change = changepoints[0]
        plateau = A * t_change ** (-gamma)
        dt = np.maximum(t - t_change, 0.0)
        f = np.where(
            t <= t_change,
            A * np.power(t, -gamma),
            plateau + C * (1.0 - np.exp(-dt / lam)),
        )
        return [np.ones_like(t), f]  # alpha + beta * f

    def format_params(self, row) -> str:
        return f"A={row['A']:.3g}  γ={row['gamma']:.3g}  C={row['C']:.3g}  λ={row['lam']:.3g}B  t1={row['t_change']:.3g}B"


class TwoPhasePerLangAC:
    nonlinear_param_names = ["gamma", "lam"]
    linear_param_names = ["alpha", "A", "C"]

    def get_bounds(self, t_min: float, t_max: float, changepoints: list[float]) -> list[tuple[float, float]]:
        lam_max = max(t_max - changepoints[0], 0.1)
        return [
            (0.01, 5.0),      # gamma
            (0.01, lam_max),  # lam
        ]

    def validate_params(self, nonlinear_params: list[float], changepoints: list[float]) -> bool:
        gamma, lam = nonlinear_params
        return gamma > 0.0 and lam > 0.0

    def evaluate_basis(self, t: np.ndarray, nonlinear_params: list[float], changepoints: list[float]) -> list[np.ndarray]:
        gamma, lam = nonlinear_params
        t_change = changepoints[0]
        phase1 = np.power(t, -gamma)
        dt = np.maximum(t - t_change, 0.0)
        phase2 = np.where(t <= t_change, 0.0, 1.0 - np.exp(-dt / lam))
        return [np.ones_like(t), phase1, phase2]  # alpha + A * phase1 + C * phase2

    def format_params(self, row) -> str:
        return f"γ={row['gamma']:.3g}  λ={row['lam']:.3g}B  t1={row['t_change']:.3g}B"


# ---- 3-Phase Models ----

class ThreePhaseSharedAC:
    nonlinear_param_names = ["A", "gamma", "C2", "lam2", "C3", "lam3"]
    linear_param_names = ["alpha", "beta"]

    def get_bounds(self, t_min: float, t_max: float, changepoints: list[float]) -> list[tuple[float, float]]:
        t1, t2 = changepoints
        lam2_max = max(t2 - t1, 0.1)
        lam3_max = max(t_max - t2, 0.1)
        return [
            (1e-6, 1e10),       # A
            (0.01, 5.0),        # gamma
            (-1e10, 1e10),      # C2
            (0.01, lam2_max),   # lam2
            (-1e10, 1e10),      # C3
            (0.01, lam3_max),   # lam3
        ]

    def validate_params(self, nonlinear_params: list[float], changepoints: list[float]) -> bool:
        A, gamma, C2, lam2, C3, lam3 = nonlinear_params
        t1, t2 = changepoints
        return A > 0.0 and gamma > 0.0 and lam2 > 0.0 and lam3 > 0.0 and t1 < t2

    def evaluate_basis(self, t: np.ndarray, nonlinear_params: list[float], changepoints: list[float]) -> list[np.ndarray]:
        A, gamma, C2, lam2, C3, lam3 = nonlinear_params
        t1, t2 = changepoints
        plateau1 = A * t1 ** (-gamma)
        dt2 = np.maximum(t - t1, 0.0)
        dt3 = np.maximum(t - t2, 0.0)
        phase2_val = plateau1 + C2 * (1.0 - np.exp(-dt2 / lam2))
        plateau2 = plateau1 + C2 * (1.0 - np.exp(-(t2 - t1) / lam2))
        phase3_val = plateau2 + C3 * (1.0 - np.exp(-dt3 / lam3))
        f = np.where(t <= t1, A * np.power(t, -gamma),
                     np.where(t <= t2, phase2_val, phase3_val))
        return [np.ones_like(t), f]  # alpha + beta * f

    def format_params(self, row) -> str:
        return f"A={row['A']:.3g}  γ={row['gamma']:.3g}  C2={row['C2']:.3g}  λ2={row['lam2']:.3g}B  C3={row['C3']:.3g}  λ3={row['lam3']:.3g}B  t1={row['t_change']:.3g}B  t2={row['t_change2']:.3g}B"


class ThreePhasePerLangAC:
    nonlinear_param_names = ["gamma", "lam2", "lam3"]
    linear_param_names = ["alpha", "A", "C2", "C3"]

    def get_bounds(self, t_min: float, t_max: float, changepoints: list[float]) -> list[tuple[float, float]]:
        t1, t2 = changepoints
        lam2_max = max(t2 - t1, 0.1)
        lam3_max = max(t_max - t2, 0.1)
        return [
            (0.01, 5.0),        # gamma
            (0.01, lam2_max),   # lam2
            (0.01, lam3_max),   # lam3
        ]

    def validate_params(self, nonlinear_params: list[float], changepoints: list[float]) -> bool:
        gamma, lam2, lam3 = nonlinear_params
        t1, t2 = changepoints
        return gamma > 0.0 and lam2 > 0.0 and lam3 > 0.0 and t1 < t2

    def evaluate_basis(self, t: np.ndarray, nonlinear_params: list[float], changepoints: list[float]) -> list[np.ndarray]:
        gamma, lam2, lam3 = nonlinear_params
        t1, t2 = changepoints
        phase1_b = np.power(t, -gamma)
        dt2 = np.maximum(t - t1, 0.0)
        dt3 = np.maximum(t - t2, 0.0)
        phase2_b = np.where(t <= t1, 0.0, 1.0 - np.exp(-dt2 / lam2))
        phase3_b = np.where(t <= t2, 0.0, 1.0 - np.exp(-dt3 / lam3))
        return [np.ones_like(t), phase1_b, phase2_b, phase3_b]

    def format_params(self, row) -> str:
        return f"γ={row['gamma']:.3g}  λ2={row['lam2']:.3g}B  λ3={row['lam3']:.3g}B  t1={row['t_change']:.3g}B  t2={row['t_change2']:.3g}B"


class ThreePhaseDualTailAC:
    nonlinear_param_names = ["gamma", "lam2", "lam3"]
    linear_param_names = ["alpha", "A", "C2", "C3_decay", "C3_growth"]
    linear_bounds = [
        (-np.inf, np.inf),  # alpha
        (-np.inf, np.inf),  # A
        (-np.inf, np.inf),  # C2
        (-np.inf, 0.0),     # C3_decay
        (0.0, np.inf),      # C3_growth
    ]
    # Indices 3 (C3_decay) and 4 (C3_growth) are mutually exclusive: at most one
    # should be non-zero per language.  fit_engine uses this to do per-language
    # branch selection after the global DE pass.
    linear_exclusive_groups = [[3, 4]]

    def get_bounds(self, t_min: float, t_max: float, changepoints: list[float]) -> list[tuple[float, float]]:
        t1, t2 = changepoints
        lam2_max = max(t2 - t1, 0.1)
        lam3_max = max(t_max - t2, 0.1)
        return [
            (0.01, 5.0),        # gamma
            (0.01, lam2_max),   # lam2
            (0.01, lam3_max),   # lam3
        ]

    def validate_params(self, nonlinear_params: list[float], changepoints: list[float]) -> bool:
        gamma, lam2, lam3 = nonlinear_params
        t1, t2 = changepoints
        return gamma > 0.0 and lam2 > 0.0 and lam3 > 0.0 and t1 < t2

    def evaluate_basis(self, t: np.ndarray, nonlinear_params: list[float], changepoints: list[float]) -> list[np.ndarray]:
        gamma, lam2, lam3 = nonlinear_params
        t1, t2 = changepoints
        phase1_b = np.power(t, -gamma)
        dt2 = np.maximum(t - t1, 0.0)
        dt3 = np.maximum(t - t2, 0.0)
        phase2_b = np.where(t <= t1, 0.0, 1.0 - np.exp(-dt2 / lam2))
        phase3_decay = np.where(t <= t2, 0.0, 1.0 - np.exp(-dt3 / lam3))
        with np.errstate(over='ignore'):
            phase3_growth = np.where(t <= t2, 0.0, np.exp(dt3 / lam3) - 1.0)
        return [np.ones_like(t), phase1_b, phase2_b, phase3_decay, phase3_growth]

    def format_params(self, row) -> str:
        return f"γ={row['gamma']:.3g}  λ2={row['lam2']:.3g}B  λ3={row['lam3']:.3g}B  t1={row['t_change']:.3g}B  t2={row['t_change2']:.3g}B"


class ThreePhaseDualLamAC:
    """Like ThreePhaseDualTailAC but with separate lam3_decay / lam3_growth so each
    regime has its own characteristic timescale.  Branch exclusivity is still enforced
    per language during fitting."""
    nonlinear_param_names = ["gamma", "lam2", "lam3_decay", "lam3_growth"]
    linear_param_names = ["alpha", "A", "C2", "C3_decay", "C3_growth"]
    linear_bounds = [
        (-np.inf, np.inf),  # alpha
        (-np.inf, np.inf),  # A
        (-np.inf, np.inf),  # C2
        (-np.inf, 0.0),     # C3_decay
        (0.0, np.inf),      # C3_growth
    ]
    linear_exclusive_groups = [[3, 4]]

    def get_bounds(self, t_min: float, t_max: float, changepoints: list[float]) -> list[tuple[float, float]]:
        t1, t2 = changepoints
        lam2_max = max(t2 - t1, 0.1)
        lam3_max = max(t_max - t2, 0.1)
        return [
            (0.01, 5.0),        # gamma
            (0.01, lam2_max),   # lam2
            (0.01, lam3_max),   # lam3_decay
            (0.01, lam3_max),   # lam3_growth
        ]

    def validate_params(self, nonlinear_params: list[float], changepoints: list[float]) -> bool:
        gamma, lam2, lam3_decay, lam3_growth = nonlinear_params
        t1, t2 = changepoints
        return gamma > 0.0 and lam2 > 0.0 and lam3_decay > 0.0 and lam3_growth > 0.0 and t1 < t2

    def evaluate_basis(self, t: np.ndarray, nonlinear_params: list[float], changepoints: list[float]) -> list[np.ndarray]:
        gamma, lam2, lam3_decay, lam3_growth = nonlinear_params
        t1, t2 = changepoints
        phase1_b = np.power(t, -gamma)
        dt2 = np.maximum(t - t1, 0.0)
        dt3 = np.maximum(t - t2, 0.0)
        phase2_b = np.where(t <= t1, 0.0, 1.0 - np.exp(-dt2 / lam2))
        phase3_decay = np.where(t <= t2, 0.0, 1.0 - np.exp(-dt3 / lam3_decay))
        with np.errstate(over='ignore'):
            phase3_growth = np.where(t <= t2, 0.0, np.exp(dt3 / lam3_growth) - 1.0)
        return [np.ones_like(t), phase1_b, phase2_b, phase3_decay, phase3_growth]

    def format_params(self, row) -> str:
        return f"γ={row['gamma']:.3g}  λ2={row['lam2']:.3g}B  λ3d={row['lam3_decay']:.3g}B  λ3g={row['lam3_growth']:.3g}B  t1={row['t_change']:.3g}B  t2={row['t_change2']:.3g}B"


# ---- Fitting Engine ----

def _unpack_params(x: np.ndarray, fixed_changepoints: list[float | None]) -> tuple[list[float], list[float]]:
    x_list = list(x)
    cps = []
    for fixed_cp in fixed_changepoints:
        if fixed_cp is not None:
            cps.append(fixed_cp)
        else:
            cps.append(x_list.pop())
    return x_list, cps


def _fit_coeffs(
    X: np.ndarray,
    R: np.ndarray,
    lb: list[float] | None,
    ub: list[float] | None,
    exclusive: list[list[int]] | None,
) -> np.ndarray:
    if lb is None:
        coeffs, _, _, _ = np.linalg.lstsq(X, R, rcond=None)
        return coeffs
    coeffs = lsq_linear(X, R, bounds=(lb, ub)).x
    if exclusive:
        for group in exclusive:
            if sum(1 for i in group if abs(coeffs[i]) > 1e-10) > 1:
                best_sse = np.inf
                best_coeffs = coeffs
                zero_set = set(group)
                for keep_i in group:
                    keep_cols = [c for c in range(X.shape[1]) if c not in zero_set or c == keep_i]
                    X_sel = X[:, keep_cols]
                    c_red = lsq_linear(X_sel, R, bounds=([lb[c] for c in keep_cols], [ub[c] for c in keep_cols])).x
                    c_sel = np.zeros(len(coeffs))
                    for idx, c in enumerate(keep_cols):
                        c_sel[c] = c_red[idx]
                    res = R - X @ c_sel
                    sse = float(np.dot(res, res))
                    if sse < best_sse:
                        best_sse = sse
                        best_coeffs = c_sel
                coeffs = best_coeffs
    return coeffs


def _build_bounds(
    model: PiecewiseModel,
    t_min: float,
    t_max: float,
    fixed_changepoints: list[float | None],
) -> list[tuple[float, float]]:
    n = len(fixed_changepoints)
    dummy_cps = [cp if cp is not None else t_min + (t_max - t_min) * (i + 1) / (n + 1)
                 for i, cp in enumerate(fixed_changepoints)]
    bounds = model.get_bounds(t_min, t_max, dummy_cps)
    # 1e-4 T is 100M tokens — small enough margin for both T and B scales.
    margin = 1e-4
    for cp in fixed_changepoints:
        if cp is None:
            bounds.append((t_min + margin, t_max - margin))
    return bounds


def _total_sse(
    x: np.ndarray,
    model: PiecewiseModel,
    t_by_lang: dict[str, np.ndarray],
    R_by_lang: dict[str, np.ndarray],
    fixed_changepoints: list[float | None],
    lb: list[float] | None,
    ub: list[float] | None,
    exclusive: list[list[int]] | None,
) -> float:
    nonlinear, cps = _unpack_params(x, fixed_changepoints)
    if not model.validate_params(nonlinear, cps):
        return 1e18
    sse = 0.0
    for lang in t_by_lang:
        basis = model.evaluate_basis(t_by_lang[lang], nonlinear, cps)
        if any(not np.all(np.isfinite(b)) or np.max(np.abs(b)) > 1e15 for b in basis):
            return 1e18
        X = np.column_stack(basis)
        coeffs = _fit_coeffs(X, R_by_lang[lang], lb, ub, exclusive)
        residuals = R_by_lang[lang] - X @ coeffs
        sse += float(np.dot(residuals, residuals))
    return sse


def _collect_lang_results(
    model: PiecewiseModel,
    t_by_lang: dict[str, np.ndarray],
    R_by_lang: dict[str, np.ndarray],
    nonlinear: list[float],
    cps: list[float],
    lb: list[float] | None,
    ub: list[float] | None,
    exclusive: list[list[int]] | None,
) -> dict[str, dict]:
    lang_results = {}
    for lang in t_by_lang:
        basis = model.evaluate_basis(t_by_lang[lang], nonlinear, cps)
        X = np.column_stack(basis)
        R = R_by_lang[lang]
        coeffs = _fit_coeffs(X, R, lb, ub, exclusive)
        residuals = R - X @ coeffs
        sse = float(np.dot(residuals, residuals))
        ss_tot = float(np.sum((R - R.mean()) ** 2))
        r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
        lang_results[lang] = {
            "linear_params": dict(zip(model.linear_param_names, coeffs)),
            "sse": sse,
            "r2": r2,
        }
    return lang_results


def fit_engine(
    model: PiecewiseModel,
    t_by_lang: dict[str, np.ndarray],
    R_by_lang: dict[str, np.ndarray],
    fixed_changepoints: list[float | None],
    t_min: float,
    t_max: float,
    seed: int = 0
):
    lb = [b[0] for b in model.linear_bounds] if hasattr(model, 'linear_bounds') and model.linear_bounds is not None else None
    ub = [b[1] for b in model.linear_bounds] if hasattr(model, 'linear_bounds') and model.linear_bounds is not None else None
    exclusive = getattr(model, 'linear_exclusive_groups', None) if lb is not None else None

    bounds = _build_bounds(model, t_min, t_max, fixed_changepoints)
    result = differential_evolution(
        _total_sse, bounds=bounds, seed=seed,
        args=(model, t_by_lang, R_by_lang, fixed_changepoints, lb, ub, exclusive),
        maxiter=2000, tol=1e-10, mutation=(0.5, 1.5), recombination=0.7,
        popsize=20, polish=True, workers=1,
    )

    best_nonlinear, best_cps = _unpack_params(result.x, fixed_changepoints)
    lang_results = _collect_lang_results(model, t_by_lang, R_by_lang, best_nonlinear, best_cps, lb, ub, exclusive)
    return best_nonlinear, best_cps, lang_results


# ---- Public notebook API ----

def fit_rankme_from_df(
    df: pd.DataFrame,
    layer: str,
    aggregation: str,
    model: PiecewiseModel,
    changepoints: list[float | None],
    metric: str = "rankme",
    seed: int = 0,
    t_scale: float = T_SCALE,
) -> pd.DataFrame:
    work = df.copy()
    work["t"] = work["checkpoint"].apply(parse_checkpoint)
    work = work.dropna(subset=["t", metric])

    mask = (work["layer"] == layer) & (work["aggregation"] == aggregation)
    subset = work[mask].copy()
    if subset.empty:
        raise ValueError(f"No data for layer={layer!r}, aggregation={aggregation!r}")

    languages = sorted(subset["dataset"].unique())
    t_by_lang: dict[str, np.ndarray] = {}
    R_by_lang: dict[str, np.ndarray] = {}
    for lang in languages:
        sub = subset[subset["dataset"] == lang].sort_values("t")
        t_by_lang[lang] = sub["t"].values.astype(float) / t_scale
        R_by_lang[lang] = sub[metric].values.astype(float)

    t_all = np.concatenate(list(t_by_lang.values()))
    t_min, t_max = t_all.min(), t_all.max()

    fixed_cps = [cp / t_scale if cp is not None else None for cp in changepoints]

    best_nonlinear, best_cps, lang_results = fit_engine(
        model, t_by_lang, R_by_lang, fixed_cps, t_min, t_max, seed
    )

    rows = []
    for lang in languages:
        res = lang_results[lang]
        row = {
            "layer": layer,
            "aggregation": aggregation,
            "language": lang,
            "r2": res["r2"],
            "sse": res["sse"]
        }
        
        nl_dict = dict(zip(model.nonlinear_param_names, best_nonlinear))
        for name, val in nl_dict.items():
            if name.startswith("lam"):
                row[name] = val * t_scale
            elif name == "A":
                row[name] = val * (t_scale ** nl_dict.get("gamma", 0))
            else:
                row[name] = val

        # Unpack linear params and scale back A if it was a linear param
        lin_dict = res["linear_params"]
        row.update(lin_dict)
        if "A" in lin_dict and "gamma" in nl_dict:
            row["A"] = lin_dict["A"] * (t_scale ** nl_dict["gamma"])

        for i, cp in enumerate(best_cps):
            key = "t_change" if i == 0 else f"t_change{i + 1}"
            row[key] = cp * t_scale

        rows.append(row)

    return pd.DataFrame(rows)


def plot_fitted_laws(
    params_df: pd.DataFrame,
    df: pd.DataFrame,
    model: PiecewiseModel,
    metric: str = "rankme",
    output_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import os
    from matplotlib.patches import Patch

    layer = params_df["layer"].iloc[0]
    aggregation = params_df["aggregation"].iloc[0]

    work = df.copy()
    work["t"] = work["checkpoint"].apply(parse_checkpoint)
    work = work.dropna(subset=["t", metric])
    subset = work[(work["layer"] == layer) & (work["aggregation"] == aggregation)]

    languages = list(params_df["language"])
    t_by_lang = {}
    R_by_lang = {}
    for lang in languages:
        sub = subset[subset["dataset"] == lang].sort_values("t")
        t_by_lang[lang] = sub["t"].values.astype(float)
        R_by_lang[lang] = sub[metric].values.astype(float)

    t_min_all = min(t.min() for t in t_by_lang.values())
    t_max_all = max(t.max() for t in t_by_lang.values())
    t_plot = np.linspace(t_min_all, t_max_all, 600)
    
    t_range = t_max_all - t_min_all
    padding = t_range * 0.05
    t_pad_start = 0.0
    t_pad_end = t_max_all + padding

    ncols = min(4, len(languages))
    nrows = (len(languages) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for i, row in enumerate(params_df.itertuples(index=False)):
        lang = row.language
        ax = axes.flat[i]
        row_dict = row._asdict()

        # Reconstruct params without t_scale since plotting operates in B tokens natively
        cps = []
        if "t_change" in row_dict: cps.append(row_dict["t_change"])
        if "t_change2" in row_dict: cps.append(row_dict["t_change2"])
        # Support any number of changepoints dynamically
        idx = 3
        while f"t_change{idx}" in row_dict:
            cps.append(row_dict[f"t_change{idx}"])
            idx += 1

        nl_params = []
        for name in model.nonlinear_param_names:
            nl_params.append(row_dict[name])

        basis = model.evaluate_basis(t_plot, nl_params, cps)
        
        R_pred = np.zeros_like(t_plot)
        for name, b in zip(model.linear_param_names, basis):
            R_pred += row_dict[name] * b

        def _r_at(tv):
            # Clamp evaluation for the background shading edges to avoid evaluating the model out of bounds
            clamped_tv = np.clip(tv, t_min_all, t_max_all)
            return float(R_pred[np.argmin(np.abs(t_plot - clamped_tv))])

        phase_spans = []
        boundaries = [t_pad_start] + cps + [t_pad_end]
        for idx in range(len(boundaries) - 1):
            x_lo, x_hi = boundaries[idx], boundaries[idx+1]
            r_lo, r_hi = _r_at(x_lo), _r_at(x_hi)
            phase_spans.append((x_lo, x_hi, r_hi > r_lo))

        for x_lo, x_hi, growing in phase_spans:
            ax.axvspan(x_lo, x_hi,
                       color=COLOR_ENTROPY_SEEKING if growing else COLOR_COMPRESSION_SEEKING,
                       alpha=0.15, zorder=0)

        ax.scatter(t_by_lang[lang], R_by_lang[lang], s=18, zorder=5, color="#333333", label="data")
        ax.plot(t_plot, R_pred, color="crimson", linewidth=1.5, label="fit")
        for cp in cps:
            ax.axvline(cp, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{lang}  R²={row.r2:.3f}", fontsize=9)
        ax.set_xlabel("Tokens (B)", fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_xlim(t_pad_start, t_pad_end)
        ax.tick_params(labelsize=7)

    for j in range(len(languages), len(axes.flat)):
        axes.flat[j].set_visible(False)

    r0 = params_df.iloc[0]
    param_str = model.format_params(r0)
    model_tag = model.__class__.__name__
            
    fig.suptitle(f"{layer} / {aggregation} / {model_tag}\n{param_str}", fontsize=10)
    fig.legend(
        handles=[
            Patch(facecolor=COLOR_ENTROPY_SEEKING, alpha=0.4, label="Entropy-seeking (growth)"),
            Patch(facecolor=COLOR_COMPRESSION_SEEKING, alpha=0.4, label="Compression-seeking (decline)"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {output_path}")
    else:
        plt.show()

