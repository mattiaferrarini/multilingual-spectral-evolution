"""
Fit the piecewise model to RankMe scores for a fixed (layer, aggregation) pair.

2-phase Model (Fuxi):
    R_ℓ(t) = α_ℓ + β_ℓ · f(t)

    f(t) = A · t^{-γ}                                  for t ≤ t_change
    Phase 2 (t > t_change):
      A · t_change^{-γ} + C · (1 - exp(-(t - t_change)/λ))

Shared parameters: A, γ, C, λ [, t_change if --estimate-changepoint].
Per-language parameters: α_ℓ, β_ℓ  (profiled out analytically via OLS).

Alternative 2-phase Model (with --per-language-ac):
    R_ℓ(t) = α_ℓ + A_ℓ · t^{-γ} + C_ℓ · (1 − exp(−(t − t_change)/λ))
    where γ, λ are shared, and α_ℓ, A_ℓ, C_ℓ are per-language.

3-phase Model (Apertus, --three-phase):
    Phase 1 (t ≤ t1):        A · t^{-γ}
    Phase 2 (t1 < t ≤ t2):   plateau1 + C2 · (1 − exp(−(t−t1)/λ2))
    Phase 3 (t > t2):        plateau2 + C3 · (1 − exp(−(t−t2)/λ3))

Alternative 3-phase Model (with --per-language-ac):
    R_ℓ(t) = α_ℓ + A_ℓ · t^{-γ} + C2_ℓ · p2(t) + C3_ℓ · p3(t)
    where γ, λ2, λ3 are shared, and α_ℓ, A_ℓ, C2_ℓ, C3_ℓ are per-language.

t is measured in billions of tokens throughout.
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

T_CHANGE_FIXED = 241.0  # billions of tokens (default changepoint)
T_SCALE = 1000.0       # divide B → T before fitting for numerical stability

COLOR_ENTROPY_SEEKING = "tomato"      # phase is growing (RankMe increases)
COLOR_COMPRESSION_SEEKING = "steelblue"  # phase is declining (RankMe decreases)


def parse_checkpoint(s: str) -> float | None:
    # plain numeric: "210B", "10.5B", "500M"
    m = re.match(r"^(\d+(?:\.\d+)?)([BMT]?)$", str(s), re.IGNORECASE)
    if m:
        v, unit = float(m.group(1)), m.group(2).upper()
        if unit == "B": return v
        if unit == "M": return v / 1_000.0
        if unit == "T": return v * 1_000.0
        return v
    # Apertus format: step{N}-tokens{M}[BT]
    m = re.match(r"step\d+-tokens(\d+(?:\.\d+)?)([BT])", str(s), re.IGNORECASE)
    if m:
        v, unit = float(m.group(1)), m.group(2).upper()
        return v * 1_000.0 if unit == "T" else v
    return None


# ---- Model 1: shared A, γ, C, λ (and optionally t_change) ----

def compute_f(t: np.ndarray, A: float, gamma: float, C: float, lam: float, t_change: float) -> np.ndarray:
    """Phase 2: A·t_change^{-γ} + C·(1 − exp(−(t−t_change)/λ)).
    Continuous at t_change by construction."""
    plateau = A * t_change ** (-gamma)
    dt = np.maximum(t - t_change, 0.0)
    return np.where(
        t <= t_change,
        A * np.power(t, -gamma),
        plateau + C * (1.0 - np.exp(-dt / lam)),
    )

def _apply_f(t: np.ndarray, shared: tuple, t_change: float) -> np.ndarray:
    """Unpacks shared params and computes f."""
    A, gamma, C, lam = shared
    return compute_f(t, A, gamma, C, lam, t_change)


def fit_per_language(f_vals: np.ndarray, R_vals: np.ndarray) -> tuple[float, float, float]:
    """OLS: R = α + β·f.  Returns (α, β, SSE)."""
    X = np.column_stack([np.ones_like(f_vals), f_vals])
    coeffs, _, _, _ = np.linalg.lstsq(X, R_vals, rcond=None)
    alpha, beta = coeffs
    residuals = R_vals - (alpha + beta * f_vals)
    return alpha, beta, float(np.dot(residuals, residuals))


# ---- Model 2: per-language A_ℓ, C_ℓ with shared γ, λ (and optionally t_change) ----

def fit_per_language_ac(t_vals: np.ndarray, R_vals: np.ndarray, gamma: float, lam: float, t_change: float) -> tuple[float, float, float, float]:
    """OLS: R = α + A·t^{-γ} + C·phase2_basis(t). Returns (α, A, C, SSE)."""
    phase1_basis = np.power(t_vals, -gamma)
    dt = np.maximum(t_vals - t_change, 0.0)
    phase2_basis = np.where(t_vals <= t_change, 0.0, 1.0 - np.exp(-dt / lam))
    
    X = np.column_stack([np.ones_like(t_vals), phase1_basis, phase2_basis])
    coeffs, _, _, _ = np.linalg.lstsq(X, R_vals, rcond=None)
    alpha, A, C = coeffs
    residuals = R_vals - X @ coeffs
    return alpha, A, C, float(np.dot(residuals, residuals))


# ---- 3-phase Model (for models with two changepoints, e.g. Apertus) ----

def compute_f3(
    t: np.ndarray,
    A: float, gamma: float,
    C2: float, lam2: float,
    C3: float, lam3: float,
    t1: float, t2: float,
) -> np.ndarray:
    """3-phase piecewise, continuous at both changepoints.

    Phase 1 (t ≤ t1):       A · t^{-γ}
    Phase 2 (t1 < t ≤ t2):  plateau1 + C2 · (1 − exp(−(t−t1)/λ2))
    Phase 3 (t > t2):       plateau2 + C3 · (1 − exp(−(t−t2)/λ3))
    """
    plateau1 = A * t1 ** (-gamma)
    dt2 = np.maximum(t - t1, 0.0)
    dt3 = np.maximum(t - t2, 0.0)
    phase2_val = plateau1 + C2 * (1.0 - np.exp(-dt2 / lam2))
    plateau2 = plateau1 + C2 * (1.0 - np.exp(-(t2 - t1) / lam2))
    phase3_val = plateau2 + C3 * (1.0 - np.exp(-dt3 / lam3))
    return np.where(t <= t1, A * np.power(t, -gamma),
                    np.where(t <= t2, phase2_val, phase3_val))


def fit_per_language_ac3(
    t_vals: np.ndarray, R_vals: np.ndarray,
    gamma: float, lam2: float, lam3: float,
    t1: float, t2: float,
) -> tuple[float, float, float, float, float]:
    """OLS: R = α + A·t^{-γ} + C2·p2(t) + C3·p3(t).  Returns (α, A, C2, C3, SSE)."""
    phase1_b = np.power(t_vals, -gamma)
    dt2 = np.maximum(t_vals - t1, 0.0)
    dt3 = np.maximum(t_vals - t2, 0.0)
    phase2_b = np.where(t_vals <= t1, 0.0, 1.0 - np.exp(-dt2 / lam2))
    phase3_b = np.where(t_vals <= t2, 0.0, 1.0 - np.exp(-dt3 / lam3))
    X = np.column_stack([np.ones_like(t_vals), phase1_b, phase2_b, phase3_b])
    coeffs, _, _, _ = np.linalg.lstsq(X, R_vals, rcond=None)
    alpha, A, C2, C3 = coeffs
    residuals = R_vals - X @ coeffs
    return alpha, A, C2, C3, float(np.dot(residuals, residuals))


# ---- Optimization objective ----

def total_sse(shared_params, t_by_lang, R_by_lang, fix_t_change: bool, per_language_ac: bool, t_change_fixed: float = T_CHANGE_FIXED) -> float:
    if fix_t_change:
        core, t_change = shared_params, t_change_fixed
    else:
        *core, t_change = shared_params
        core = tuple(core)

    if per_language_ac:
        gamma, lam = core
        if gamma <= 0.0 or lam <= 0.0:
            return 1e18
            
        sse = 0.0
        for lang in t_by_lang:
            _, _, _, lang_sse = fit_per_language_ac(t_by_lang[lang], R_by_lang[lang], gamma, lam, t_change)
            sse += lang_sse
        return sse
    else:
        A, gamma, C, lam = core
        if A <= 0.0 or gamma <= 0.0 or lam <= 0.0:
            return 1e18

        sse = 0.0
        for lang in t_by_lang:
            f_vals = _apply_f(t_by_lang[lang], core, t_change)
            if not np.all(np.isfinite(f_vals)):
                return 1e18
            _, _, lang_sse = fit_per_language(f_vals, R_by_lang[lang])
            sse += lang_sse
        return sse


# ---- Fitting procedure ----

def fit_model(
    t_by_lang: dict,
    R_by_lang: dict,
    fix_t_change: bool,
    per_language_ac: bool,
    seed: int = 0,
    t_change_value: float | None = None,
) -> tuple:
    """
    Optimise shared parameters via differential evolution (global), then refine.
    Returns (core_params_tuple, t_change, result).

    t_change_value: explicit fixed changepoint (B tokens) when fix_t_change=True.
        Defaults to T_CHANGE_FIXED when None.
    """
    t_all = np.concatenate(list(t_by_lang.values()))
    t_min, t_max = t_all.min(), t_all.max()

    _t_change_fixed = t_change_value if (fix_t_change and t_change_value is not None) else T_CHANGE_FIXED

    # Phase-2 span upper bound for λ: prevents exponential from degenerating to linear
    lam_max = max(t_max - _t_change_fixed, 0.1)

    if per_language_ac:
        bounds_core = [
            (0.01, 5.0),      # gamma
            (0.01, lam_max),  # λ: bounded by phase-2 span
        ]
    else:
        bounds_core = [
            (1e-6, 1e10),     # A
            (0.01, 5.0),      # gamma
            (-1e10, 1e10),    # C (amplitude; sign unconstrained)
            (0.01, lam_max),  # λ: bounded by phase-2 span
        ]

    bounds = bounds_core if fix_t_change else bounds_core + [(t_min + 1.0, t_max - 1.0)]

    result = differential_evolution(
        total_sse,
        bounds=bounds,
        args=(t_by_lang, R_by_lang, fix_t_change, per_language_ac, _t_change_fixed),
        seed=seed,
        maxiter=2000,
        tol=1e-10,
        mutation=(0.5, 1.5),
        recombination=0.7,
        popsize=20,
        polish=True,
        workers=1,
    )

    if fix_t_change:
        core = tuple(result.x)
        t_change = _t_change_fixed
    else:
        *core_list, t_change = result.x
        core = tuple(core_list)

    return core, t_change, result


# ---- 3-phase optimization ----

def total_sse3(
    params: np.ndarray,
    t_by_lang: dict,
    R_by_lang: dict,
    fix_t1: bool,
    fix_t2: bool,
    per_language_ac: bool,
    t1_fixed: float,
    t2_fixed: float,
) -> float:
    p = list(params)
    if fix_t1 and fix_t2:
        core = tuple(p)
        t1, t2 = t1_fixed, t2_fixed
    elif fix_t1:
        *core_l, t2 = p
        core, t1 = tuple(core_l), t1_fixed
    elif fix_t2:
        *core_l, t1 = p
        core, t2 = tuple(core_l), t2_fixed
    else:
        *core_l, t1, t2 = p
        core = tuple(core_l)

    if t1 >= t2:
        return 1e18

    if per_language_ac:
        gamma, lam2, lam3 = core
        if gamma <= 0.0 or lam2 <= 0.0 or lam3 <= 0.0:
            return 1e18
        sse = 0.0
        for lang in t_by_lang:
            _, _, _, _, lang_sse = fit_per_language_ac3(
                t_by_lang[lang], R_by_lang[lang], gamma, lam2, lam3, t1, t2)
            sse += lang_sse
        return sse
    else:
        A, gamma, C2, lam2, C3, lam3 = core
        if A <= 0.0 or gamma <= 0.0 or lam2 <= 0.0 or lam3 <= 0.0:
            return 1e18
        sse = 0.0
        for lang in t_by_lang:
            f_vals = compute_f3(t_by_lang[lang], A, gamma, C2, lam2, C3, lam3, t1, t2)
            if not np.all(np.isfinite(f_vals)):
                return 1e18
            _, _, lang_sse = fit_per_language(f_vals, R_by_lang[lang])
            sse += lang_sse
        return sse


def fit_model3(
    t_by_lang: dict,
    R_by_lang: dict,
    fix_t1: bool,
    fix_t2: bool,
    per_language_ac: bool,
    seed: int = 0,
    t1_value: float | None = None,
    t2_value: float | None = None,
) -> tuple:
    """Optimise 3-phase shared parameters via differential evolution.
    Returns (core_params_tuple, t1, t2, result).

    fix_t1 / fix_t2: if True the corresponding changepoint is held fixed at
    t1_value / t2_value (required when fixed).  If False it is estimated.
    """
    t_all = np.concatenate(list(t_by_lang.values()))
    t_min, t_max = t_all.min(), t_all.max()

    _t1 = t1_value if (fix_t1 and t1_value is not None) else T_CHANGE_FIXED
    _t2 = t2_value if (fix_t2 and t2_value is not None) else (t_min + t_max) / 2.0

    # Phase-specific upper bounds for λ: an exponential with λ >> phase span degenerates
    # to a straight line over the observed data, making the phases unidentifiable.
    lam2_max = max(_t2 - _t1, 0.1)
    lam3_max = max(t_max - _t2, 0.1)

    if per_language_ac:
        bounds_core = [
            (0.01, 5.0),        # gamma
            (0.01, lam2_max),   # λ2: bounded by phase-2 span
            (0.01, lam3_max),   # λ3: bounded by phase-3 span
        ]
    else:
        bounds_core = [
            (1e-6, 1e10),       # A
            (0.01, 5.0),        # gamma
            (-1e10, 1e10),      # C2 (sign unconstrained)
            (0.01, lam2_max),   # λ2: bounded by phase-2 span
            (-1e10, 1e10),      # C3 (sign unconstrained)
            (0.01, lam3_max),   # λ3: bounded by phase-3 span
        ]

    bounds = list(bounds_core)
    if not fix_t1:
        bounds.append((t_min + 1.0, t_max - 1.0))
    if not fix_t2:
        bounds.append((t_min + 1.0, t_max - 1.0))

    result = differential_evolution(
        total_sse3,
        bounds=bounds,
        args=(t_by_lang, R_by_lang, fix_t1, fix_t2, per_language_ac, _t1, _t2),
        seed=seed,
        maxiter=2000,
        tol=1e-10,
        mutation=(0.5, 1.5),
        recombination=0.7,
        popsize=20,
        polish=True,
        workers=1,
    )

    p = list(result.x)
    if fix_t1 and fix_t2:
        core = tuple(p)
        t1, t2 = _t1, _t2
    elif fix_t1:
        *core_l, t2 = p
        core, t1 = tuple(core_l), _t1
    elif fix_t2:
        *core_l, t1 = p
        core, t2 = tuple(core_l), _t2
    else:
        *core_l, t1, t2 = p
        core = tuple(core_l)

    return core, t1, t2, result


# ---- Public notebook API ----

def fit_rankme_from_df(
    df: pd.DataFrame,
    layer: str,
    aggregation: str,
    t_change: float | None = None,
    t_change2: float | None = None,
    per_language_ac: bool = False,
    n_phases: int = 2,
    metric: str = "rankme",
    seed: int = 0,
    t_scale: float = T_SCALE,
) -> pd.DataFrame:
    """Fit the piecewise model to *df* and return fitted parameters as a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics CSV loaded into a DataFrame (columns: checkpoint, dataset,
        layer, aggregation, <metric>, ...).  Both Fuxi ("210B") and Apertus
        ("step50000-tokens210B") checkpoint formats are supported.
    layer : str
        Layer to fit, e.g. ``"layer_29"``.
    aggregation : str
        Aggregation method: ``"last"`` or ``"mean"``.
    t_change : float | None
        First changepoint in billions of tokens.  ``None`` → estimated from data.
    t_change2 : float | None
        Second changepoint in billions of tokens (3-phase only).
        ``None`` → estimated from data.
    per_language_ac : bool
        If True, amplitude parameters are per-language while shape parameters
        are shared.
    n_phases : int
        Number of phases: 2 (default, Fuxi-style) or 3 (Apertus-style).
    metric : str
        Response column (default: ``"rankme"``).
    seed : int
        Random seed for differential evolution.
    t_scale : float
        Divide token counts by this value before fitting for numerical
        stability.  Params are converted back to billions before return.
        Default: ``T_SCALE`` (1000.0, i.e. fit in trillions).

    Returns
    -------
    pd.DataFrame
        One row per language.  Metadata columns: layer, aggregation, language,
        t_change [, t_change2 for n_phases=3].  Parameter columns:

        * 2-phase shared:       A, gamma, C, lam, alpha, beta, r2, sse
        * 2-phase per-lang-AC:  gamma, lam, alpha, A, C, r2, sse
        * 3-phase shared:       A, gamma, C2, lam2, C3, lam3, alpha, beta, r2, sse
        * 3-phase per-lang-AC:  gamma, lam2, lam3, alpha, A, C2, C3, r2, sse
    """
    work = df.copy()
    work["t"] = work["checkpoint"].apply(parse_checkpoint)
    work = work.dropna(subset=["t", metric])

    mask = (work["layer"] == layer) & (work["aggregation"] == aggregation)
    subset = work[mask].copy()
    if subset.empty:
        raise ValueError(f"No data for layer={layer!r}, aggregation={aggregation!r}")

    languages = sorted(subset["dataset"].unique())
    # Scale t to T tokens for numerically stable fitting; R is unchanged.
    t_by_lang: dict[str, np.ndarray] = {}
    R_by_lang: dict[str, np.ndarray] = {}
    for lang in languages:
        sub = subset[subset["dataset"] == lang].sort_values("t")
        t_by_lang[lang] = sub["t"].values.astype(float) / t_scale
        R_by_lang[lang] = sub[metric].values.astype(float)

    # Changepoints in scaled tokens for the optimizer
    t1_sc = t_change / t_scale if t_change is not None else None
    t2_sc = t_change2 / t_scale if t_change2 is not None else None

    if n_phases == 3:
        fix_t1 = t_change is not None
        fix_t2 = t_change2 is not None
        core, t1_fit_sc, t2_fit_sc, _ = fit_model3(
            t_by_lang, R_by_lang,
            fix_t1=fix_t1, fix_t2=fix_t2,
            per_language_ac=per_language_ac,
            seed=seed,
            t1_value=t1_sc,
            t2_value=t2_sc,
        )
        # Convert changepoints back to B for storage
        t1_fit = t1_fit_sc * t_scale
        t2_fit = t2_fit_sc * t_scale
        rows = []
        for lang in languages:
            R_mean = R_by_lang[lang].mean()
            ss_tot = float(np.sum((R_by_lang[lang] - R_mean) ** 2))
            if per_language_ac:
                gamma, lam2_sc, lam3_sc = core
                alpha, A_sc, C2, C3, sse = fit_per_language_ac3(
                    t_by_lang[lang], R_by_lang[lang], gamma, lam2_sc, lam3_sc, t1_fit_sc, t2_fit_sc)
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
                rows.append(dict(
                    layer=layer, aggregation=aggregation, language=lang,
                    t_change=t1_fit, t_change2=t2_fit,
                    gamma=gamma, lam2=lam2_sc * t_scale, lam3=lam3_sc * t_scale,
                    alpha=alpha, A=A_sc * t_scale**gamma, C2=C2, C3=C3, r2=r2, sse=sse,
                ))
            else:
                A_sh_sc, gamma, C2_sh, lam2_sc, C3_sh, lam3_sc = core
                f_vals = compute_f3(t_by_lang[lang], A_sh_sc, gamma, C2_sh, lam2_sc, C3_sh, lam3_sc, t1_fit_sc, t2_fit_sc)
                alpha, beta, sse = fit_per_language(f_vals, R_by_lang[lang])
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
                rows.append(dict(
                    layer=layer, aggregation=aggregation, language=lang,
                    t_change=t1_fit, t_change2=t2_fit,
                    A=A_sh_sc * t_scale**gamma, gamma=gamma,
                    C2=C2_sh, lam2=lam2_sc * t_scale,
                    C3=C3_sh, lam3=lam3_sc * t_scale,
                    alpha=alpha, beta=beta, r2=r2, sse=sse,
                ))
        return pd.DataFrame(rows)

    # --- 2-phase ---
    fix_t_change = t_change is not None
    core, t_change_fit_sc, _ = fit_model(
        t_by_lang, R_by_lang,
        fix_t_change=fix_t_change,
        per_language_ac=per_language_ac,
        seed=seed,
        t_change_value=t1_sc if t1_sc is not None else T_CHANGE_FIXED / t_scale,
    )
    t_change_fit = t_change_fit_sc * t_scale  # back to B for storage

    rows = []
    for lang in languages:
        R_mean = R_by_lang[lang].mean()
        ss_tot = float(np.sum((R_by_lang[lang] - R_mean) ** 2))

        if per_language_ac:
            gamma, lam_sc = core
            alpha, A_sc, C, sse = fit_per_language_ac(
                t_by_lang[lang], R_by_lang[lang], gamma, lam_sc, t_change_fit_sc)
            r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
            rows.append(dict(
                layer=layer, aggregation=aggregation, language=lang,
                t_change=t_change_fit, gamma=gamma, lam=lam_sc * t_scale,
                alpha=alpha, A=A_sc * t_scale**gamma, C=C, r2=r2, sse=sse,
            ))
        else:
            A_sh_sc, gamma, C_sh, lam_sc = core
            f_vals = _apply_f(t_by_lang[lang], core, t_change_fit_sc)
            alpha, beta, sse = fit_per_language(f_vals, R_by_lang[lang])
            r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
            rows.append(dict(
                layer=layer, aggregation=aggregation, language=lang,
                t_change=t_change_fit, A=A_sh_sc * t_scale**gamma, gamma=gamma,
                C=C_sh, lam=lam_sc * t_scale,
                alpha=alpha, beta=beta, r2=r2, sse=sse,
            ))

    return pd.DataFrame(rows)


def plot_fitted_laws(
    params_df: pd.DataFrame,
    df: pd.DataFrame,
    metric: str = "rankme",
    output_path: str | None = None,
) -> None:
    """Plot data points and fitted curves for every language in *params_df*.

    Parameters
    ----------
    params_df : pd.DataFrame
        Output of :func:`fit_rankme_from_df`.
    df : pd.DataFrame
        Original metrics DataFrame used to produce *params_df* (needed for
        raw data points).
    metric : str
        Column used as the response variable (default: ``"rankme"``).
    output_path : str | None
        If given, save the figure to this path instead of showing it.
    """
    import matplotlib.pyplot as plt

    three_phase = "t_change2" in params_df.columns
    per_language_ac = "beta" not in params_df.columns

    layer = params_df["layer"].iloc[0]
    aggregation = params_df["aggregation"].iloc[0]
    t_change = float(params_df["t_change"].iloc[0])

    # extract shared parameters (same on every row)
    if three_phase:
        t_change2 = float(params_df["t_change2"].iloc[0])
        gamma = float(params_df["gamma"].iloc[0])
        lam2 = float(params_df["lam2"].iloc[0])
        lam3 = float(params_df["lam3"].iloc[0])
        if not per_language_ac:
            A    = float(params_df["A"].iloc[0])
            C2   = float(params_df["C2"].iloc[0])
            C3   = float(params_df["C3"].iloc[0])
    else:
        if per_language_ac:
            gamma = float(params_df["gamma"].iloc[0])
            lam   = float(params_df["lam"].iloc[0])
            core  = (gamma, lam)
        else:
            A     = float(params_df["A"].iloc[0])
            gamma = float(params_df["gamma"].iloc[0])
            C     = float(params_df["C"].iloc[0])
            lam   = float(params_df["lam"].iloc[0])
            core  = (A, gamma, C, lam)

    # raw data from original df
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

    if three_phase and not per_language_ac:
        f_plot = compute_f3(t_plot, A, gamma, C2, lam2, C3, lam3, t_change, t_change2)
    elif not three_phase and not per_language_ac:
        f_plot = _apply_f(t_plot, core, t_change)

    from matplotlib.patches import Patch

    ncols = min(4, len(languages))
    nrows = (len(languages) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for i, row in enumerate(params_df.itertuples(index=False)):
        lang = row.language
        ax = axes.flat[i]

        if three_phase:
            if per_language_ac:
                phase1_b = np.power(t_plot, -gamma)
                dt2 = np.maximum(t_plot - t_change, 0.0)
                dt3 = np.maximum(t_plot - t_change2, 0.0)
                phase2_b = np.where(t_plot <= t_change, 0.0, 1.0 - np.exp(-dt2 / lam2))
                phase3_b = np.where(t_plot <= t_change2, 0.0, 1.0 - np.exp(-dt3 / lam3))
                R_pred = row.alpha + row.A * phase1_b + row.C2 * phase2_b + row.C3 * phase3_b
            else:
                R_pred = row.alpha + row.beta * f_plot
        else:
            if per_language_ac:
                phase1 = np.power(t_plot, -gamma)
                dt = np.maximum(t_plot - t_change, 0.0)
                phase2 = np.where(t_plot <= t_change, 0.0, 1.0 - np.exp(-dt / lam))
                R_pred = row.alpha + row.A * phase1 + row.C * phase2
            else:
                R_pred = row.alpha + row.beta * f_plot

        # Per-language background shading: compare fitted R at phase boundaries
        def _r_at(tv):
            return float(R_pred[np.argmin(np.abs(t_plot - tv))])

        r0, r1 = _r_at(t_min_all), _r_at(t_change)
        if three_phase:
            r2, r3 = _r_at(t_change2), _r_at(t_max_all)
            phase_spans = [
                (t_min_all, t_change,  r1 > r0),
                (t_change,  t_change2, r2 > r1),
                (t_change2, t_max_all, r3 > r2),
            ]
        else:
            r2 = _r_at(t_max_all)
            phase_spans = [
                (t_min_all, t_change,  r1 > r0),
                (t_change,  t_max_all, r2 > r1),
            ]
        for x_lo, x_hi, growing in phase_spans:
            ax.axvspan(x_lo, x_hi,
                       color=COLOR_ENTROPY_SEEKING if growing else COLOR_COMPRESSION_SEEKING,
                       alpha=0.15, zorder=0)

        ax.scatter(t_by_lang[lang], R_by_lang[lang], s=18, zorder=5, color="#333333", label="data")
        ax.plot(t_plot, R_pred, color="crimson", linewidth=1.5, label="fit")
        ax.axvline(t_change, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        if three_phase:
            ax.axvline(t_change2, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{lang}  R²={row.r2:.3f}", fontsize=9)
        ax.set_xlabel("Tokens (B)", fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(languages), len(axes.flat)):
        axes.flat[j].set_visible(False)

    if three_phase:
        if per_language_ac:
            param_str = (f"γ={gamma:.3g}  λ2={lam2:.3g}B  λ3={lam3:.3g}B"
                         f"  t1={t_change:.3g}B  t2={t_change2:.3g}B")
        else:
            param_str = (f"A={A:.3g}  γ={gamma:.3g}  C2={C2:.3g}  λ2={lam2:.3g}B"
                         f"  C3={C3:.3g}  λ3={lam3:.3g}B  t1={t_change:.3g}B  t2={t_change2:.3g}B")
        model_tag = "3ph-per-lang-AC" if per_language_ac else "3ph-shared-AC"
    else:
        if per_language_ac:
            param_str = f"γ={gamma:.3g}  λ={lam:.3g}B  t_change={t_change:.3g}B"
        else:
            param_str = f"A={A:.3g}  γ={gamma:.3g}  C={C:.3g}  λ={lam:.3g}B  t_change={t_change:.3g}B"
        model_tag = "per-lang-AC" if per_language_ac else "shared-AC"
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


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Fit piecewise RankMe model for a given layer and aggregation."
    )
    parser.add_argument("--results", default="results/fuxi.csv", help="Path to metrics CSV.")
    parser.add_argument("--layer", required=True, help="Layer name, e.g. 'layer_15'.")
    parser.add_argument(
        "--aggregation", required=True, choices=["last", "mean"], help="Aggregation method."
    )
    parser.add_argument(
        "--estimate-changepoint",
        action="store_true",
        default=False,
        help="Estimate changepoint(s) instead of fixing them. "
             "For --three-phase, both t1 and t2 are estimated.",
    )
    parser.add_argument(
        "--per-language-ac",
        action="store_true",
        default=False,
        help="Switch to per-language amplitude parameters (A, C / A, C2, C3).",
    )
    parser.add_argument(
        "--three-phase",
        action="store_true",
        default=False,
        help="Use 3-phase model (for Apertus-style data with two changepoints).",
    )
    parser.add_argument(
        "--t-change",
        type=float,
        default=None,
        help="First changepoint in B tokens (3-phase only; overrides T_CHANGE_FIXED default).",
    )
    parser.add_argument(
        "--t-change2",
        type=float,
        default=None,
        help="Second changepoint in B tokens (3-phase only; required unless --estimate-changepoint).",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Save a fit plot to this path (e.g. fit.png). Skipped if not provided.",
    )
    args = parser.parse_args()

    if args.three_phase and not args.estimate_changepoint and args.t_change2 is None:
        parser.error("--three-phase requires --t-change2 <value> or --estimate-changepoint")

    # Load and prepare data
    df = pd.read_csv(args.results)
    df["t"] = df["checkpoint"].apply(parse_checkpoint)
    df = df.dropna(subset=["t"])

    mask = (df["layer"] == args.layer) & (df["aggregation"] == args.aggregation)
    df_sub = df[mask].copy()
    if df_sub.empty:
        print(
            f"ERROR: no data for layer={args.layer!r}, aggregation={args.aggregation!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    languages = sorted(df_sub["dataset"].unique())
    t_by_lang: dict[str, np.ndarray] = {}
    R_by_lang: dict[str, np.ndarray] = {}
    for lang in languages:
        sub = df_sub[df_sub["dataset"] == lang].sort_values("t")
        t_by_lang[lang] = sub["t"].values.astype(float) / T_SCALE  # T tokens for fitting
        R_by_lang[lang] = sub["rankme"].values.astype(float)

    three_phase = args.three_phase
    per_language_ac = args.per_language_ac
    n_phases_str = "3-phase" if three_phase else "2-phase"

    print(f"Layer={args.layer}  aggregation={args.aggregation}  {n_phases_str}  per_lang_ac={per_language_ac}")
    print(f"Languages ({len(languages)}): {', '.join(languages)}")
    print(f"Checkpoints: {len(df_sub['t'].unique())}  "
          f"(range {df_sub['t'].min():.0f}B – {df_sub['t'].max():.0f}B tokens)")
    print("Optimising…")

    per_lang_params = {}

    if three_phase:
        fix_t1 = not args.estimate_changepoint
        fix_t2 = not args.estimate_changepoint
        t1_val = args.t_change if args.t_change is not None else T_CHANGE_FIXED
        t2_val = args.t_change2
        # Scale changepoints to T tokens for the optimizer
        t1_val_sc = t1_val / T_SCALE
        t2_val_sc = t2_val / T_SCALE if t2_val is not None else None
        if fix_t1:
            print(f"  t1 fixed at {t1_val:.3g}B,  t2 fixed at {t2_val:.3g}B")
        else:
            print("  t1 and t2 estimated")

        core, t1_sc, t2_sc, result = fit_model3(
            t_by_lang, R_by_lang,
            fix_t1=fix_t1, fix_t2=fix_t2,
            per_language_ac=per_language_ac,
            t1_value=t1_val_sc if fix_t1 else None,
            t2_value=t2_val_sc if fix_t2 else None,
        )
        # Convert back to B for printing / plotting
        t1 = t1_sc * T_SCALE
        t2 = t2_sc * T_SCALE

        for lang in languages:
            R_mean = R_by_lang[lang].mean()
            ss_tot = float(np.sum((R_by_lang[lang] - R_mean) ** 2))
            if per_language_ac:
                gamma, lam2_sc, lam3_sc = core
                alpha, A_sc, C2, C3, sse = fit_per_language_ac3(
                    t_by_lang[lang], R_by_lang[lang], gamma, lam2_sc, lam3_sc, t1_sc, t2_sc)
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
                per_lang_params[lang] = dict(
                    alpha=alpha, A=A_sc * T_SCALE**gamma, C2=C2, C3=C3, sse=sse, r2=r2)
            else:
                A_sh_sc, gamma, C2_sh, lam2_sc, C3_sh, lam3_sc = core
                f_vals = compute_f3(t_by_lang[lang], A_sh_sc, gamma, C2_sh, lam2_sc, C3_sh, lam3_sc, t1_sc, t2_sc)
                alpha, beta, sse = fit_per_language(f_vals, R_by_lang[lang])
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
                per_lang_params[lang] = dict(alpha=alpha, beta=beta, sse=sse, r2=r2)

        print(f"\n{'─'*65}")
        if per_language_ac:
            gamma, lam2_sc, lam3_sc = core
            print("Shared parameters  (3-phase, per-language-ac=True)")
            print(f"{'─'*65}")
            print(f"  γ         = {gamma:.6g}")
            print(f"  λ2        = {lam2_sc * T_SCALE:.6g} B tokens")
            print(f"  λ3        = {lam3_sc * T_SCALE:.6g} B tokens")
        else:
            A_sh_sc, gamma, C2_sh, lam2_sc, C3_sh, lam3_sc = core
            print("Shared parameters  (3-phase, shared-AC)")
            print(f"{'─'*65}")
            print(f"  A         = {A_sh_sc * T_SCALE**gamma:.6g}")
            print(f"  γ         = {gamma:.6g}")
            print(f"  C2        = {C2_sh:.6g}  (phase-2 amplitude)")
            print(f"  λ2        = {lam2_sc * T_SCALE:.6g} B tokens")
            print(f"  C3        = {C3_sh:.6g}  (phase-3 amplitude)")
            print(f"  λ3        = {lam3_sc * T_SCALE:.6g} B tokens")
        print(f"  t1        = {t1:.6g} B tokens")
        print(f"  t2        = {t2:.6g} B tokens")
        print(f"  Optimiser : success={result.success},  total SSE={result.fun:.6g}")

        print(f"\n{'─'*85}")
        if per_language_ac:
            print(f"{'Language':<16} {'α':>12} {'A':>12} {'C2':>12} {'C3':>12} {'R²':>8}")
            print(f"{'─'*85}")
            for lang in languages:
                p = per_lang_params[lang]
                print(f"{lang:<16} {p['alpha']:>12.4f} {p['A']:>12.4e} {p['C2']:>12.4e} {p['C3']:>12.4e} {p['r2']:>8.4f}")
        else:
            print(f"{'Language':<16} {'α':>12} {'β':>14} {'R²':>8}")
            print(f"{'─'*55}")
            for lang in languages:
                p = per_lang_params[lang]
                print(f"{lang:<16} {p['alpha']:>12.4f} {p['beta']:>14.4e} {p['r2']:>8.4f}")

        # Optional plot (3-phase); build t_plot in B using B-unit params
        if args.output_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # t_by_lang is in T; rebuild plot range in B
            t_min_all = min(t.min() for t in t_by_lang.values()) * T_SCALE
            t_max_all = max(t.max() for t in t_by_lang.values()) * T_SCALE
            t_plot = np.linspace(t_min_all, t_max_all, 600)  # B

            if not per_language_ac:
                A_sh_sc, gamma, C2_sh, lam2_sc, C3_sh, lam3_sc = core
                # Use B-unit params with B-unit t_plot (mathematically equivalent)
                f_plot = compute_f3(t_plot, A_sh_sc * T_SCALE**gamma, gamma,
                                    C2_sh, lam2_sc * T_SCALE, C3_sh, lam3_sc * T_SCALE, t1, t2)

            n_langs = len(languages)
            ncols = min(4, n_langs)
            nrows = (n_langs + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            for i, lang in enumerate(languages):
                ax = axes_flat[i]
                p = per_lang_params[lang]
                if per_language_ac:
                    gamma, lam2_sc, lam3_sc = core
                    lam2_b, lam3_b = lam2_sc * T_SCALE, lam3_sc * T_SCALE
                    phase1_b = np.power(t_plot, -gamma)
                    dt2 = np.maximum(t_plot - t1, 0.0)
                    dt3 = np.maximum(t_plot - t2, 0.0)
                    phase2_b = np.where(t_plot <= t1, 0.0, 1.0 - np.exp(-dt2 / lam2_b))
                    phase3_b = np.where(t_plot <= t2, 0.0, 1.0 - np.exp(-dt3 / lam3_b))
                    R_pred_plot = p["alpha"] + p["A"] * phase1_b + p["C2"] * phase2_b + p["C3"] * phase3_b
                else:
                    R_pred_plot = p["alpha"] + p["beta"] * f_plot
                ax.scatter(t_by_lang[lang] * T_SCALE, R_by_lang[lang], s=18, zorder=5, color="steelblue", label="data")
                ax.plot(t_plot, R_pred_plot, color="crimson", linewidth=1.5, label="fit")
                ax.axvline(t1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axvline(t2, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_title(f"{lang}  R²={p['r2']:.3f}", fontsize=9)
                ax.set_xlabel("Tokens (B)", fontsize=8)
                ax.set_ylabel("RankMe", fontsize=8)
                ax.tick_params(labelsize=7)

            for j in range(n_langs, len(axes_flat)):
                axes_flat[j].set_visible(False)

            if per_language_ac:
                gamma, lam2_sc, lam3_sc = core
                param_str = (f"γ={gamma:.3g}  λ2={lam2_sc*T_SCALE:.3g}B  λ3={lam3_sc*T_SCALE:.3g}B"
                             f"  t1={t1:.3g}B  t2={t2:.3g}B")
                model_tag = "3ph-per-lang-AC"
            else:
                A_sh_sc, gamma, C2_sh, lam2_sc, C3_sh, lam3_sc = core
                param_str = (f"A={A_sh_sc*T_SCALE**gamma:.3g}  γ={gamma:.3g}"
                             f"  C2={C2_sh:.3g}  λ2={lam2_sc*T_SCALE:.3g}B"
                             f"  C3={C3_sh:.3g}  λ3={lam3_sc*T_SCALE:.3g}B"
                             f"  t1={t1:.3g}B  t2={t2:.3g}B")
                model_tag = "3ph-shared-AC"
            fig.suptitle(f"{args.layer} / {args.aggregation} / {model_tag}\n{param_str}", fontsize=10)
            plt.tight_layout()
            plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved → {args.output_plot}")

    else:
        # --- 2-phase ---
        fix_t_change = not args.estimate_changepoint
        t_change_val = T_CHANGE_FIXED
        t_change_val_sc = t_change_val / T_SCALE
        print(f"  t_change: {'fixed at ' + str(t_change_val) + 'B' if fix_t_change else 'estimated'}")

        core, t_change_sc, result = fit_model(
            t_by_lang, R_by_lang, fix_t_change, per_language_ac,
            t_change_value=t_change_val_sc,
        )
        t_change = t_change_sc * T_SCALE  # back to B

        for lang in languages:
            R_mean = R_by_lang[lang].mean()
            ss_tot = float(np.sum((R_by_lang[lang] - R_mean) ** 2))
            if per_language_ac:
                gamma, lam_sc = core
                alpha, A_sc, C, sse = fit_per_language_ac(
                    t_by_lang[lang], R_by_lang[lang], gamma, lam_sc, t_change_sc)
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
                per_lang_params[lang] = dict(alpha=alpha, A=A_sc * T_SCALE**gamma, C=C, sse=sse, r2=r2)
            else:
                f_vals = _apply_f(t_by_lang[lang], core, t_change_sc)
                alpha, beta, sse = fit_per_language(f_vals, R_by_lang[lang])
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
                per_lang_params[lang] = dict(alpha=alpha, beta=beta, sse=sse, r2=r2)

        print(f"\n{'─'*55}")
        if per_language_ac:
            gamma, lam_sc = core
            print("Shared parameters  (per-language-ac=True)")
            print(f"{'─'*55}")
            print(f"  γ         = {gamma:.6g}")
            print(f"  λ         = {lam_sc * T_SCALE:.6g} B tokens  (Exponential scale)")
        else:
            A_sh_sc, gamma, C_sh, lam_sc = core
            print("Shared parameters  (phase2=exponential)")
            print(f"{'─'*55}")
            print(f"  A         = {A_sh_sc * T_SCALE**gamma:.6g}")
            print(f"  γ         = {gamma:.6g}")
            print(f"  C         = {C_sh:.6g}  (phase-2 amplitude)")
            print(f"  λ         = {lam_sc * T_SCALE:.6g} B tokens  (Exponential scale)")
        print(f"  t_change  = {t_change:.6g} B tokens")
        print(f"  Optimiser : success={result.success},  total SSE={result.fun:.6g}")

        print(f"\n{'─'*75}")
        if per_language_ac:
            print(f"{'Language':<16} {'α':>12} {'A':>12} {'C':>12} {'C/A':>10} {'R²':>8}")
            print(f"{'─'*75}")
            for lang in languages:
                p = per_lang_params[lang]
                c_over_a = p['C'] / p['A'] if p['A'] != 0 else float('nan')
                print(f"{lang:<16} {p['alpha']:>12.4f} {p['A']:>12.4e} {p['C']:>12.4e} {c_over_a:>10.4f} {p['r2']:>8.4f}")
        else:
            print(f"{'Language':<16} {'α':>12} {'β':>14} {'R²':>8}")
            print(f"{'─'*55}")
            for lang in languages:
                p = per_lang_params[lang]
                print(f"{lang:<16} {p['alpha']:>12.4f} {p['beta']:>14.4e} {p['r2']:>8.4f}")

        if args.output_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # t_by_lang is in T; build plot range in B
            t_min_all = min(t.min() for t in t_by_lang.values()) * T_SCALE
            t_max_all = max(t.max() for t in t_by_lang.values()) * T_SCALE
            t_plot = np.linspace(t_min_all, t_max_all, 600)  # B

            if not per_language_ac:
                A_sh_sc, gamma, C_sh, lam_sc = core
                # Use B-unit params with B-unit t_plot
                core_b = (A_sh_sc * T_SCALE**gamma, gamma, C_sh, lam_sc * T_SCALE)
                f_plot = _apply_f(t_plot, core_b, t_change)

            n_langs = len(languages)
            ncols = min(4, n_langs)
            nrows = (n_langs + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            for i, lang in enumerate(languages):
                ax = axes_flat[i]
                p = per_lang_params[lang]
                if per_language_ac:
                    gamma, lam_sc = core
                    lam_b = lam_sc * T_SCALE
                    phase1_basis = np.power(t_plot, -gamma)
                    dt = np.maximum(t_plot - t_change, 0.0)
                    phase2_basis = np.where(t_plot <= t_change, 0.0, 1.0 - np.exp(-dt / lam_b))
                    R_pred_plot = p["alpha"] + p["A"] * phase1_basis + p["C"] * phase2_basis
                else:
                    R_pred_plot = p["alpha"] + p["beta"] * f_plot
                ax.scatter(t_by_lang[lang] * T_SCALE, R_by_lang[lang], s=18, zorder=5, color="steelblue", label="data")
                ax.plot(t_plot, R_pred_plot, color="crimson", linewidth=1.5, label="fit")
                ax.axvline(t_change, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_title(f"{lang}  R²={p['r2']:.3f}", fontsize=9)
                ax.set_xlabel("Tokens (B)", fontsize=8)
                ax.set_ylabel("RankMe", fontsize=8)
                ax.tick_params(labelsize=7)

            for j in range(n_langs, len(axes_flat)):
                axes_flat[j].set_visible(False)

            if per_language_ac:
                gamma, lam_sc = core
                param_str = f"γ={gamma:.3g}  λ={lam_sc*T_SCALE:.3g}B  t_change={t_change:.3g}B"
                model_tag = "per-lang-AC"
            else:
                A_sh_sc, gamma, C_sh, lam_sc = core
                param_str = (f"A={A_sh_sc*T_SCALE**gamma:.3g}  γ={gamma:.3g}"
                             f"  C={C_sh:.3g}  λ={lam_sc*T_SCALE:.3g}B  t_change={t_change:.3g}B")
                model_tag = "shared-AC"
            fig.suptitle(f"{args.layer} / {args.aggregation} / {model_tag}\n{param_str}", fontsize=10)
            plt.tight_layout()
            plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved → {args.output_plot}")


if __name__ == "__main__":
    main()
