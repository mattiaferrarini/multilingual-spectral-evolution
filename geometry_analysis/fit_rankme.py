"""
Fit the piecewise model to RankMe scores for a fixed (layer, aggregation) pair.

Model:
    R_ℓ(t) = α_ℓ + β_ℓ · f(t)

    f(t) = A · t^{-γ}                                  for t ≤ t_change
    
    Phase 2 (t > t_change):
      exponential: A · t_change^{-γ} + C · (1 - exp(-(t - t_change)/λ))

Shared parameters: A, γ, C, λ [, t_change if --estimate-changepoint].
Per-language parameters: α_ℓ, β_ℓ  (profiled out analytically via OLS).

t is measured in billions of tokens throughout.

Alternative Model (with --per-language-ac):
    R_ℓ(t) = α_ℓ + A_ℓ · t^{-γ} + C_ℓ · (1 − exp(−(t − t_change)/λ))
    where γ, λ are shared, and α_ℓ, A_ℓ, C_ℓ are per-language.
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

T_CHANGE_FIXED = 241.0  # billions of tokens (default changepoint)


def parse_checkpoint(s: str) -> float | None:
    m = re.match(r"^(\d+(?:\.\d+)?)([BM]?)$", str(s))
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    if unit == "B":
        return v
    if unit == "M":
        return v / 1_000.0
    return v


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


# ---- Optimization objective ----

def total_sse(shared_params, t_by_lang, R_by_lang, fix_t_change: bool, per_language_ac: bool) -> float:
    if fix_t_change:
        core, t_change = shared_params, T_CHANGE_FIXED
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
) -> tuple:
    """
    Optimise shared parameters via differential evolution (global), then refine.
    Returns (core_params_tuple, t_change, result).
    """
    t_all = np.concatenate(list(t_by_lang.values()))
    t_min, t_max = t_all.min(), t_all.max()

    if per_language_ac:
        bounds_core = [
            (0.01, 5.0),            # gamma
            (0.01, t_max - t_min),  # λ: scale in B tokens
        ]
    else:
        bounds_core = [
            (1e-6, 1e10),           # A
            (0.01, 5.0),            # gamma
            (-1e10, 1e10),          # C (amplitude; sign unconstrained)
            (0.01, t_max - t_min),  # λ: scale in B tokens
        ]
        
    bounds = bounds_core if fix_t_change else bounds_core + [(t_min + 1.0, t_max - 1.0)]

    result = differential_evolution(
        total_sse,
        bounds=bounds,
        args=(t_by_lang, R_by_lang, fix_t_change, per_language_ac),
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
        t_change = T_CHANGE_FIXED
    else:
        *core_list, t_change = result.x
        core = tuple(core_list)

    return core, t_change, result


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
        help="Estimate t_change instead of fixing it at 241B.",
    )
    parser.add_argument(
        "--per-language-ac",
        action="store_true",
        default=False,
        help="Switch to per-language A and C parameters.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Save a fit plot to this path (e.g. fit.png). Skipped if not provided.",
    )
    args = parser.parse_args()

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
        t_by_lang[lang] = sub["t"].values.astype(float)
        R_by_lang[lang] = sub["rankme"].values.astype(float)

    fix_t_change = not args.estimate_changepoint
    per_language_ac = args.per_language_ac
    
    print(f"Layer={args.layer}  aggregation={args.aggregation}  phase2=exponential  per_lang_ac={per_language_ac}")
    print(f"Languages ({len(languages)}): {', '.join(languages)}")
    print(f"Checkpoints: {len(df_sub['t'].unique())}  "
          f"(range {df_sub['t'].min():.0f}B – {df_sub['t'].max():.0f}B tokens)")
    print(f"t_change: {'fixed at ' + str(T_CHANGE_FIXED) + 'B' if fix_t_change else 'estimated'}")
    print("Optimising…")

    # Fit
    core, t_change, result = fit_model(t_by_lang, R_by_lang, fix_t_change, per_language_ac)

    # Compute per-language parameters and R² scores
    per_lang_params = {}
    for lang in languages:
        R_mean = R_by_lang[lang].mean()
        ss_tot = float(np.sum((R_by_lang[lang] - R_mean) ** 2))
        
        if per_language_ac:
            gamma, lam = core
            alpha, A, C, sse = fit_per_language_ac(t_by_lang[lang], R_by_lang[lang], gamma, lam, t_change)
            r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
            per_lang_params[lang] = dict(alpha=alpha, A=A, C=C, sse=sse, r2=r2)
        else:
            f_vals = _apply_f(t_by_lang[lang], core, t_change)
            alpha, beta, sse = fit_per_language(f_vals, R_by_lang[lang])
            r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
            per_lang_params[lang] = dict(alpha=alpha, beta=beta, sse=sse, r2=r2)

    # Log results
    print(f"\n{'─'*55}")
    if per_language_ac:
        gamma, lam = core
        print("Shared parameters  (per-language-ac=True)")
        print(f"{'─'*55}")
        print(f"  γ         = {gamma:.6g}")
        print(f"  λ         = {lam:.6g} B tokens  (Exponential scale)")
    else:
        A, gamma, C, lam = core
        print("Shared parameters  (phase2=exponential)")
        print(f"{'─'*55}")
        print(f"  A         = {A:.6g}")
        print(f"  γ         = {gamma:.6g}")
        print(f"  C         = {C:.6g}  (phase-2 amplitude)")
        print(f"  λ         = {lam:.6g} B tokens  (Exponential scale)")
    
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

    # Optional plotting
    if args.output_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_min_all = min(t.min() for t in t_by_lang.values())
        t_max_all = max(t.max() for t in t_by_lang.values())
        t_plot = np.linspace(t_min_all, t_max_all, 600)
        
        if not per_language_ac:
            f_plot = _apply_f(t_plot, core, t_change)

        n_langs = len(languages)
        ncols = min(4, n_langs)
        nrows = (n_langs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for i, lang in enumerate(languages):
            ax = axes_flat[i]
            t_arr = t_by_lang[lang]
            R_arr = R_by_lang[lang]
            p = per_lang_params[lang]
            
            if per_language_ac:
                gamma, lam = core
                phase1_basis = np.power(t_plot, -gamma)
                dt = np.maximum(t_plot - t_change, 0.0)
                phase2_basis = np.where(t_plot <= t_change, 0.0, 1.0 - np.exp(-dt / lam))
                R_pred_plot = p["alpha"] + p["A"] * phase1_basis + p["C"] * phase2_basis
            else:
                R_pred_plot = p["alpha"] + p["beta"] * f_plot
                
            ax.scatter(t_arr, R_arr, s=18, zorder=5, color="steelblue", label="data")
            ax.plot(t_plot, R_pred_plot, color="crimson", linewidth=1.5, label="fit")
            ax.axvline(t_change, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(f"{lang}  R²={p['r2']:.3f}", fontsize=9)
            ax.set_xlabel("Tokens (B)", fontsize=8)
            ax.set_ylabel("RankMe", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(n_langs, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if per_language_ac:
            gamma, lam = core
            param_str = f"γ={gamma:.3g}  λ={lam:.3g}B  t_change={t_change:.3g}B"
        else:
            A, gamma, C, lam = core
            param_str = f"A={A:.3g}  γ={gamma:.3g}  C={C:.3g}  λ={lam:.3g}B  t_change={t_change:.3g}B"
            
        fig.suptitle(f"{args.layer} / {args.aggregation} / exponential / {'per-lang-AC' if per_language_ac else 'shared-AC'}\n{param_str}", fontsize=10)
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {args.output_plot}")


if __name__ == "__main__":
    main()
