"""
Utility package for the downstream evaluation results notebook.

Modules:
    checkpoints     — checkpoint name parsing and token-axis formatting
    data            — loading RankMe and merged evaluation CSVs
    phases          — entropy/compression phase identification from RankMe curves
    grokking        — grokking onset detection from accuracy trajectories
    correlations    — Spearman/Pearson correlation analysis
    plots           — all visualisation functions
    layer_selection — cross-language stratification analysis for layer selection
"""

from .config import load_config
from .checkpoints import ckpt_to_tokens, sort_checkpoints, format_tokens, apply_token_formatter
from .data import load_rankme_data, load_eval_data
from .phases import compute_geometry
from .alpha_phases import compute_alpha_phases
from .grokking import compute_grokking
from .correlations import compute_correlations_table, compute_alpha_correlations_table
from .layer_selection import compute_stratification, plot_stratification, stratification_summary
from .plots import (plot_rankme_phases, plot_overlay,
                    plot_alpha_phases, plot_alpha_overlay,
                    plot_predictor_scatter,
                    plot_alpha_correlation_scatter_combined, plot_alpha_rate_scatter_combined,
                    PHASE_COLORS, PHASE_LABELS, ALPHA_PHASE_COLORS, ALPHA_PHASE_LABELS)
