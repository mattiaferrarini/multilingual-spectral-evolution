"""
Utility package for the downstream evaluation results notebook.

Modules:
    checkpoints      — checkpoint name parsing and token-axis formatting
    data             — loading RankMe and merged evaluation CSVs
    phases           — entropy/compression phase identification from RankMe curves
    grokking         — grokking onset detection from accuracy trajectories
    correlations     — Spearman/Pearson correlation analysis
    plots            — all visualisation functions
    layer_selection  — cross-language stratification analysis for layer selection
    overlay_widget   — interactive ipywidgets UI for overlay plot task selection
"""

from .config import load_config, init_models
from .checkpoints import ckpt_to_tokens, sort_checkpoints, format_tokens, apply_token_formatter
from .data import load_rankme_data, load_eval_data, load_model_data
from .phases import compute_geometry, compute_and_show_geometry
from .alpha_phases import compute_alpha_phases, compute_and_show_alpha_phases
from .grokking import compute_grokking, compute_and_show_grokking
from .correlations import compute_correlations_table, compute_and_show_correlations, compute_alpha_correlations_table, compute_and_show_alpha_correlations, compute_checkpoint_correlations, compute_and_show_checkpoint_correlations
from .layer_selection import compute_stratification, plot_stratification, stratification_summary, show_stratification, layer_selection_sweep
from .plots import (plot_rankme_phases, show_rankme_phases, plot_overlay,
                    plot_alpha_phases, show_alpha_phases, plot_alpha_overlay,
                    plot_predictor_scatter, show_predictor_scatter,
                    plot_correlation_heatmap, show_correlation_heatmap,
                    plot_alpha_correlation_heatmap, show_alpha_correlation_heatmap,
                    plot_alpha_correlation_scatter_combined, plot_alpha_rate_scatter_combined,
                    plot_checkpoint_correlations, show_checkpoint_correlations,
                    PHASE_COLORS, PHASE_LABELS, ALPHA_PHASE_COLORS, ALPHA_PHASE_LABELS)
from .overlay_widget import show_overlay_widget, show_alpha_overlay_widget
