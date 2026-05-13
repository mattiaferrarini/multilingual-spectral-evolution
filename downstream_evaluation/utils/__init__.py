"""
Utility package for the downstream evaluation results notebook.

Modules:
    checkpoints  — checkpoint name parsing and token-axis formatting
    data         — loading RankMe and merged evaluation CSVs
    phases       — entropy/compression phase identification from RankMe curves
    grokking     — grokking onset detection from accuracy trajectories
    correlations — Spearman/Pearson correlation analysis
    plots        — all visualisation functions
"""

from .config import load_config
from .checkpoints import ckpt_to_tokens, sort_checkpoints, format_tokens, apply_token_formatter
from .data import load_rankme_data, load_eval_data
from .phases import compute_phases
from .grokking import compute_grokking
from .correlations import compute_correlations_table
from .plots import plot_rankme_phases, plot_overlay, plot_correlation_scatter, PHASE_COLORS, PHASE_LABELS
