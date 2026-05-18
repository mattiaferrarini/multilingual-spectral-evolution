"""
Checkpoint resolution for HuggingFace Hub models.
Supports auto-discovery of all branches, explicit lists, and filtering.
"""

import re
import logging
from huggingface_hub import list_repo_refs

logger = logging.getLogger(__name__)

_APERTUS_BRANCH_RE = re.compile(r'^step(\d+)-tokens(\d+)([BT])$')


def _checkpoint_sort_key(name):
    """Sort by training progress: Apertus token count, or leading number, else last."""
    m = _APERTUS_BRANCH_RE.match(name)
    if m:
        return float(m.group(2)) * (1000 if m.group(3) == "T" else 1)
    m = re.match(r'^(\d+(?:\.\d+)?)', name)
    return float(m.group(1)) if m else float("inf")


def discover_branches(model_name, exclude=("main",), branch_filter_pattern=None):
    """List all HuggingFace Hub branches for a model, sorted by training progress."""
    refs = list_repo_refs(model_name)
    all_branches = [b.name for b in refs.branches]
    branches = [b for b in all_branches if b not in exclude]
    if branch_filter_pattern is not None:
        pat = re.compile(branch_filter_pattern)
        before = len(branches)
        branches = [b for b in branches if pat.match(b)]
        logger.info(f"Branch filter '{branch_filter_pattern}': {before} -> {len(branches)} branches")
    branches = sorted(branches, key=_checkpoint_sort_key)
    if "main" in all_branches and "main" not in branches:
        branches.append("main")
    logger.info(f"Discovered {len(branches)} checkpoint branch(es): {branches}")
    return branches


def resolve_checkpoints(model_name, checkpoints_config, branch_filter_pattern=None):
    """
    Resolve checkpoint list from config value.
      null / None  → [None]  (default / main branch, no revision pin)
      "all"        → auto-discover all branches on HuggingFace Hub
      list         → use as-is (sorted by training progress)
    """
    if checkpoints_config is None:
        return [None]
    spec = checkpoints_config
    if spec == "all" or spec == ["all"]:
        return discover_branches(model_name, branch_filter_pattern=branch_filter_pattern)
    if isinstance(spec, list):
        return sorted([str(c) for c in spec], key=_checkpoint_sort_key)
    raise ValueError(f"Invalid checkpoints config: {spec!r}. Use 'all', null, or a list.")


def apply_checkpoint_filters(checkpoints, checkpoint_step=None, max_checkpoints=None):
    """
    Apply step and cap filters, preserving "main" at the end if present.
    Mirrors the geometry_analysis filtering logic.
    """
    has_main = "main" in checkpoints
    filtered = [c for c in checkpoints if c != "main"]

    if checkpoint_step is not None and int(checkpoint_step) > 1:
        checkpoint_step = int(checkpoint_step)
        before = len(filtered)
        filtered = filtered[::checkpoint_step]
        logger.info(f"checkpoint_step={checkpoint_step}: keeping {len(filtered)}/{before} checkpoints")

    if max_checkpoints is not None and len(filtered) > int(max_checkpoints):
        logger.info(f"Capping checkpoints from {len(filtered)} to {max_checkpoints} (max_checkpoints)")
        filtered = filtered[:int(max_checkpoints)]

    if has_main:
        filtered.append("main")

    return filtered


def ckpt_label(ckpt):
    """Human-readable label: 'default' when ckpt is None (no revision pinned)."""
    return ckpt if ckpt is not None else "default"
