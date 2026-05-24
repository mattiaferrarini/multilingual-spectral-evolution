"""
Checkpoint name parsing and token-axis formatting utilities.

Converts checkpoint names (e.g. "531B", "step2627139-tokens15T") to a common
numeric unit (billions of tokens) used as the x-axis across all plots.
"""

import re
from matplotlib.ticker import FuncFormatter

_APERTUS_RE = re.compile(r"^step\d+-tokens(\d+)([BT])$")
_GENERIC_RE  = re.compile(r"^(\d+(?:\.\d+)?)")


def ckpt_to_tokens(name: str) -> float:
    """Convert a checkpoint name to billions of tokens."""
    m = _APERTUS_RE.match(str(name))
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2) == "T" else val
    m = _GENERIC_RE.match(str(name))
    return float(m.group(1)) if m else float("inf")


def sort_checkpoints(checkpoints) -> list:
    return sorted(checkpoints, key=ckpt_to_tokens)


def format_tokens(x, pos=None) -> str:
    """Axis label: T for values ≥ 1000 B, otherwise B."""
    if x >= 1000:
        return f"{x/1000:.4g}T"
    return f"{x:.4g}B"


def apply_token_formatter(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(format_tokens))
    ax.set_xlabel("Tokens")
