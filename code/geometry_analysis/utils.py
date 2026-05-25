import re
import os
import pandas as pd
import logging
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_APERTUS_BRANCH_RE = re.compile(r'^step(\d+)-tokens(\d+)([BT])$')


def _fmt(seconds):
    """
    Format seconds in human-readable form.
    """
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _checkpoint_sort_key(name):
    """
    Sort checkpoints numerically. 
    """
    # Apertus format: step{N}-tokens{M}[BT] — sort by token count in B (T = 1000B).
    m = _APERTUS_BRANCH_RE.match(name)
    if m:
        tokens = float(m.group(2)) * (1000 if m.group(3) == "T" else 1)
        return tokens
    # Generic format: leading number (e.g. "10B", "100B").
    m = re.match(r'^(\d+(?:\.\d+)?)', name)
    return float(m.group(1)) if m else float("inf")


def discover_branches(model_name, exclude=("main",), branch_filter_pattern=None):
    """
    Find all the different branches (checkpoints) available for a model on HuggingFace Hub . 
    """
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
    Decides which checkpoints to process.
    Two main options:
        - "all": Discover all branches on HuggingFace Hub and treat them as checkpoints.
        - A list: Like [0, 10, "last"]. We'll resolve "last" to the actual final layer index.
    Returns a sorted list of checkpoint labels.
    """
    if checkpoints_config is None:
        return [None]
    spec = checkpoints_config
    if spec == "all" or spec == ["all"]:
        return discover_branches(model_name, branch_filter_pattern=branch_filter_pattern)
    if isinstance(spec, list):
        return sorted([str(c) for c in spec], key=_checkpoint_sort_key)
    raise ValueError(f"Invalid checkpoints config: {spec!r}. Use 'all', null, or a list.")


def load_model(model_name, revision, model_cfg):
    """
    Load model and tokenizer from HuggingFace Hub.
    """
    kwargs = {"device_map": "auto"}
    if revision is not None:
        kwargs["revision"] = revision
    if model_cfg.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    kwargs["dtype"] = model_cfg.get("torch_dtype", "bfloat16")
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    label = revision or "default"
    logger.info(f"Loading model '{model_name}' revision='{label}'")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Direct access to layers for standard architectures (Llama, Apertus, etc.)
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        num_layers = len(hf_model.model.layers)
    elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        num_layers = len(hf_model.transformer.h)
    else:
        num_layers = getattr(hf_model.config, "num_hidden_layers", 0)

    return hf_model, tokenizer, num_layers


def load_existing_results(results_path):
    """
    Load existing results from CSV if it exists, to avoid re-computing metrics for (checkpoint, dataset) pairs we've already done.
    """
    if not os.path.exists(results_path):
        return [], set()
    df = pd.read_csv(results_path)
    if df.empty or "checkpoint" not in df.columns or "dataset" not in df.columns:
        return [], set()
    rows = df.to_dict("records")
    completed_pairs = set(zip(df["checkpoint"], df["dataset"]))
    return rows, completed_pairs


def _ckpt_label(ckpt):
    """Simple helper to give a nice name to a checkpoint (uses 'default' if it's None)."""
    return ckpt if ckpt is not None else "default"
