import time
import torch
import logging
from utils import _fmt

logger = logging.getLogger(__name__)


def resolve_layer_indices(layers_config, num_layers):
    """
    Filter layers indices to collect based on the config. 
    
    Layers can be specified as:
    - "all": All layers.
    - A list: Like [0, 10, "last"]. Will resolve "last" to the actual final layer index.
    - A dict: With 'start', 'end', and 'step' to define a range (like a Python slice).

    Returns a sorted list of unique layer indices.
    """
    if layers_config in ("all", ["all"]):
        return list(range(num_layers))
    if isinstance(layers_config, list):
        resolved = []
        for l in layers_config:
            if l == "last":
                resolved.append(num_layers - 1)
            elif isinstance(l, int):
                resolved.append(l)
            else:
                raise ValueError(f"Unknown layer specifier: {l!r}")
        return sorted(set(resolved))
    if isinstance(layers_config, dict):
        start = int(layers_config.get("start", 0))
        end = int(layers_config.get("end", num_layers))
        step = int(layers_config.get("step", 1))
        return list(range(start, end, step))
    raise ValueError(f"Invalid layers config: {layers_config!r}")


def apply_aggregation(layer_tensor, seq_lens, aggregation, input_ids=None, special_ids=None):
    """
    Takes the raw hidden states and squashes them into a single vector per sequence, ignoring padding and special tokens.
    
    Available methods:
    - "last": Just take the vector from the very last meaningful token.
    - "mean": Average all the token vectors together.
    """
    out = []
    for i, slen in enumerate(seq_lens):
        acts = layer_tensor[i, :slen, :]  # (slen, hidden_dim)
        
        if input_ids is not None and special_ids is not None:
            # Get token IDs for this sequence
            seq_ids = input_ids[i, :slen].tolist()
            # Find indices of non-special tokens
            valid_indices = [idx for idx, tok in enumerate(seq_ids) if tok not in special_ids]
            acts = acts[valid_indices, :]
        
        effective_len = acts.shape[0]
        if effective_len <= 0:
            continue

        if aggregation == "last":
            out.append(acts[-1])
        elif aggregation == "mean":
            out.append(acts.mean(dim=0))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")
    
    if not out:
        return None
    return torch.stack(out, dim=0)    # (bsz, hidden_dim)


def collect_activations(hf_model, tokenizer, samples, layer_indices, max_seq_len, aggregation, batch_size,
                        progress_interval=100, label="", debug=False):
    """
    Collect activations for specified layers and samples, applying the chosen aggregation method.
    
    Loops through data in batches, feeds it through the model, and uses forward hooks to collect hidden states of the specified layers.
    It then aggregates those states (e.g., takes the mean) to produce one (N, hidden_dim) tensor for each layer.
    """
    n_total = len(samples)
    accum = {f"layer_{i}": [] for i in layer_indices}
    n_collected = 0
    last_log_at = 0
    t0 = time.time()

    # Direct access to layers for standard architectures
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        hf_layers = hf_model.model.layers
    elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        hf_layers = hf_model.transformer.h
    else:
        raise RuntimeError("Could not locate decoder layers in the model.")

    device = next(hf_model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    special_ids = set(tokenizer.all_special_ids)
    
    samples_logged = 0

    for i in range(0, n_total, batch_size):
        batch_ids = samples[i : i + batch_size]
        
        if debug and samples_logged < 3:
            for j, ids in enumerate(batch_ids):
                if samples_logged < 3:
                    decoded = tokenizer.decode(ids)
                    logger.info(f"[{label}] DEBUG SAMPLE {samples_logged}: {decoded}")
                    samples_logged += 1
        
        # manual padding to the longest span in this batch
        batch_len = max(len(ids) for ids in batch_ids)
        input_ids = torch.full((len(batch_ids), batch_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch_ids), batch_len), dtype=torch.long, device=device)
        for j, ids in enumerate(batch_ids):
            input_ids[j, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[j, :len(ids)] = 1

        seq_lens = attention_mask.sum(dim=1).tolist()
        captured = {}
        hooks = []

        def make_hook(key):
            def _h(*args):
                out = args[2]
                hs = out[0] if isinstance(out, (tuple, list)) else out
                captured[key] = hs.detach().cpu()
            return _h

        for idx in layer_indices:
            hooks.append(hf_layers[idx].register_forward_hook(make_hook(f"layer_{idx}")))

        try:
            with torch.no_grad():
                hf_model(input_ids, attention_mask=attention_mask)
        finally:
            for h in hooks:
                h.remove()

        input_ids_cpu = input_ids.cpu()
        for layer_name, tensor in captured.items():
            agg_result = apply_aggregation(
                tensor, seq_lens, aggregation, 
                input_ids=input_ids_cpu, special_ids=special_ids
            )
            if agg_result is not None:
                accum[layer_name].append(agg_result)

        n_collected += len(batch_ids)
        if n_collected - last_log_at >= progress_interval or n_collected == n_total:
            elapsed = time.time() - t0
            rate = n_collected / elapsed
            remaining = (n_total - n_collected) / rate if rate > 0 else 0
            logger.info(
                f"[{label}] {n_collected}/{n_total} seqs"
                f" | elapsed {_fmt(elapsed)}"
                f" | ~{_fmt(remaining)} remaining"
                f" | {rate:.1f} seq/s"
            )
            last_log_at = n_collected

    elapsed = time.time() - t0
    logger.info(
        f"[{label}] Done — {n_collected} seqs in {_fmt(elapsed)}"
        f" ({n_collected / elapsed:.1f} seq/s)"
    )
    return {k: torch.cat(v, dim=0) for k, v in accum.items() if v}
