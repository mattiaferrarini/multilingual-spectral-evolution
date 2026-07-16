import os
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Set seeds for reproducibility across random, numpy, and torch.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hf_dataset(path, subset, split, streaming):
    load_kwargs = dict(split=split, streaming=streaming)
    if subset:
        load_kwargs["name"] = subset
    return load_dataset(path, **load_kwargs)


def sample_and_tokenize_dataset(ds_config, num_sequences, tokenizer, max_seq_len, min_seq_len, seed=42, streaming=False):
    """
    Loads a dataset, tokenizes the text, and filters for sequences that meet length requirements.
    When streaming=False (default), downloads the full subset first for proper uniform sampling.
    When streaming=True, uses a buffer-based shuffle which may oversample the start of the dataset.
    """
    path = ds_config["path"]
    subset = ds_config.get("subset", None)
    split = ds_config.get("split", "train")
    text_col = ds_config.get("text_column", "text")

    logger.info(
        f"Sampling {num_sequences} valid sequences from '{path}'"
        + (f" subset='{subset}'" if subset else "")
        + f" (min_len={min_seq_len}, max_len={max_seq_len}, seed={seed}, streaming={streaming})"
    )

    dataset = load_hf_dataset(path, subset, split, streaming=streaming)
    logger.info("Shuffling dataset...")
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    else:
        dataset = dataset.shuffle(seed=seed)
    logger.info("Shuffle complete.")

    valid_samples = []
    chunk_size = 1000  # Process in chunks for tokenizer efficiency
    buffer = []

    it = iter(dataset)
    special_ids = set(tokenizer.all_special_ids)

    def process_buffer(buf):
        """Internal helper to tokenize a batch and filter by 'content' length (ignoring special tokens)."""
        if not buf:
            return []
        enc = tokenizer(
            buf,
            truncation=True,
            max_length=max_seq_len,
            padding=False
        )
        tokenized = enc["input_ids"]
        valid = []
        for ids in tokenized:
            content_len = sum(1 for tok in ids if tok not in special_ids)
            if content_len >= min_seq_len:
                valid.append(ids)
        return valid

    try:
        while len(valid_samples) < num_sequences:
            row = next(it)
            text = row.get(text_col, "")
            if text:
                buffer.append(text)

            if len(buffer) >= chunk_size:
                for ids in process_buffer(buffer):
                    valid_samples.append(ids)
                    if len(valid_samples) == num_sequences:
                        break
                buffer = []
                if len(valid_samples) < num_sequences:
                    logger.info(f"Progress: {len(valid_samples)}/{num_sequences} valid samples collected...")
    except StopIteration:
        # Process remaining buffer
        if len(valid_samples) < num_sequences and buffer:
            for ids in process_buffer(buffer):
                valid_samples.append(ids)
                if len(valid_samples) == num_sequences:
                    break

        if len(valid_samples) < num_sequences:
            logger.warning(f"Dataset exhausted! Only collected {len(valid_samples)}/{num_sequences} valid samples.")

    logger.info(f"Final collection: {len(valid_samples)} valid samples.")
    
    # Ensure deterministic sample order across executions
    valid_samples.sort()
    
    return valid_samples


def prepare_dataset_samples(config, pending_ds_names, results_dir, model_name, model_cfg, coll):
    """
    Coordinates the sampling of all datasets listed in the config. 
    If the samples have already been cached on disk, it loads them to save time. Otherwise, it runs the sampling logic and saves the results for next time.
    """
    seed = coll.get("seed", 42)
    set_seed(seed)
    num_sequences = coll.get("num_sequences", 10000)
    max_seq_len = coll.get("max_sequence_length", 512)
    min_seq_len = coll.get("min_sequence_length", 0)
    streaming = coll.get("streaming", False)

    tokenizer_name = model_cfg.get("tokenizer_name", model_name)
    logger.info(f"Loading tokenizer for '{tokenizer_name}' to prepare spans...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    
    # Namespace the cache by tokenizer identity: different models sharing the same
    # samples_dir must not reuse each other's tokenized spans (vocab/ids differ per model).
    tokenizer_dir = tokenizer_name.replace("/", "__")

    dataset_samples = {}
    for ds_config in config["datasets"]:
        ds_name = ds_config["name"]
        if ds_name in pending_ds_names:
            ds_cache_dir = os.path.join(results_dir, ds_config["path"], tokenizer_dir)
            os.makedirs(ds_cache_dir, exist_ok=True)
            cache_file = os.path.join(ds_cache_dir, f"samples_{ds_name}_{num_sequences}_{max_seq_len}_{seed}.pt")
            if os.path.exists(cache_file):
                logger.info(f"⚠️ [CACHE HIT] Loading pre-sampled data for '{ds_name}' from {cache_file}")
                dataset_samples[ds_name] = torch.load(cache_file, weights_only=False)
            else:
                samples = sample_and_tokenize_dataset(
                    ds_config, num_sequences, tokenizer, max_seq_len, min_seq_len, seed=seed, streaming=streaming
                )
                torch.save(samples, cache_file)
                dataset_samples[ds_name] = samples
    del tokenizer

    for ds_name, samples in dataset_samples.items():
        assert len(samples) == num_sequences, (
            f"Dataset '{ds_name}': expected {num_sequences} samples, got {len(samples)}"
        )

    return dataset_samples
