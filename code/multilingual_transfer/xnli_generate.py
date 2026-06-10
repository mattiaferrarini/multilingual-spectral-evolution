"""
XNLI Cross-Lingual ICL Evaluation
Tests multilingual transfer between all language pairs via in-context learning.
Designed for pre-trained (non-instruction-tuned) models: uses raw completion, no chat template.

For each of n_eval independent draws:
  - k context examples are sampled from the validation split (balanced labels, without replacement)
  - 1 test example is sampled from the test split (i.i.d. across draws)
  - All (src_lang, tgt_lang) pairs are evaluated under the same draw

All checkpoints are evaluated on the identical context and test draws (pre-computed once).
One predictions CSV and one generations JSONL are saved per checkpoint.
Premise/hypothesis text and full prompts are not stored; use test_idx + tgt_lang/src_lang to reconstruct from XNLI.

Usage: python code/multilingual_transfer/xnli_generate.py --config code/multilingual_transfer/configs/xnli.yaml
"""

import gc
import os
import re
import sys
import json
import time
import logging
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from datasets import load_dataset
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoints import resolve_checkpoints, apply_checkpoint_filters, ckpt_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_KEYWORDS = {"entailment": 0, "neutral": 1, "contradiction": 2}

# Completion-style prompt — no instruction wrapper, suitable for pre-trained models.
_SHOT_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}\nLabel (Entailment/Neutral/Contradiction): {label}"
_TEST_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}\nLabel (Entailment/Neutral/Contradiction):"



def build_icl_prompt(context_examples: list[dict], test_example: dict) -> str:
    """Concatenate k labeled shots (in src_lang) followed by an unlabeled test (in tgt_lang)."""
    parts = [
        _SHOT_TEMPLATE.format(
            premise=ex["premise"],
            hypothesis=ex["hypothesis"],
            label=LABEL_MAP[ex["label"]],
        )
        for ex in context_examples
    ]
    parts.append(_TEST_TEMPLATE.format(
        premise=test_example["premise"],
        hypothesis=test_example["hypothesis"],
    ))
    return "\n\n".join(parts)


def parse_label(response: str) -> int | None:
    """Extract a label id from the model's response: first-word check then full scan."""
    text = response.strip().lower()
    if not text:
        return None
    # Split on any non-word character so punctuation attached to the first token is ignored.
    tokens = re.split(r"\W+", text)
    first = tokens[0] if tokens else ""
    for keyword, label_id in LABEL_KEYWORDS.items():
        if first.startswith(keyword):
            return label_id
    # Full scan with word boundaries to avoid partial matches (e.g. "neutral" in "neutralize").
    for keyword, label_id in LABEL_KEYWORDS.items():
        if re.search(rf"\b{keyword}\b", text):
            return label_id
    return None


def precompute_by_label(dataset) -> dict[int, list[int]]:
    """Group dataset indices by label. Labels are parallel across XNLI languages."""
    by_label: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for i, ex in enumerate(dataset):
        by_label[ex["label"]].append(i)
    return by_label


def _sample_one_balanced_context(by_label: dict[int, list[int]], max_k: int, rng) -> list[int]:
    """
    Sample max_k indices in interleaved label order [ent, neu, con, ent, neu, con, ...].
    Any prefix [:k] is therefore approximately balanced for all k values.
    """
    per_label = (max_k + 2) // 3
    chosen = {
        label_id: rng.choice(by_label[label_id], size=min(per_label, len(by_label[label_id])), replace=False).tolist()
        for label_id in range(3)
    }
    indices: list[int] = []
    for i in range(per_label):
        for label_id in range(3):
            if i < len(chosen[label_id]):
                indices.append(chosen[label_id][i])
    return indices[:max_k]


def sample_test_indices(eval_size: int, n_eval: int, seed: int) -> list[int]:
    """Sample n_eval test indices i.i.d. with replacement."""
    rng_eval = np.random.default_rng(seed + 1)
    return [int(rng_eval.integers(0, eval_size)) for _ in range(n_eval)]


def sample_ctx_for_k(by_label: dict[int, list[int]], k: int, n_eval: int, seed: int) -> list[list[int]]:
    """
    Sample n_eval balanced context index lists of length k, using a k-specific RNG.
    Seeding per k makes draws independent across k values: adding a new k does not
    shift the RNG state for any existing k.
    """
    rng_ctx = np.random.default_rng(seed + 2 + k)
    return [
        _sample_one_balanced_context(by_label, k, rng_ctx) if k > 0 else []
        for _ in range(n_eval)
    ]


def load_full_split(languages: list[str], split: str) -> dict[str, list[dict]]:
    """Load the entire split for each language. Enables O(1) lookup by any draw index."""
    data: dict[str, list[dict]] = {}
    for lang in languages:
        logger.info(f"  Loading {split} [{lang}]...")
        ds = load_dataset("facebook/xnli", lang, split=split)
        data[lang] = [dict(ex) for ex in ds]
    return data


@dataclass
class _Output:
    text: str


@dataclass
class _RequestOutput:
    outputs: list[_Output]


class HFModel:
    """HuggingFace Transformers fallback for raw text completion (no chat template)."""

    def __init__(self, model_path: str, revision: str | None = None, seed: int = 42, batch_size: int = 8):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id or self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.seed = seed
        self.batch_size = batch_size

    def generate(self, prompts: list[str], sampling_params=None, **kwargs) -> list[_RequestOutput]:
        temperature = sampling_params.temperature if sampling_params else 0.0
        max_new_tokens = sampling_params.max_tokens if sampling_params else 16

        from transformers import DynamicCache
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda _: None

        torch.manual_seed(self.seed)
        results: list[_RequestOutput | None] = [None] * len(prompts)
        for batch_start in range(0, len(prompts), self.batch_size):
            batch = prompts[batch_start : batch_start + self.batch_size]
            tokenized = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            prompt_len = tokenized["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = self.model.generate(
                    **tokenized,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )
            for i in range(len(batch)):
                text = self.tokenizer.decode(output_ids[i][prompt_len:], skip_special_tokens=True)
                results[batch_start + i] = _RequestOutput(outputs=[_Output(text=text)])
        return results


def load_model(model_path: str, revision: str | None = None, seed: int = 42, hf_batch_size: int = 8) -> LLM | HFModel:
    kwargs = dict(model=model_path, trust_remote_code=True, dtype="bfloat16", seed=seed)
    if revision is not None:
        kwargs["revision"] = revision
    try:
        return LLM(**kwargs)
    except Exception as e1:
        try:
            return LLM(**kwargs, model_impl="transformers")
        except Exception as e2:
            logger.warning(f"vLLM failed ({e1} | {e2}), falling back to HuggingFace.")
            return HFModel(model_path, revision=revision, seed=seed, batch_size=hf_batch_size)


def unload_model(llm: LLM | HFModel) -> None:
    """Release GPU memory between checkpoints. Handles both vLLM V0 and V1 shutdown APIs."""
    if isinstance(llm, LLM):
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            try:
                engine_core = getattr(engine, "engine_core", None)
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
                    logger.info("vLLM V1 engine_core shutdown completed.")
            except Exception as e:
                logger.warning("vLLM V1 engine_core shutdown failed: %s", e)
            try:
                model_executor = getattr(engine, "model_executor", None)
                if model_executor is not None and hasattr(model_executor, "shutdown"):
                    model_executor.shutdown()
                    logger.info("vLLM V0 model_executor shutdown completed.")
            except Exception as e:
                logger.warning("vLLM V0 model_executor shutdown failed: %s", e)
            try:
                if hasattr(engine, "shutdown"):
                    engine.shutdown()
            except Exception as e:
                logger.warning("vLLM engine shutdown failed: %s", e)

    del llm
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        logger.warning("CUDA cache cleanup failed: %s", e)
    logger.info("Waiting for vLLM worker processes to release GPU memory...")
    time.sleep(15)
    logger.info("vLLM memory cleanup completed.")


def generate_completions(
    llm: LLM | HFModel,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """Raw text completion — no chat template applied."""
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def generate_pair_rows(
    llm: LLM | HFModel,
    src_lang: str,
    tgt_lang: str,
    src_lang_idx: int,
    k: int,
    draws: list[tuple[list[int], int]],
    ctx_datasets: dict[str, list[dict]],
    eval_datasets: dict[str, list[dict]],
    done_draw_is: set[int],
    temperature: float,
    max_tokens: int,
    seed: int,
) -> list[dict]:
    """Return all result rows for a single (src_lang, tgt_lang, k), passing all prompts to the model at once."""
    pending = [(i, draws[i]) for i in range(len(draws)) if i not in done_draw_is]
    if not pending:
        return []

    t0 = time.monotonic()
    prompts = []
    for draw_i, (ctx_indices, test_idx) in pending:
        # Slice to k for balance, then shuffle to remove the interleaved label pattern.
        # Seed is unique per (draw, k, src_lang) so order is deterministic.
        sliced = list(ctx_indices)
        np.random.default_rng(seed + 2 + draw_i * 10000 + k * 1000 + src_lang_idx).shuffle(sliced)
        ctx_examples = [ctx_datasets[src_lang][j] for j in sliced]
        test_example = eval_datasets[tgt_lang][test_idx]
        prompts.append(build_icl_prompt(ctx_examples, test_example))
    logger.debug(f"    [timing] prompt build:  {time.monotonic() - t0:.2f}s")

    t0 = time.monotonic()
    responses = generate_completions(llm, prompts, temperature=temperature, max_tokens=max_tokens)
    logger.debug(f"    [timing] inference:     {time.monotonic() - t0:.2f}s")

    t0 = time.monotonic()
    predicted_ids = [parse_label(r) for r in responses]

    rows = []
    for (draw_i, (ctx_indices, test_idx)), prompt, response, predicted_id in zip(pending, prompts, responses, predicted_ids):
        test_example = eval_datasets[tgt_lang][test_idx]
        rows.append({
            "draw_i": draw_i,
            "k": k,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "test_idx": test_idx,
            "gold_label": LABEL_MAP[test_example["label"]],
            "gold_label_id": test_example["label"],
            "prompt": prompt,
            "response": response,
            "predicted_label": LABEL_MAP.get(predicted_id, "unknown"),
            "predicted_label_id": predicted_id,
            "correct": predicted_id == test_example["label"],
        })
    logger.debug(f"    [timing] postprocess:   {time.monotonic() - t0:.2f}s")
    return rows


def run_checkpoint(
    llm: LLM | HFModel,
    label: str,
    pairs: list[tuple[str, str]],
    k_values: list[int],
    ctx_draws: dict[int, list[list[int]]],
    test_indices: list[int],
    languages: list[str],
    ctx_datasets: dict[str, list[dict]],
    eval_datasets: dict[str, list[dict]],
    n_eval: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    output_dir: str,
    model_name: str,
) -> None:
    """Evaluate one checkpoint across all (k, src_lang, tgt_lang) combos and write per-checkpoint files."""
    pred_path = os.path.join(output_dir, f"{model_name}_{label}_predictions.csv")
    gen_path = os.path.join(output_dir, f"{model_name}_{label}_generations.jsonl")
    summary_path = os.path.join(output_dir, f"{model_name}_{label}_summary.csv")

    if os.path.exists(pred_path):
        existing_df = pd.read_csv(pred_path)
        done_set: set[tuple] = {(r["draw_i"], r["k"], r["src_lang"], r["tgt_lang"]) for r in existing_df.to_dict(orient="records")}
        logger.info(f"  Resuming from {len(existing_df)} existing rows in {pred_path}")
    else:
        done_set = set()

    total_experiments = len(k_values) * len(pairs)
    completed_experiments = sum(
        1 for k in k_values for src_lang, tgt_lang in pairs
        if len({t[0] for t in done_set if t[1:] == (k, src_lang, tgt_lang)}) == n_eval
    )
    loop_start = time.monotonic()

    for k in k_values:
        draws_k = list(zip(ctx_draws[k], test_indices))
        for src_lang, tgt_lang in pairs:
            src_lang_idx = languages.index(src_lang)
            done_draw_is = {t[0] for t in done_set if t[1:] == (k, src_lang, tgt_lang)}
            n_pending = n_eval - len(done_draw_is)

            if n_pending == 0:
                logger.info(f"  [k={k}] [{src_lang}→{tgt_lang}] Already complete.")
                continue

            if done_draw_is:
                logger.info(f"  [k={k}] [{src_lang}→{tgt_lang}] Resuming: {len(done_draw_is)} done, {n_pending} remaining.")

            pair_rows = generate_pair_rows(
                llm, src_lang, tgt_lang, src_lang_idx, k,
                draws_k, ctx_datasets, eval_datasets,
                done_draw_is, temperature, max_tokens, seed,
            )
            t0 = time.monotonic()
            write_header = not os.path.exists(pred_path)
            for row in pair_rows:
                row.pop("prompt")
            pd.DataFrame(pair_rows).to_csv(pred_path, mode="a", header=write_header, index=False)
            with open(gen_path, "a") as f:
                for row in pair_rows:
                    f.write(json.dumps({
                        "draw_i": row["draw_i"], "k": row["k"],
                        "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
                        "test_idx": row["test_idx"], "response": row["response"],
                        "gold_label_id": row["gold_label_id"],
                    }, ensure_ascii=False) + "\n")
            logger.debug(f"    [timing] file write:    {time.monotonic() - t0:.2f}s")
            done_set.update((r["draw_i"], r["k"], r["src_lang"], r["tgt_lang"]) for r in pair_rows)

            completed_experiments += 1
            elapsed = time.monotonic() - loop_start
            remaining = total_experiments - completed_experiments
            eta_s = (elapsed / completed_experiments) * remaining if completed_experiments > 0 else 0
            h, m = divmod(int(eta_s), 3600)
            m, s = divmod(m, 60)
            eta_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
            acc = sum(r["correct"] for r in pair_rows) / len(pair_rows) if pair_rows else float("nan")
            logger.info(f"  [k={k}] [{src_lang}→{tgt_lang}] [{len(pair_rows)}/{n_pending}] acc={acc:.3f} saved. ETA: {eta_str} ({completed_experiments}/{total_experiments})")

    pred_df = pd.read_csv(pred_path)
    summary_df = (
        pred_df[pred_df["predicted_label_id"].notna()]
        .groupby(["k", "src_lang", "tgt_lang"], sort=True)
        .agg(mean_accuracy=("correct", "mean"), std_accuracy=("correct", "std"), n=("correct", "count"))
        .reset_index()
    )
    summary_df.to_csv(summary_path, index=False)

    for k in k_values:
        matrix = (
            summary_df[summary_df["k"] == k]
            .pivot(index="src_lang", columns="tgt_lang", values="mean_accuracy")
        )
        logger.info(f"\n[{label}] k={k} mean accuracy (rows=context lang, cols=eval lang):\n{matrix.to_string()}")

    logger.info(f"  Predictions  → {pred_path}")
    logger.info(f"  Generations  → {gen_path}")
    logger.info(f"  Summary      → {summary_path}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="XNLI cross-lingual ICL evaluation")
    parser.add_argument("--config", default="code/multilingual_transfer/configs/xnli.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    if isinstance(model_cfg, str):
        model_cfg = {"name": model_cfg}
    model_path = model_cfg["name"]
    model_name = model_path.rstrip("/").split("/")[-1]

    output_dir = config.get("output_dir", "results/xnli")
    languages = list(config["languages"].keys())
    gen = config.get("generation", {})
    temperature = gen.get("temperature", 0.0)
    max_tokens = gen.get("max_tokens", 16)
    seed = gen.get("seed", 42)
    hf_batch_size = gen.get("hf_batch_size", 8)
    icl = config.get("icl", {})
    k_raw = icl.get("k", 4)
    k_values: list[int] = k_raw if isinstance(k_raw, list) else [k_raw]
    n_eval = icl.get("n_eval", 500)
    context_split = icl.get("context_split", "validation")
    eval_split = icl.get("eval_split", "test")

    # --- Resolve checkpoints ---
    checkpoints = resolve_checkpoints(
        model_path,
        model_cfg.get("checkpoints"),
        branch_filter_pattern=model_cfg.get("branch_filter_pattern"),
    )
    checkpoints = apply_checkpoint_filters(
        checkpoints,
        checkpoint_step=model_cfg.get("checkpoint_step"),
        max_checkpoints=model_cfg.get("max_checkpoints"),
    )
    logger.info(f"Model: {model_name} | checkpoints: {len(checkpoints)} | k={k_values} | n_eval={n_eval} | {len(languages)} languages")

    # --- Pre-compute draws (ONCE — shared across all checkpoints) ---
    ref_lang = languages[0]
    logger.info(f"Loading reference {context_split} split [{ref_lang}]...")
    ref_ctx_ds = load_dataset("facebook/xnli", ref_lang, split=context_split)
    ref_eval_size = len(load_dataset("facebook/xnli", ref_lang, split=eval_split))

    by_label = precompute_by_label(ref_ctx_ds)

    logger.info(f"Sampling {n_eval} test indices...")
    test_indices = sample_test_indices(ref_eval_size, n_eval, seed)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([
        {"draw_i": i, "test_idx": test_indices[i]}
        for i in range(n_eval)
    ]).to_csv(os.path.join(output_dir, "test_indices.csv"), index=False)

    logger.info("Pre-computing context draws for all k values...")
    ctx_draws: dict[int, list[list[int]]] = {k: sample_ctx_for_k(by_label, k, n_eval, seed) for k in k_values}

    logger.info(f"Loading full {context_split} split for all languages...")
    ctx_datasets = load_full_split(languages, context_split)

    logger.info(f"Loading full {eval_split} split for all languages...")
    eval_datasets = load_full_split(languages, eval_split)

    # --- Checkpoint loop ---
    pairs = list(product(languages, repeat=2))
    logger.info(f"{len(pairs)} language pairs × {len(k_values)} k values = {len(pairs) * len(k_values)} experiments per checkpoint")

    run_start = time.monotonic()
    ckpt_times: list[float] = []

    for ckpt_idx, ckpt in enumerate(checkpoints):
        label = ckpt_label(ckpt)
        logger.info(
            f"\n{'='*60}\n"
            f"Checkpoint {ckpt_idx + 1}/{len(checkpoints)}: {label}\n"
            f"{'='*60}"
        )

        # Skip if this checkpoint is fully done.
        pred_path = os.path.join(output_dir, f"{model_name}_{label}_predictions.csv")
        if os.path.exists(pred_path):
            existing_df = pd.read_csv(pred_path)
            done_set = {(r["draw_i"], r["k"], r["src_lang"], r["tgt_lang"]) for r in existing_df.to_dict(orient="records")}
            already_done = sum(
                1 for k in k_values for src_lang, tgt_lang in pairs
                if len({t[0] for t in done_set if t[1:] == (k, src_lang, tgt_lang)}) == n_eval
            )
            if already_done == len(k_values) * len(pairs):
                logger.info(f"Checkpoint {label}: all experiments complete, skipping model load.")
                continue

        ckpt_start = time.monotonic()
        logger.info(f"Loading model (revision={label!r})...")
        llm = load_model(model_path, revision=ckpt, seed=seed, hf_batch_size=hf_batch_size)

        run_checkpoint(
            llm=llm,
            label=label,
            pairs=pairs,
            k_values=k_values,
            ctx_draws=ctx_draws,
            test_indices=test_indices,
            languages=languages,
            ctx_datasets=ctx_datasets,
            eval_datasets=eval_datasets,
            n_eval=n_eval,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            output_dir=output_dir,
            model_name=model_name,
        )

        logger.info(f"Unloading model for checkpoint {label}...")
        unload_model(llm)

        ckpt_elapsed = time.monotonic() - ckpt_start
        ckpt_times.append(ckpt_elapsed)
        total_elapsed = time.monotonic() - run_start
        done_count = len(ckpt_times)
        remaining_count = len(checkpoints) - ckpt_idx - 1
        avg = total_elapsed / done_count
        eta = avg * remaining_count
        logger.info(
            f"Checkpoint {label} done | took {_fmt(ckpt_elapsed)}"
            f" | total elapsed {_fmt(total_elapsed)}"
            f" | ~{_fmt(eta)} remaining ({remaining_count} checkpoint(s) @ avg {_fmt(avg)} each)"
        )

    logger.info(f"\nAll checkpoints done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
