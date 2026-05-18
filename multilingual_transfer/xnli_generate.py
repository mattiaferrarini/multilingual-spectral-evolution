"""
XNLI Cross-Lingual ICL Evaluation
Tests multilingual transfer between all language pairs via in-context learning.
Designed for pre-trained (non-instruction-tuned) models: uses raw completion, no chat template.

For each of n_eval independent draws:
  - k context examples are sampled from the validation split (balanced labels, without replacement)
  - 1 test example is sampled from the test split (i.i.d. across draws)
  - All (src_lang, tgt_lang) pairs are evaluated under the same draw

Outputs a single tall-format CSV with one row per prediction.
Mean and std accuracy per (k, src_lang, tgt_lang) are reported in a separate summary CSV.

Usage: python xnli_generate.py --config multilingual_transfer/configs/xnli.yaml
"""

import os
import re
import json
import time
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from datasets import load_dataset
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def sample_draws(
    by_label: dict[int, list[int]],
    eval_size: int,
    max_k: int,
    n_eval: int,
    seed: int,
) -> list[tuple[list[int], int]]:
    """
    Pre-generate n_eval independent (ctx_indices, test_idx) draws.
    ctx_indices: max_k validation indices sampled with balanced labels (without replacement per draw).
    test_idx: one test set index sampled i.i.d. across draws.
    Separate RNG streams for context and eval keep them independent.
    """
    rng_ctx = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 1)
    draws = []
    for _ in range(n_eval):
        ctx_indices = _sample_one_balanced_context(by_label, max_k, rng_ctx) if max_k > 0 else []
        test_idx = int(rng_eval.integers(0, eval_size))
        draws.append((ctx_indices, test_idx))
    return draws


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

    def __init__(self, model_path: str, seed: int = 42, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id or self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.seed = seed
        self.batch_size = batch_size

    def generate(self, prompts: list[str], sampling_params=None, **kwargs) -> list[_RequestOutput]:
        import torch
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


def load_model(model_path: str, seed: int = 42, hf_batch_size: int = 8) -> LLM | HFModel:
    try:
        return LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", seed=seed)
    except Exception as e1:
        try:
            return LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", seed=seed, model_impl="transformers")
        except Exception as e2:
            logger.warning(f"vLLM failed ({e1} | {e2}), falling back to HuggingFace.")
            return HFModel(model_path, seed=seed, batch_size=hf_batch_size)


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

    prompts = []
    for draw_i, (ctx_indices, test_idx) in pending:
        # Slice to k for balance, then shuffle to remove the interleaved label pattern.
        # Seed is unique per (draw, k, src_lang) so order is deterministic.
        sliced = list(ctx_indices[:k])
        np.random.default_rng(seed + 2 + draw_i * 10000 + k * 1000 + src_lang_idx).shuffle(sliced)
        ctx_examples = [ctx_datasets[src_lang][j] for j in sliced]
        test_example = eval_datasets[tgt_lang][test_idx]
        prompts.append(build_icl_prompt(ctx_examples, test_example))

    responses = generate_completions(llm, prompts, temperature=temperature, max_tokens=max_tokens)
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
            "premise": test_example["premise"],
            "hypothesis": test_example["hypothesis"],
            "gold_label": LABEL_MAP[test_example["label"]],
            "gold_label_id": test_example["label"],
            "prompt": prompt,
            "response": response,
            "predicted_label": LABEL_MAP.get(predicted_id, "unknown"),
            "predicted_label_id": predicted_id,
            "correct": predicted_id == test_example["label"],
        })
    return rows


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="XNLI cross-lingual ICL evaluation")
    parser.add_argument("--config", default="multilingual_transfer/configs/xnli.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_path = config["model"]
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
    max_k = max(k_values) if k_values else 0
    n_eval = icl.get("n_eval", 500)
    context_split = icl.get("context_split", "validation")
    eval_split = icl.get("eval_split", "test")

    model_name = model_path.rstrip("/").split("/")[-1]
    logger.info(f"Model: {model_name} | k={k_values} | n_eval={n_eval} | {len(languages)} languages")

    # Use one reference language to determine dataset sizes and label distribution.
    # Labels are identical across all XNLI languages, so any language works.
    ref_lang = languages[0]
    logger.info(f"Loading reference {context_split} split [{ref_lang}]...")
    ref_ctx_ds = load_dataset("facebook/xnli", ref_lang, split=context_split)
    ref_eval_size = len(load_dataset("facebook/xnli", ref_lang, split=eval_split))

    by_label = precompute_by_label(ref_ctx_ds)

    logger.info(f"Sampling {n_eval} draws (max_k={max_k})...")
    draws = sample_draws(by_label, ref_eval_size, max_k, n_eval, seed)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([
        {"draw_i": i, "ctx_indices": str(d[0]), "test_idx": d[1]}
        for i, d in enumerate(draws)
    ]).to_csv(os.path.join(output_dir, "draws.csv"), index=False)

    logger.info(f"Loading full {context_split} split for all languages...")
    ctx_datasets = load_full_split(languages, context_split)

    logger.info(f"Loading full {eval_split} split for all languages...")
    eval_datasets = load_full_split(languages, eval_split)

    logger.info("Loading model...")
    llm = load_model(model_path, seed=seed, hf_batch_size=hf_batch_size)

    pairs = list(product(languages, repeat=2))
    logger.info(f"{len(pairs)} language pairs × {len(k_values)} k values = {len(pairs) * len(k_values)} experiments")

    pred_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    gen_path = os.path.join(output_dir, f"{model_name}_generations.json")
    if os.path.exists(pred_path):
        all_rows = pd.read_csv(pred_path).to_dict(orient="records")
        logger.info(f"Resuming from {len(all_rows)} existing rows in {pred_path}")
    else:
        all_rows = []
    if os.path.exists(gen_path) and os.path.exists(pred_path):
        with open(gen_path) as f:
            gen_rows = json.load(f)
        logger.info(f"Loaded {len(gen_rows)} existing generation records from {gen_path}")
    else:
        gen_rows = []

    done_set: set[tuple] = {(r["draw_i"], r["k"], r["src_lang"], r["tgt_lang"]) for r in all_rows}

    total_experiments = len(k_values) * len(pairs)
    completed_experiments = sum(
        1 for k in k_values for src_lang, tgt_lang in pairs
        if len({t[0] for t in done_set if t[1:] == (k, src_lang, tgt_lang)}) == n_eval
    )
    loop_start = time.monotonic()

    for k in k_values:
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
                draws, ctx_datasets, eval_datasets,
                done_draw_is, temperature, max_tokens, seed,
            )
            for row in pair_rows:
                gen_rows.append({
                    "draw_i": row["draw_i"], "k": row["k"],
                    "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
                    "test_idx": row["test_idx"], "prompt": row.pop("prompt"),
                    "response": row["response"], "gold_label_id": row["gold_label_id"],
                })
            all_rows.extend(pair_rows)
            done_set.update((r["draw_i"], r["k"], r["src_lang"], r["tgt_lang"]) for r in pair_rows)
            pd.DataFrame(all_rows).to_csv(pred_path, index=False)
            with open(gen_path, "w") as f:
                json.dump(gen_rows, f, ensure_ascii=False, indent=2)

            completed_experiments += 1
            elapsed = time.monotonic() - loop_start
            remaining = total_experiments - completed_experiments
            eta_s = (elapsed / completed_experiments) * remaining
            h, m = divmod(int(eta_s), 3600)
            m, s = divmod(m, 60)
            eta_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
            acc = sum(r["correct"] for r in pair_rows) / len(pair_rows) if pair_rows else float("nan")
            logger.info(f"  [k={k}] [{src_lang}→{tgt_lang}] [{len(pair_rows)}/{n_pending}] acc={acc:.3f} saved. ETA: {eta_str} ({completed_experiments}/{total_experiments})")

    pred_df = pd.DataFrame(all_rows)

    summary_df = (
        pred_df[pred_df["predicted_label_id"].notna()]
        .groupby(["k", "src_lang", "tgt_lang"], sort=True)
        .agg(mean_accuracy=("correct", "mean"), std_accuracy=("correct", "std"), n=("correct", "count"))
        .reset_index()
    )
    summary_path = os.path.join(output_dir, f"{model_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    for k in k_values:
        matrix = (
            summary_df[summary_df["k"] == k]
            .pivot(index="src_lang", columns="tgt_lang", values="mean_accuracy")
        )
        logger.info(f"\nk={k} mean accuracy (rows=context lang, cols=eval lang):\n{matrix.to_string()}")

    logger.info(f"\nPredictions  → {pred_path}")
    logger.info(f"Generations  → {gen_path}")
    logger.info(f"Summary      → {summary_path}")


if __name__ == "__main__":
    main()
