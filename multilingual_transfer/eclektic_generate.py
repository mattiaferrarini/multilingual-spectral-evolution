"""
ECLeKTic Generation Script
Runs local HuggingFace models via vLLM and saves responses as JSON files.
Run this on the cluster, then use ekclektic_score.py to judge the outputs.

See https://arxiv.org/abs/2502.21228 for dataset details.
"""

import os
import logging
import argparse
import yaml
import pandas as pd
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def separate_example_per_lang(df, languages):
    new_examples = []
    for _, row in df.iterrows():
        for lang in languages:
            new_ex = {
                "q_id": row["q_id"],
                "original_lang": row["original_lang"],
                "lang": lang,
                "title": row["title"],
                "url": row["url"],
                "orig_content": row["content"],
                "orig_question": row["question"],
                "orig_answer": row["answer"],
                "question": row[f"{lang}_q"],
                "answer": row[f"{lang}_a"],
                "content": row[f"{lang}_c"],
            }
            new_examples.append(new_ex)
    return pd.DataFrame(new_examples)


def load_model(model_path: str, seed: int | None = None) -> LLM:
    try:
        return LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", seed=seed if seed is not None else 42)
    except Exception:
        logger.warning(f"⚠️ Native vLLM backend failed for {model_path}, falling back to transformers backend.")
        return LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", seed=seed if seed is not None else 42, model_impl="transformers")


def generate_batch(
    llm: LLM,
    questions: list[str],
    sys: str = "",
    temperature: float = 1.0,
    n: int = 1,
    max_tokens: int = 512,
) -> list[list[str]]:
    """Batch-generate responses. Returns list[questions] x list[generations]."""
    conversations = []
    for q in questions:
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": q})
        conversations.append(messages)

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
    return [[o.outputs[i].text for i in range(n)] for o in outputs]


def load_existing_results(out_path: str, n_generations: int) -> tuple[list[dict], set[tuple]]:
    """Load existing results; return (rows, done_pairs) where done_pairs are fully-generated (q_id, lang) tuples."""
    if not os.path.exists(out_path):
        return [], set()

    existing = pd.read_json(out_path, orient="records").to_dict(orient="records")
    counts: dict[tuple, int] = {}
    for row in existing:
        key = (row["q_id"], row["lang"])
        counts[key] = counts.get(key, 0) + 1

    if counts and max(counts.values()) != n_generations:
        logger.warning(
            f"Existing file has {max(counts.values())} generation(s) per question "
            f"but n_generations={n_generations}. Wiping file and starting fresh."
        )
        os.remove(out_path)
        return [], set()

    done_pairs = {k for k, v in counts.items() if v >= n_generations}
    return existing, done_pairs


def generate_for_model(
    model_path: str,
    data: pd.DataFrame,
    output_dir: str,
    temperature: float = 1.0,
    n_generations: int = 1,
    chunk_size: int = 50,
    seed: int | None = None,
) -> str:
    model_name = model_path.rstrip("/").split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_generations.json")

    logger.info(f"Generating: {model_name}  ({model_path})")

    existing_rows, done_pairs = load_existing_results(out_path, n_generations)
    pending = data[~data.apply(lambda r: (r["q_id"], r["lang"]) in done_pairs, axis=1)]

    if pending.empty:
        logger.info(f"All {len(data)} questions already generated. Skipping.")
        return out_path

    if done_pairs:
        logger.info(f"Resuming: {len(done_pairs)} done, {len(pending)} remaining.")

    all_rows = list(existing_rows)
    llm = load_model(model_path, seed=seed)

    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending.iloc[chunk_start : chunk_start + chunk_size]
        raw_responses = generate_batch(llm, chunk["question"].tolist(), temperature=temperature, n=n_generations)

        for (_, qrow), generations in zip(chunk.iterrows(), raw_responses):
            for gen_idx, response in enumerate(generations):
                all_rows.append({
                    "q_id": qrow["q_id"],
                    "lang": qrow["lang"],
                    "original_lang": qrow["original_lang"],
                    "generation_idx": gen_idx,
                    "response": response,
                    "question": qrow["question"],
                    "answer": qrow["answer"],
                    "content": qrow["content"],
                })

        pd.DataFrame(all_rows).to_json(out_path, orient="records", indent=2)
        logger.info(f"[{chunk_start + len(chunk)}/{len(pending)}] saved to {out_path}")

    del llm
    logger.info(f"Done. {len(all_rows)} total rows in {out_path}")
    return out_path


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ECLeKTic generation with local vLLM models")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_paths = config["models"]
    data_path = config.get("data_path", "/kaggle/input/eclektic/eclektic_main.jsonl")
    max_questions = config.get("max_questions", None)
    output_dir = config.get("output_dir", "results")
    languages = config["languages"]
    temperature = config.get("temperature", 1.0)
    n_generations = config.get("n_generations", 1)
    chunk_size = config.get("chunk_size", 50)
    seed = config.get("seed", None)

    logger.info("Loading data...")
    with open(data_path) as f:
        data = pd.read_json(f, lines=True, orient="records")
    data = separate_example_per_lang(data, languages)
    if max_questions is not None:
        data = data.head(max_questions * len(languages))

    for model_path in model_paths:
        generate_for_model(model_path, data, output_dir, temperature=temperature, n_generations=n_generations, chunk_size=chunk_size, seed=seed)


if __name__ == "__main__":
    main()
