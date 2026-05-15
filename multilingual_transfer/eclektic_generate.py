"""
ECLeKTic Generation Script
Runs local HuggingFace models via vLLM and saves responses as JSON files.
Run this on the cluster, then use ekclektic_score.py to judge the outputs.

See https://arxiv.org/abs/2502.21228 for dataset details.
"""

import os
import argparse
import yaml
import pandas as pd
from vllm import LLM, SamplingParams
from dotenv import load_dotenv


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


def load_model(model_path: str) -> LLM:
    return LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")


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
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    return [[o.outputs[i].text for i in range(n)] for o in outputs]


def generate_for_model(
    model_path: str,
    data: pd.DataFrame,
    output_dir: str,
    temperature: float = 1.0,
    n_generations: int = 1,
) -> str:
    model_name = model_path.rstrip("/").split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Generating: {model_name}  ({model_path})")
    print(f"{'='*60}")

    llm = load_model(model_path)
    raw_responses = generate_batch(llm, data["question"].tolist(), temperature=temperature, n=n_generations)
    del llm

    rows = []
    for (_, qrow), generations in zip(data.iterrows(), raw_responses):
        for gen_idx, response in enumerate(generations):
            rows.append({
                "q_id": qrow["q_id"],
                "lang": qrow["lang"],
                "original_lang": qrow["original_lang"],
                "generation_idx": gen_idx,
                "response": response,
                "question": qrow["question"],
                "answer": qrow["answer"],
                "content": qrow["content"],
            })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_generations.json")
    pd.DataFrame(rows).to_json(out_path, orient="records", indent=2)
    print(f"Saved {len(rows)} rows to {out_path}")
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

    print("Loading data...")
    with open(data_path) as f:
        data = pd.read_json(f, lines=True, orient="records")
    data = separate_example_per_lang(data, languages)
    if max_questions is not None:
        data = data.head(max_questions * len(languages))

    for model_path in model_paths:
        generate_for_model(model_path, data, output_dir, temperature=temperature, n_generations=n_generations)


if __name__ == "__main__":
    main()
