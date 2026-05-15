"""
ECLeKTic Scoring Script
Loads generation JSON files produced by ekclektic_generate.py and judges them
using OpenAI models with majority voting. Outputs per-model result CSVs and a summary.

Outputs:
  - Overall score: main score measuring success over the QA task and
    the model's ability to transfer knowledge across languages.
  - Transfer score: focuses only on knowledge transfer, does not
    penalize for wrong answers in source languages.

See https://arxiv.org/abs/2502.21228 for further details.
"""

import os
import glob
import argparse
import yaml
import pandas as pd
from functools import reduce
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
swissai_api_key = os.environ.get("CSCS_SERVING_API", "")
SWISSAI_API_URL = "https://api.swissai.cscs.ch/v1"

_openai_client = None
_swissai_client = None


def openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=openai_api_key)
    return _openai_client


def swissai_client():
    global _swissai_client
    if _swissai_client is None:
        _swissai_client = OpenAI(api_key=swissai_api_key, base_url=SWISSAI_API_URL)
    return _swissai_client


def generate_api(model, prompt):
    client = openai_client() if "gpt" in model else swissai_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def eval_prompt(question, content, predicted_answer, lang, languages):
    prompt = """\
**Task:** Determine if an answer to the question is supported by a given text.

**Input (in {target_language}):**
- Text
- Question
- Answer

**Single Word Output (in English):**
- YES: Answer is derived from the text.
- NO: Answer is not derived from the text.

Text:
{context}

Question:
{question}

Answer:
{predicted_answer}

Output:
"""
    return prompt.format(
        context=content,
        question=question,
        target_language=languages[lang],
        predicted_answer=predicted_answer,
    )


def eval_one(examples, eval_model, languages):
    evals = []
    for _, row in tqdm(examples.iterrows(), total=len(examples), desc=f"judge={eval_model}"):
        prompt = eval_prompt(
            row["question"], row["content"], row["response"], row["lang"], languages
        )
        response = generate_api(eval_model, prompt)
        is_correct = "yes" in response.lower()
        evals.append([row["q_id"], row["lang"], row["generation_idx"], is_correct])

    return pd.DataFrame(data=evals, columns=["q_id", "lang", "generation_idx", f"correct_{eval_model}"])


def _compute_metrics(eval_data: pd.DataFrame) -> dict:
    correct_in_lang_qids = set(
        eval_data[(eval_data["correct"]) & (eval_data["lang"] == eval_data["original_lang"])]["q_id"].tolist()
    )
    scored_data = eval_data[eval_data["lang"] != eval_data["original_lang"]]
    successes = (
        (scored_data["correct"]) & (scored_data["q_id"].isin(correct_in_lang_qids))
    ).tolist()

    overall_score = sum(successes) / len(scored_data)

    transfer_data = eval_data[eval_data["q_id"].isin(correct_in_lang_qids)]
    transfer_score = (
        ((transfer_data["correct"]) & (transfer_data["q_id"].isin(correct_in_lang_qids))).sum()
        / len(transfer_data)
        if len(transfer_data) > 0
        else 0.0
    )
    return {"overall_score": overall_score, "transfer_score": transfer_score}


def score_model(generations_path: str, output_dir: str, languages: dict, eval_models: list) -> dict:
    model_name = os.path.basename(generations_path).replace("_generations.json", "")
    print(f"\n{'='*60}")
    print(f"Scoring: {model_name}")
    print(f"{'='*60}")

    eval_data = pd.read_json(generations_path, orient="records")
    n_generations = eval_data["generation_idx"].max() + 1

    merge_keys = ["q_id", "lang", "generation_idx"]
    all_evals = [eval_one(eval_data, m, languages) for m in eval_models]
    all_evals = reduce(lambda left, right: pd.merge(left, right, on=merge_keys), all_evals)

    correct_columns = [f"correct_{m}" for m in eval_models]
    all_evals["correct"] = all_evals[correct_columns].sum(axis=1) >= (len(eval_models) // 2 + 1)
    all_evals.drop(columns=correct_columns, inplace=True)

    eval_data = pd.merge(all_evals, eval_data, on=merge_keys)

    per_gen = [_compute_metrics(eval_data[eval_data["generation_idx"] == g]) for g in range(n_generations)]
    overall_scores = [m["overall_score"] for m in per_gen]
    transfer_scores = [m["transfer_score"] for m in per_gen]

    overall_score = float(np.mean(overall_scores))
    overall_std = float(np.std(overall_scores, ddof=1)) if n_generations > 1 else None
    transfer_score = float(np.mean(transfer_scores))
    transfer_std = float(np.std(transfer_scores, ddof=1)) if n_generations > 1 else None

    print(f"\nResults for {model_name}:")
    if overall_std is not None:
        print(f"  Overall score:   {overall_score:.4f} ±{overall_std:.4f} (std over {n_generations} generations)")
        print(f"  Transfer score:  {transfer_score:.4f} ±{transfer_std:.4f}")
    else:
        print(f"  Overall score:   {overall_score:.4f}")
        print(f"  Transfer score:  {transfer_score:.4f}")

    question_results = (
        eval_data.groupby(["q_id", "lang", "original_lang"])
        .agg(n_yes=("correct", "sum"), n_total=("correct", "count"))
        .reset_index()
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_results.csv")
    question_results.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return {
        "model": model_name,
        "overall_score": overall_score,
        "overall_std": overall_std,
        "transfer_score": transfer_score,
        "transfer_std": transfer_std,
    }


def main():
    parser = argparse.ArgumentParser(description="ECLeKTic scoring with OpenAI judge model")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--generations_dir",
        default=None,
        help="Directory containing *_generations.json files (defaults to config output_dir)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    languages = config["languages"]
    eval_models = config["eval_models"]
    output_dir = config.get("output_dir", "results")
    generations_dir = args.generations_dir or output_dir

    generation_files = sorted(glob.glob(os.path.join(generations_dir, "*_generations.json")))
    if not generation_files:
        raise FileNotFoundError(f"No *_generations.json files found in {generations_dir}")

    print(f"Found {len(generation_files)} generation file(s):")
    for f in generation_files:
        print(f"  {f}")

    summary_rows = []
    for gen_file in generation_files:
        row = score_model(gen_file, output_dir, languages, eval_models)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
