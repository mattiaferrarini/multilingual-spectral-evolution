"""
ECLeKTic Scoring Script
Loads generation JSON files produced by eclektic_generate.py and judges them
using OpenAI models with majority voting. Saves raw per-judge judgments to CSV.

Output: {model_name}_judgments.csv per model, with one row per
(q_id, lang, generation_idx) and columns correct_{judge} and majority-vote correct.

See https://arxiv.org/abs/2502.21228 for further details.
"""

import os
import glob
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

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
        evals.append([row["q_id"], row["lang"], row["generation_idx"], eval_model, is_correct])

    return pd.DataFrame(data=evals, columns=["q_id", "lang", "generation_idx", "judge", "correct"])


def score_model(generations_path: str, output_dir: str, languages: dict, eval_models: list):
    model_name = os.path.basename(generations_path).replace("_generations.json", "")
    print(f"\n{'='*60}")
    print(f"Scoring: {model_name}")
    print(f"{'='*60}")

    eval_data = pd.read_json(generations_path, orient="records")
    original_lang = eval_data[["q_id", "lang", "generation_idx", "original_lang"]].drop_duplicates()

    all_evals = pd.concat([eval_one(eval_data, m, languages) for m in eval_models], ignore_index=True)
    judgments = pd.merge(all_evals, original_lang, on=["q_id", "lang", "generation_idx"])

    os.makedirs(output_dir, exist_ok=True)
    judgments_path = os.path.join(output_dir, f"{model_name}_judgments.csv")
    judgments[["q_id", "lang", "generation_idx", "original_lang", "judge", "correct"]].to_csv(judgments_path, index=False)
    print(f"  Saved to {judgments_path}")


def main():
    parser = argparse.ArgumentParser(description="ECLeKTic scoring with LLM judge models")
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

    for gen_file in generation_files:
        score_model(gen_file, output_dir, languages, eval_models)


if __name__ == "__main__":
    main()
