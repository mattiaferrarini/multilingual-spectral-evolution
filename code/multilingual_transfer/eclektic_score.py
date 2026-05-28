"""
ECLeKTic Scoring Script
Loads generation JSON files produced by eclektic_generate.py and judges them
using OpenAI models with majority voting. Saves raw per-judge judgments to CSV.

Output: {model_name}_judgments.csv per model, with one row per
(q_id, lang, generation_idx) and columns correct_{judge} and majority-vote correct.

See https://arxiv.org/abs/2502.21228 for further details.

eval_method config: list of methods to run, any combination of:
  llm    – LLM-as-judge
  string – string-recall-based transfer metric
Example: eval_method: [llm, string]
"""

import os
import glob
import logging
import argparse
from math import sqrt

import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# String-recall metric helpers
# ---------------------------------------------------------------------------

def _get_confidence_margin(data, weights=None):
    if not weights:
        weights = [1] * len(data)
    p_hat = sum(data) / sum(weights)
    norm_sq = sum(w ** 2 for w in weights)
    n = sum(weights) ** 2 / norm_sq
    z = norm.ppf(0.975)
    return z * sqrt((p_hat * (1 - p_hat)) / n)


def _is_chinese_char(char):
    return (
        '一' <= char <= '鿿'
        or '㐀' <= char <= '䶿'
        or '\U00020000' <= char <= '\U0002A6DF'
        or '\U0002A700' <= char <= '\U0002B73F'
        or '\U0002B740' <= char <= '\U0002B81F'
        or '\U0002B820' <= char <= '\U0002CEAF'
        or '\U0002CEB0' <= char <= '\U0002EBEF'
        or '\U00030000' <= char <= '\U0003134F'
    )


def _word_recall(gold, prediction, lang):
    if lang in {'zh', 'ja'}:
        words, buf = [], []
        for ch in gold:
            if _is_chinese_char(ch):
                if buf:
                    words.append(''.join(buf)); buf = []
                words.append(ch)
            else:
                buf.append(ch)
        if buf:
            words.append(''.join(buf))
    else:
        words = gold.split()
    return len([w for w in words if w in prediction]) / len(words) if words else 0.0


def compute_string_metrics(df: pd.DataFrame) -> dict:
    """Return transfer_score, transfer_margin, overall_score, overall_margin.

    df must have columns: q_id, original_language, target_language, answer, prediction.
    """
    in_lang = {}
    for _, row in df.iterrows():
        if row['original_language'] == row['target_language']:
            in_lang[row['q_id']] = _word_recall(
                row['answer'], row['prediction'], row['target_language']
            )

    cl_results = []
    for _, row in df.iterrows():
        if row['original_language'] != row['target_language']:
            cl_recall = _word_recall(row['answer'], row['prediction'], row['target_language'])
            cl_results.append(cl_recall * in_lang.get(row['q_id'], 0.0))

    # number of cross-lingual languages per question (derived from data)
    n_cl_langs = len(cl_results) // len(in_lang) if in_lang else 1

    overall_score = float(np.mean(cl_results)) if cl_results else 0.0
    overall_margin = _get_confidence_margin(cl_results)

    denom = sum(in_lang.values()) * n_cl_langs
    transfer_score = sum(cl_results) / denom if denom else 0.0
    transfer_margin = _get_confidence_margin(cl_results, weights=list(in_lang.values()) * n_cl_langs)

    return {
        "transfer_score": transfer_score,
        "transfer_margin": transfer_margin,
        "overall_score": overall_score,
        "overall_margin": overall_margin,
    }


def compute_string_metrics_per_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Return transfer_score, transfer_margin, overall_score, overall_margin per (original_language, target_language) pair."""
    in_lang_rows = df[df['original_language'] == df['target_language']]
    langs_with_in_lang = set(in_lang_rows['original_language'].unique())
    rows = []
    for (orig, tgt), group in df.groupby(['original_language', 'target_language']):
        if orig == tgt or orig not in langs_with_in_lang:
            continue
        pair_q_ids = group['q_id'].unique()
        relevant_in_lang = in_lang_rows[in_lang_rows['q_id'].isin(pair_q_ids)]
        pair_df = pd.concat([relevant_in_lang, group], ignore_index=True)
        metrics = compute_string_metrics(pair_df)
        rows.append({'original_language': orig, 'target_language': tgt, **metrics})
    return pd.DataFrame(rows)


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


def score_llm(generations_path: str, output_dir: str, languages: dict, eval_models: list):
    model_name = os.path.basename(generations_path).replace("_generations.json", "")
    logger.info(f"LLM-judge scoring: {model_name}")

    eval_data = pd.read_json(generations_path, orient="records")
    original_lang = eval_data[["q_id", "lang", "generation_idx", "original_lang"]].drop_duplicates()

    all_evals = pd.concat([eval_one(eval_data, m, languages) for m in eval_models], ignore_index=True)
    judgments = pd.merge(all_evals, original_lang, on=["q_id", "lang", "generation_idx"])

    os.makedirs(output_dir, exist_ok=True)
    judgments_path = os.path.join(output_dir, f"{model_name}_judgments.csv")
    judgments[["q_id", "lang", "generation_idx", "original_lang", "judge", "correct"]].to_csv(judgments_path, index=False)
    logger.info(f"Saved to {judgments_path}")


def score_string(generations_path: str, output_dir: str):
    model_name = os.path.basename(generations_path).replace("_generations.json", "")
    logger.info(f"String-metric scoring: {model_name}")

    df = pd.read_json(generations_path, orient="records")
    df = df.rename(columns={"original_lang": "original_language", "lang": "target_language", "response": "prediction"})

    metrics = compute_string_metrics(df)
    logger.info(
        f"Transfer: {metrics['transfer_score']:.4f} ± {metrics['transfer_margin']:.4f} | "
        f"Overall: {metrics['overall_score']:.4f} ± {metrics['overall_margin']:.4f}"
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_string_scores.csv")
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    pair_metrics = compute_string_metrics_per_pair(df)
    pair_metrics.insert(0, 'model', model_name)
    pair_out_path = os.path.join(output_dir, f"{model_name}_string_scores_per_pair.csv")
    pair_metrics.to_csv(pair_out_path, index=False)
    logger.info(f"Saved per-pair scores to {pair_out_path}")


def score_model(generations_path: str, output_dir: str, languages: dict, eval_models: list, eval_method: list):
    if "llm" in eval_method:
        score_llm(generations_path, output_dir, languages, eval_models)
    if "string" in eval_method:
        score_string(generations_path, output_dir)


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
    eval_models = config.get("eval_models", [])
    output_dir = config.get("output_dir", "results")
    eval_method = config.get("eval_method", "llm")
    generations_dir = args.generations_dir or output_dir

    valid = {"llm", "string"}
    if not isinstance(eval_method, list) or not eval_method or not set(eval_method).issubset(valid):
        raise ValueError(f"eval_method must be a non-empty list of {valid}; got '{eval_method}'")
    if "llm" in eval_method and not eval_models:
        raise ValueError("eval_models must be set in config when eval_method includes 'llm'")

    generation_files = sorted(glob.glob(os.path.join(generations_dir, "*_generations.json")))
    if not generation_files:
        raise FileNotFoundError(f"No *_generations.json files found in {generations_dir}")

    logger.info(f"Found {len(generation_files)} generation file(s): {generation_files}")

    for gen_file in generation_files:
        score_model(gen_file, output_dir, languages, eval_models, eval_method)


if __name__ == "__main__":
    main()
