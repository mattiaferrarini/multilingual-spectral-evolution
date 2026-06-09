#!/usr/bin/env python3
"""
Downstream evaluation pipeline using lm-evaluation-harness.

Evaluates one model checkpoint on all benchmarks defined in configs/benchmarks.yaml
(m-MMLU 5-shot, XCOPA 0-shot, Belebele 5-shot, M-ARC 5-shot), saving one JSON per
(task, language) immediately after each evaluation to prevent data loss on cluster preemption.

Resume logic mirrors geometry_analysis/utils.py::load_existing_results:
scans output_dir at startup, builds the set of already-completed
(checkpoint, task, language) triples, and skips them in the main loop.

Usage:
    # Debug — one checkpoint, 5 examples per task
    python downstream_evaluation/evaluate.py \\
        --model TJUNLP/FuxiTranyu-8B \\
        --checkpoint 531B \\
        --output-dir results/eval \\
        --limit 5

    # Full checkpoint run
    python downstream_evaluation/evaluate.py \\
        --model swiss-ai/Apertus-8B-2509 \\
        --checkpoint step2627139-tokens15T \\
        --output-dir results/eval
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "benchmarks.yaml"


def _load_benchmarks(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["benchmarks"]


# ── Resume logic (mirrors geometry_analysis/utils.py::load_existing_results) ──

def load_completed_evals(output_dir: Path) -> set:
    """
    Scan output_dir for existing JSON result files and return the set of
    already-completed (checkpoint, task, language) triples.

    File layout: output_dir/{checkpoint}/{task}__{language}.json
    """
    completed = set()
    if not output_dir.exists():
        return completed
    for json_file in output_dir.rglob("*.json"):
        stem = json_file.stem
        if "__" not in stem:
            continue
        task, language = stem.split("__", 1)
        checkpoint = json_file.parent.name
        completed.add((checkpoint, task, language))
    logger.info(f"Resume: {len(completed)} already-completed (checkpoint, task, language) triples found.")
    return completed


def _safe_ckpt(checkpoint: str) -> str:
    """Make a checkpoint name safe for use as a filesystem directory name."""
    return checkpoint.replace("/", "_").replace(":", "_")


# ── Result persistence ────────────────────────────────────────────────────────

def save_result(results: dict, output_dir: Path, checkpoint: str, task: str, language: str):
    """Persist an lm-eval result dict to disk immediately after evaluation."""
    out_path = output_dir / _safe_ckpt(checkpoint) / f"{task}__{language}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Saved → {out_path}")


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract_metric(results: dict, metric_key: str = "acc,none") -> float | None:
    """Pull the target metric out of an lm-eval results dict."""
    for vals in results.get("results", {}).values():
        v = vals.get(metric_key)
        if v is not None:
            return float(v)
    return None


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_single_eval(lm_model, task_lm_name: str, num_fewshot: int, limit: int | None) -> dict | None:
    """
    Run lm-eval for one task on an already-loaded HFLM model.
    Returns the raw results dict, or None on failure.
    """
    try:
        import lm_eval
        return lm_eval.simple_evaluate(
            model=lm_model,
            tasks=[task_lm_name],
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=False,
        )
    except Exception as e:
        logger.error(f"    lm-eval failed for '{task_lm_name}': {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate one checkpoint on downstream benchmarks (resumable)."
    )
    p.add_argument("--model",      required=True,
                   help="HuggingFace model ID (e.g. TJUNLP/FuxiTranyu-8B)")
    p.add_argument("--checkpoint", required=True,
                   help="Branch / revision name (e.g. 531B or step2627139-tokens15T)")
    p.add_argument("--output-dir", default="results/eval",
                   help="Root directory for JSON outputs (default: results/eval).")
    p.add_argument("--config",     default=str(_DEFAULT_CONFIG),
                   help="Path to benchmarks YAML config (default: configs/benchmarks.yaml).")
    p.add_argument("--tasks",      nargs="+", default=None,
                   help="Benchmark groups to run. Default: all groups in config.")
    p.add_argument("--limit",      type=int, default=None,
                   help="Limit examples per task (use --limit 5 for a quick debug run).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ── Load benchmark config ──────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    benchmarks = _load_benchmarks(config_path)
    logger.info(f"Loaded benchmark config from {config_path}")

    task_names = args.tasks or list(benchmarks.keys())
    benchmarks = {k: v for k, v in benchmarks.items() if k in task_names}
    logger.info(f"Tasks: {list(benchmarks.keys())}")

    # ── Resume: find already-completed evaluations ─────────────────────────────
    safe = _safe_ckpt(args.checkpoint)
    completed = load_completed_evals(output_dir)

    pending = []
    for task, task_cfg in benchmarks.items():
        for language, lm_task_name in task_cfg["languages"].items():
            if (safe, task, language) in completed:
                logger.info(f"  Skip (done): {args.checkpoint} / {task} / {language}")
            else:
                pending.append((task, language, lm_task_name, task_cfg))

    if not pending:
        logger.info("All evaluations already complete for this checkpoint. Nothing to do.")
        return

    logger.info(f"{len(pending)} (task, language) pair(s) pending for checkpoint '{args.checkpoint}'.")

    # ── Load model once for all pending evaluations ────────────────────────────
    logger.info(f"Loading '{args.model}' revision='{args.checkpoint}' ...")
    try:
        from lm_eval.models.huggingface import HFLM
        lm_model = HFLM(
            pretrained=args.model,
            revision=args.checkpoint,
            dtype="bfloat16",
            trust_remote_code=args.trust_remote_code,
            batch_size=args.batch_size,
            device=args.device,
        )
    except ImportError:
        logger.error("lm_eval not installed. Run: pip install lm-eval")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    n_done = n_failed = 0
    for task, language, lm_task_name, task_cfg in pending:
        logger.info(
            f"[{task}] {language} ({lm_task_name})  "
            f"fewshot={task_cfg['num_fewshot']}  limit={args.limit}"
        )
        try:
            results = run_single_eval(lm_model, lm_task_name, task_cfg["num_fewshot"], args.limit)
            if results is None:
                n_failed += 1
                continue
            acc = extract_metric(results, task_cfg["metric"])
            logger.info(
                f"  {task_cfg['metric']} = {acc:.4f}" if acc is not None
                else "  [WARN] metric not found in output"
            )
            save_result(results, output_dir, safe, task, language)
            n_done += 1
        except MemoryError:
            logger.error(f"  OOM on {task}/{language} — skipping.")
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            n_failed += 1
        except Exception as e:
            logger.error(f"  Unexpected error on {task}/{language}: {e}")
            n_failed += 1

    logger.info(
        f"Done — {n_done} completed, {n_failed} failed out of {len(pending)} pending."
    )


if __name__ == "__main__":
    main()
