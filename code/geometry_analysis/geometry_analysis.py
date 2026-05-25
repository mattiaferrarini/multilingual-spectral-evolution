import gc
import os
import sys
import time
import yaml
import torch
import argparse
import logging
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import _fmt, _ckpt_label, resolve_checkpoints, load_model, load_existing_results
from dataset import prepare_dataset_samples
from activations import resolve_layer_indices, collect_activations
from metrics import compute_spectral_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_checkpoint(ckpt, run_idx, pending, n_total, model_name, model_cfg, coll, config, dataset_samples, completed_pairs, all_rows, results_path, layer_indices):
    """
    Process a single checkpoint: load the model, collect activations for each dataset, compute metrics, and save results.
    """
    ckpt_label = _ckpt_label(ckpt)
    n_pending = len(pending)
    logger.info(
        f"\n{'='*60}\n"
        f"Checkpoint {run_idx + 1}/{n_pending} pending ({ckpt_label})"
        f" — {n_total - n_pending + run_idx + 1}/{n_total} overall\n"
        f"{'='*60}"
    )
    ckpt_start = time.time()

    hf_model, tokenizer, num_layers = load_model(model_name, ckpt, model_cfg)

    if layer_indices is None:
        layer_indices = resolve_layer_indices(coll.get("layers", "last"), num_layers)
        logger.info(f"Layers to collect: {layer_indices}")

    aggregations = coll.get("aggregations", ["last"])
    if isinstance(aggregations, str):
        aggregations = [aggregations]
    max_seq_len = coll.get("max_sequence_length", 512)
    batch_size = int(coll.get("batch_size", 8))

    for ds_config in config["datasets"]:
        ds_name = ds_config["name"]
        if (ckpt_label, ds_name) in completed_pairs:
            logger.info(f"=== Dataset: {ds_name} — skipping (already complete) ===")
            continue
        samples = dataset_samples[ds_name]
        logger.info(f"=== Dataset: {ds_name} ({len(samples)} seqs) ===")
        activations_by_agg = collect_activations(
            hf_model, tokenizer, samples, layer_indices,
            max_seq_len, aggregations, batch_size,
            label=f"{ckpt_label} | {ds_name}",
            debug=config.get("debug", False),
        )
        for agg, activations in activations_by_agg.items():
            logger.info(f"--- Aggregation: {agg} ---")
            for layer_name, tensor in activations.items():
                logger.info(
                    f"Computing metrics: {ds_name} / {layer_name} / {agg}  shape={tuple(tensor.shape)}"
                )
                metrics = compute_spectral_metrics(tensor)
                metrics_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
                logger.info(f"  {metrics_str}")
                all_rows.append({
                    "checkpoint": ckpt_label,
                    "dataset": ds_name,
                    "layer": layer_name,
                    "aggregation": agg,
                    **metrics,
                })

        pd.DataFrame(all_rows).to_csv(results_path, index=False)
        logger.info(f"Results updated in {results_path}")

    del hf_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    ckpt_rows = [r for r in all_rows if r["checkpoint"] == ckpt_label]
    if ckpt_rows:
        df_ckpt = pd.DataFrame(ckpt_rows)
        base_cols = ["dataset", "layer", "aggregation"]
        metric_cols = [c for c in df_ckpt.columns if c not in ["checkpoint"] + base_cols]
        cols = base_cols + metric_cols
        summary = df_ckpt[cols]
        logger.info(f"\nMetrics for checkpoint '{ckpt_label}':\n{summary.to_string(index=False)}")

    return layer_indices, time.time() - ckpt_start, ckpt_label


def main(config):
    """
    Reads config, finds all the model checkpoints to analyze, prepares the datasets, and then processes each checkpoint one by one. 
    It resumes from existing results if it finds them.
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    checkpoints = resolve_checkpoints(
        model_name, model_cfg.get("checkpoints"),
        branch_filter_pattern=model_cfg.get("branch_filter_pattern"),
    )
    
    has_main = "main" in checkpoints
    if has_main:
        checkpoints.remove("main")

    checkpoint_step = model_cfg.get("checkpoint_step")
    if checkpoint_step is not None and int(checkpoint_step) > 1:
        checkpoint_step = int(checkpoint_step)
        before = len(checkpoints)
        checkpoints = checkpoints[::checkpoint_step]
        logger.info(f"checkpoint_step={checkpoint_step}: keeping {len(checkpoints)}/{before} checkpoints")
    
    max_checkpoints = model_cfg.get("max_checkpoints")
    if max_checkpoints is not None and len(checkpoints) > max_checkpoints:
        logger.info(f"Capping checkpoints from {len(checkpoints)} to {max_checkpoints} (max_checkpoints)")
        checkpoints = checkpoints[:max_checkpoints]
        
    if has_main:
        checkpoints.append("main")

    coll = config.get("collection", {})
    results_path = config.get("output", {}).get("results_path", "results/metrics.csv")
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    samples_dir = config.get("output", {}).get("samples_dir", "results/samples")
    if samples_dir:
        os.makedirs(samples_dir, exist_ok=True)

    all_rows, completed_pairs = load_existing_results(results_path)
    if completed_pairs:
        logger.info(
            f"Resuming — {len(completed_pairs)} (checkpoint, dataset) pair(s) already complete"
        )

    all_ds_names = [ds_config["name"] for ds_config in config["datasets"]]

    pending = [
        ckpt for ckpt in checkpoints
        if any((_ckpt_label(ckpt), ds) not in completed_pairs for ds in all_ds_names)
    ]
    n_total = len(checkpoints)
    n_pending = len(pending)
    logger.info(
        f"Checkpoints: {n_total} total, {n_total - n_pending} fully done, {n_pending} with pending work"
    )

    if not pending:
        logger.info("Nothing to do — all (checkpoint, dataset) pairs already complete.")
        print(pd.DataFrame(all_rows).to_string(index=False))
        return

    pending_ds_names = {
        ds for ckpt in pending for ds in all_ds_names
        if (_ckpt_label(ckpt), ds) not in completed_pairs
    }
    
    dataset_samples = {}
    if pending_ds_names:
        dataset_samples = prepare_dataset_samples(config, pending_ds_names, samples_dir, model_name, model_cfg, coll)

    layer_indices = None
    ckpt_times = []
    run_start = time.time()

    for run_idx, ckpt in enumerate(pending):
        layer_indices, ckpt_elapsed, ckpt_label = process_checkpoint(
            ckpt, run_idx, pending, n_total, 
            model_name, model_cfg, coll, config, dataset_samples, 
            completed_pairs, all_rows, results_path, layer_indices
        )

        ckpt_times.append(ckpt_elapsed)
        ckpts_done_this_run = len(ckpt_times)
        ckpts_left = n_pending - ckpts_done_this_run
        avg_ckpt_time = sum(ckpt_times) / ckpts_done_this_run
        eta = avg_ckpt_time * ckpts_left
        logger.info(
            f"Checkpoint {ckpts_done_this_run}/{n_pending} pending ({ckpt_label}) done"
            f" | took {_fmt(ckpt_elapsed)}"
            f" | run elapsed {_fmt(time.time() - run_start)}"
            f" | ~{_fmt(eta)} remaining ({ckpts_left} left @ avg {_fmt(avg_ckpt_time)} each)"
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(results_path, index=False)
    logger.info(f"All done — {results_path}  (total run time {_fmt(time.time() - run_start)})")
    print(df.to_string(index=False))


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Collect activations and compute geometry metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
