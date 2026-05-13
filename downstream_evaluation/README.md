# Downstream Evaluation Pipeline

Evaluates FuxiTranyu-8B and Apertus-8B-2509 checkpoints on **m-MMLU** (8 languages, 5-shot)
and **XCOPA** (6 languages, 0-shot) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## File overview

| File | Purpose |
|---|---|
| `configs/benchmarks.yaml` | Benchmark definitions (tasks, languages, few-shot settings) |
| `evaluate.py` | Resumable evaluation runner — one job per checkpoint |
| `submit_eval.sh` | Submits `evaluate.py` as a cluster job via `runai` |
| `merge_results.py` | Joins eval JSONs with RankMe CSV into a single merged CSV |
| `check_tasks.py` | Verifies all lm-eval task names in `benchmarks.yaml` are valid |
| `results.ipynb` | Analysis notebook (phase identification, grokking, correlations) |

## Prerequisites

1. `.env` file in the repo root with:
    ```
    GASPAR=your_epfl_username
    GROUP=g33
    CLUSTER_FOLDER=your_scratch_subfolder
    HF_TOKEN=your_hf_token
    ```

2. Verify all benchmark task names are valid (run once locally):
    ```bash
    pip install lm-eval
    python downstream_evaluation/check_tasks.py
    ```

## Running on the cluster

Submit **one job per checkpoint**. Each job is independent and resumable after preemption.

### FuxiTranyu-8B (57 checkpoints)

```bash
for ckpt in 10B 21B 31B 42B 52B 63B 73B 84B 94B 105B 115B 126B 136B 147B \
            157B 168B 178B 189B 199B 210B 220B 231B 241B 252B 262B 273B 283B \
            294B 304B 315B 325B 334B 342B 352B 363B 373B 384B 394B 405B 415B \
            426B 436B 447B 457B 468B 478B 489B 499B 510B 520B 531B 541B 552B \
            562B 573B 583B 593B; do
    ./downstream_evaluation/submit_eval.sh $ckpt fuxi
done
```

### Apertus-8B-2509

Fill in the checkpoint list from `results/apertus.csv` once available:

```bash
for ckpt in step...-tokens...B ...; do
    ./downstream_evaluation/submit_eval.sh $ckpt apertus
done
```

### Monitor jobs

```bash
# Stream logs for a specific job
runai logs -f eval-fuxi-531B -p course-cs-552-${GASPAR}

# List all running eval jobs
runai list jobs -p course-cs-552-${GASPAR} | grep eval
```

### Resume after preemption

Re-run the exact same `submit_eval.sh` command. The resume logic in `evaluate.py`
scans the output directory at startup and skips any (task, language) pairs that
already have a JSON result file.

## After all jobs complete

### 1. Merge eval results with RankMe metrics

```bash
python downstream_evaluation/merge_results.py \
    --eval-dir results/eval \
    --rankme-csv results/fuxi.csv \
    --output results/fuxi_merged.csv
```

### 2. Open the analysis notebook

```bash
jupyter notebook downstream_evaluation/results.ipynb
```

Update the `RANKME_CSV` and `EVAL_DIR` paths in the config cell if needed, then
run all cells. All plots are saved to `downstream_evaluation/plots/`.

## Output layout

```
results/eval/
├── 10B/
│   ├── m_mmlu__English.json
│   ├── m_mmlu__Chinese.json
│   ├── ...
│   └── xcopa__Tamil.json
├── 21B/
│   └── ...
└── 593B/
    └── ...
```

Each JSON contains the raw lm-eval output including accuracy, stderr, and run config.
