# Downstream Evaluation Pipeline

Evaluates FuxiTranyu-8B and Apertus-8B-2509 checkpoints on **m-MMLU** (8 languages, 5-shot)
and **XCOPA** (6 languages, 0-shot) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## File overview

| File | Purpose |
| --- | --- |
| `configs/benchmarks.yaml` | Benchmark definitions (tasks, languages, few-shot settings) |
| `evaluate.py` | Resumable evaluation runner — one job per checkpoint |
| `submit_eval.sh` | Submits `evaluate.py` as a cluster job via `runai`; auto-runs `merge_results.py` after eval |
| `merge_results.py` | Joins eval JSONs with RankMe CSV into a single merged CSV |
| `fetch_json_results.py` | Copies individual eval JSONs from cluster to local (debugging / re-merge) |
| `check_tasks.py` | Verifies all lm-eval task names in `benchmarks.yaml` are valid |
| `results.ipynb` | Analysis notebook (phase identification, grokking, correlations) |

## Prerequisites

1. `.env` file in the repo root with:

    ```env
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

## Model details

| Model | Checkpoints | Layers | Last layer |
| --- | --- | --- | --- |
| FuxiTranyu-8B | 57 (10B → 593B tokens) | 30 (layer_0 … layer_29) | `layer_29` |
| Apertus-8B-2509 | 44 (210B → 15T tokens) | 32 (layer_0 … layer_31) | `layer_31` |

> **Note — Tamil missing from Apertus RankMe:** The geometry analysis did not include Tamil,
> so XCOPA Tamil rows will have NaN for RankMe metrics in the Apertus merged CSV.
> Tamil is present in the Fuxi RankMe data.

## Running on the cluster

Submit **one job per checkpoint**. Each job is independent and resumable after preemption.

Use `submit_all.sh` to submit all checkpoints for a model in one command — it reads
checkpoints from the RankMe CSV, filters out unrecognised entries, and calls
`submit_eval.sh` once per checkpoint:

```bash
./downstream_evaluation/submit_all.sh fuxi      # all 57 Fuxi checkpoints
./downstream_evaluation/submit_all.sh apertus   # all 44 Apertus checkpoints
./downstream_evaluation/submit_all.sh fuxi 5    # smoke test with limit=5
```

To submit a single checkpoint manually:

```bash
./downstream_evaluation/submit_eval.sh 531B fuxi
./downstream_evaluation/submit_eval.sh step2627139-tokens15T apertus
```

### Monitor jobs

```bash
# Stream logs for a specific job
runai training logs -f <job-name> -p course-cs-552-${GASPAR}

# List all running eval jobs
runai workload list -p course-cs-552-${GASPAR}
```

### Resume after preemption

Re-run the exact same `submit_eval.sh` command. The resume logic in `evaluate.py`
scans the output directory at startup and skips any (task, language) pairs that
already have a JSON result file.

## After all jobs complete

### 1. Run merge once manually

Each eval job only runs `evaluate.py`. Running `merge_results.py` per-job would cause
race conditions when multiple checkpoints finish at the same time. Run it once manually
after all jobs are done (spin up a temp shell first if needed):

**Fuxi:**

```bash
runai training exec <shell-job> -p course-cs-552-<gaspar> -- bash -c \
    "cd /scratch/<folder>/open-project-m2-jpmg && \
     python3 downstream_evaluation/merge_results.py \
       --eval-dir results/eval \
       --rankme-csv results/fuxi.csv \
       --output results/fuxi_merged.csv \
       --layer layer_29"
```

**Apertus:**

```bash
runai training exec <shell-job> -p course-cs-552-<gaspar> -- bash -c \
    "cd /scratch/<folder>/open-project-m2-jpmg && \
     python3 downstream_evaluation/merge_results.py \
       --eval-dir results/eval \
       --rankme-csv results/apertus.csv \
       --output results/apertus_merged.csv \
       --layer layer_31"
```

### 2. Fetch the merged CSV locally

**Fuxi:**

```bash
runai training exec <shell-job> -p course-cs-552-<gaspar> -- \
    cat /scratch/<folder>/open-project-m2-jpmg/results/fuxi_merged.csv > results/fuxi_merged.csv
```

**Apertus:**

```bash
runai training exec <shell-job> -p course-cs-552-<gaspar> -- \
    cat /scratch/<folder>/open-project-m2-jpmg/results/apertus_merged.csv > results/apertus_merged.csv
```

> If you need the raw JSONs (e.g. to re-run merge with a different layer), use
> `downstream_evaluation/fetch_json_results.py`.

### 2. Open the analysis notebook

```bash
jupyter notebook downstream_evaluation/results.ipynb
```

> 🚨 **Set `MODEL` before running all cells.** The CSV path, last layer, and model label
> are all derived from this single variable — do not run the notebook without setting it first.

In the first config cell, set:

```python
MODEL = "fuxi"      # or "apertus"
```

| Model | `MODEL` value |
| --- | --- |
| FuxiTranyu-8B | `"fuxi"` |
| Apertus-8B-2509 | `"apertus"` |

All plots are saved to `downstream_evaluation/plots/`.

## Output layout

```text
results/eval/
├── 10B/                          ← Fuxi checkpoint
│   ├── m_mmlu__English.json
│   ├── m_mmlu__Chinese.json
│   ├── ...
│   └── xcopa__Vietnamese.json
├── step50000-tokens210B/         ← Apertus checkpoint
│   ├── m_mmlu__English.json
│   └── ...
└── step2627139-tokens15T/
    └── ...
```

Each JSON contains the raw lm-eval output including accuracy, stderr, and run config.
