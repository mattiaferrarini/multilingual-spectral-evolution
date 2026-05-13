# Downstream Evaluation Pipeline

Evaluates FuxiTranyu-8B and Apertus-8B-2509 checkpoints on **m-MMLU** (8 languages, 5-shot)
and **XCOPA** (6 languages, 0-shot) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## File overview

| File | Purpose |
| --- | --- |
| `configs/benchmarks.yaml` | Benchmark definitions (tasks, languages, few-shot settings) |
| `evaluate.py` | Resumable evaluation runner — one job per checkpoint |
| `submit_eval.sh` | Submits `evaluate.py` as a cluster job via `runai` |
| `merge_results.py` | Joins eval JSONs with RankMe CSV into a single merged CSV |
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

### Apertus-8B-2509 (44 checkpoints)

```bash
for ckpt in step50000-tokens210B step100000-tokens420B step150000-tokens630B \
            step200000-tokens840B step250000-tokens1050B step300000-tokens1260B \
            step350000-tokens1470B step400000-tokens1680B step450000-tokens1890B \
            step500000-tokens2100B step550000-tokens2310B step600000-tokens2520B \
            step650000-tokens2730B step700000-tokens2940B step750000-tokens3150B \
            step800000-tokens3360B step850000-tokens3570B step900000-tokens3780B \
            step950000-tokens3990B step1000000-tokens4200B step1194000-tokens5014B \
            step1432000-tokens6014B step1670000-tokens7014B step1678000-tokens7047B \
            step1700000-tokens7232B step1750000-tokens7652B step1800000-tokens8072B \
            step1850000-tokens8492B step1900000-tokens8912B step1950000-tokens9332B \
            step2000000-tokens9752B step2050000-tokens10172B step2100000-tokens10592B \
            step2150000-tokens11012B step2200000-tokens11432B step2250000-tokens11852B \
            step2300000-tokens12272B step2350000-tokens12692B step2400000-tokens13112B \
            step2450000-tokens13532B step2500000-tokens13952B step2550000-tokens14372B \
            step2600000-tokens14792B step2627139-tokens15T; do
    ./downstream_evaluation/submit_eval.sh $ckpt apertus
done
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

### 1. Merge eval results with RankMe metrics

**Fuxi** (last layer = 29):

```bash
python downstream_evaluation/merge_results.py \
    --eval-dir results/eval \
    --rankme-csv results/fuxi.csv \
    --output results/fuxi_merged.csv \
    --layer layer_29
```

**Apertus** (last layer = 31):

```bash
python downstream_evaluation/merge_results.py \
    --eval-dir results/eval \
    --rankme-csv results/apertus.csv \
    --output results/apertus_merged.csv \
    --layer layer_31
```

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
│   └── xcopa__Tamil.json
├── step50000-tokens210B/         ← Apertus checkpoint
│   ├── m_mmlu__English.json
│   └── ...
└── step2627139-tokens15T/
    └── ...
```

Each JSON contains the raw lm-eval output including accuracy, stderr, and run config.
