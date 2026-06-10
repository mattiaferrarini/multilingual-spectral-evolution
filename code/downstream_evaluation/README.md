# Downstream Evaluation Pipeline

Evaluates FuxiTranyu-8B and Apertus-8B-2509 checkpoints on four multilingual benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

| Benchmark | Shot | Languages |
| --- | --- | --- |
| m-MMLU | 5-shot | 8 |
| XCOPA | 0-shot | 5 |
| Belebele | 5-shot | 10 |
| M-ARC | 5-shot | 9 |

## File overview

| File | Purpose |
| --- | --- |
| `configs/benchmarks.yaml` | Benchmark definitions (tasks, languages, few-shot settings) |
| `evaluate.py` | Resumable evaluation runner — one job per checkpoint |
| `submit_eval.sh` | Submits `evaluate.py` as a cluster job |
| `submit_all.sh` | Submits all checkpoints for a model in one command |
| `merge_results.py` | Joins eval JSONs with RankMe CSV into a single merged CSV |
| `fetch_json_results.py` | Copies eval JSONs from cluster to local |
| `check_tasks.py` | Verifies all lm-eval task names in `benchmarks.yaml` are valid |

## Running on the cluster

From your local machine (requires `runai` CLI configured), submit all checkpoints for a model:

```bash
./code/downstream_evaluation/submit_all.sh fuxi      # all 57 Fuxi checkpoints
./code/downstream_evaluation/submit_all.sh apertus   # all 44 Apertus checkpoints
```

Re-running the same command is safe — `evaluate.py` skips already-completed (task, language) pairs.

## Merging results

After all jobs complete, run once from a cluster shell pod:

```bash
# FuxiTranyu
python3 code/downstream_evaluation/merge_results.py \
    --eval-dir results/eval \
    --rankme-csv results/fuxi_fine_wiki.csv \
    --output results/fuxi_fine_wiki_mmlu_xcopa_belebele_marc_layer21_merged.csv \
    --layer layer_21

# Apertus
python3 code/downstream_evaluation/merge_results.py \
    --eval-dir results/eval \
    --rankme-csv results/apertus_fine_wiki.csv \
    --output results/apertus_fine_wiki_mmlu_xcopa_belebele_marc_layer19_merged.csv \
    --layer layer_19
```

Then pull the merged CSVs locally and open the analysis notebook:

```bash
jupyter notebook notebooks/joao_pinto_407597.ipynb
```
