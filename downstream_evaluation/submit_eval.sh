#!/bin/bash
# Submit a downstream evaluation job for ONE model checkpoint.
#
# Runs evaluate.py on the cluster for the given checkpoint, then exits.
# Submit once per checkpoint; the resume logic in evaluate.py handles
# partial runs after preemption without redoing completed (task, language) pairs.
#
# Usage:
#   ./downstream_evaluation/submit_eval.sh <checkpoint> [fuxi|apertus|<hf-repo-id>] [limit]
#
# The optional limit argument passes --limit N to evaluate.py (e.g. 5 examples
# per task). Use it to smoke-test a single checkpoint before the full run.
#
# Examples:
#   ./downstream_evaluation/submit_eval.sh 531B fuxi 5   # smoke test — fast, ~5 min
#   ./downstream_evaluation/submit_eval.sh 531B fuxi     # full run
#   ./downstream_evaluation/submit_eval.sh step2627139-tokens15T apertus
#
# Recommended workflow before submitting all checkpoints:
#   1. ./downstream_evaluation/submit_eval.sh 531B fuxi 5
#   2. Verify JSONs and run merge_results.py + results.ipynb locally
#   3. If OK, submit all checkpoints (no limit)
#
# To submit all Fuxi checkpoints at once:
#   for ckpt in 10B 115B 220B 325B 426B 531B; do
#       ./downstream_evaluation/submit_eval.sh $ckpt fuxi
#   done

set -euo pipefail

# ── Load .env from repo root ───────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/../.env"
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

# ============== EDIT THESE LINES IF NOT USING .env ==============
GASPAR="${GASPAR:-}"
GROUP="${GROUP:-g33}"
CLUSTER_FOLDER="${CLUSTER_FOLDER:-}"
# ================================================================

REPO_NAME="${CLUSTER_FOLDER}/open-project-m2-jpmg"

# ── Argument handling ──────────────────────────────────────────────────────────
CHECKPOINT="${1:-}"
MODEL_KEY="${2:-fuxi}"
LIMIT="${3:-}"
SHELL_JOB="<shell-job>"   # replace with your running shell job name (e.g. temp-shell3)

if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint argument is required." >&2
    echo "Usage: $0 <checkpoint> [fuxi|apertus|<hf-repo-id>]" >&2
    exit 1
fi

case "${MODEL_KEY}" in
    fuxi)
        MODEL_NAME="TJUNLP/FuxiTranyu-8B"
        RANKME_CSV="/scratch/${REPO_NAME}/results/fuxi.csv"
        MERGED_CSV="/scratch/${REPO_NAME}/results/fuxi_merged.csv"
        LAYER="layer_29"
        ;;
    apertus)
        MODEL_NAME="swiss-ai/Apertus-8B-2509"
        RANKME_CSV="/scratch/${REPO_NAME}/results/apertus.csv"
        MERGED_CSV="/scratch/${REPO_NAME}/results/apertus_merged.csv"
        LAYER="layer_31"
        ;;
    *)
        MODEL_NAME="${MODEL_KEY}"
        RANKME_CSV=""
        MERGED_CSV=""
        LAYER=""
        ;;
esac

# NOTE: merge_results.py is NOT run automatically by each job.
# Running it per-job causes race conditions when multiple checkpoints finish
# at the same time and overwrite the same merged CSV. Run it manually once
# after all jobs complete (see instructions printed below).

# ── Validation ─────────────────────────────────────────────────────────────────
if [[ -z "${GASPAR}" ]]; then
    echo "ERROR: GASPAR is not set. Define it in .env or export it." >&2; exit 1
fi
if [[ -z "${CLUSTER_FOLDER}" ]]; then
    echo "ERROR: CLUSTER_FOLDER is not set. Define it in .env or export it." >&2; exit 1
fi

# ── Job configuration ──────────────────────────────────────────────────────────
# Truncate long checkpoint names so the job name stays under the k8s limit.
# Lowercase everything — Kubernetes requires all-lowercase job names.
SAFE_CKPT="${CHECKPOINT//\//_}"
JOB_NAME=$(echo "eval-${MODEL_KEY}-${SAFE_CKPT:0:24}" | tr '[:upper:]' '[:lower:]')
PROJECT="course-cs-552-${GASPAR}"
GPUS=1
NODE="${NODE:-a100-40g}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

LIMIT_ARG=""
if [[ -n "${LIMIT}" ]]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

EVAL_COMMAND="\
cd /scratch/${REPO_NAME} && \
pip install -r requirements.txt -q && \
python downstream_evaluation/evaluate.py \
  --model ${MODEL_NAME} \
  --checkpoint ${CHECKPOINT} \
  --output-dir /scratch/${REPO_NAME}/results/eval \
  --config /scratch/${REPO_NAME}/downstream_evaluation/configs/benchmarks.yaml \
  ${LIMIT_ARG}"

echo ">>> Submitting eval job '${JOB_NAME}'"
echo "    Model      : ${MODEL_NAME}"
echo "    Checkpoint : ${CHECKPOINT}"
echo "    Limit      : ${LIMIT:-none (full run)}"
echo "    Eval output: /scratch/${REPO_NAME}/results/eval/${SAFE_CKPT}/"

runai training submit "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu-devices-request "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment-variable HF_HOME=/scratch/hf_cache \
  --environment-variable HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment-variable EVAL_COMMAND="${EVAL_COMMAND}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd /scratch && \
    eval \"\${EVAL_COMMAND}\""

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs : runai training logs -f ${JOB_NAME} -p ${PROJECT}
Describe    : runai training standard describe ${JOB_NAME} -p ${PROJECT}
Stop        : runai training delete ${JOB_NAME} -p ${PROJECT}

The job is resumable: re-submit with the same command after preemption.
Completed (task, language) pairs are skipped automatically.

See README for how to merge results and fetch the CSV after all jobs complete.
EOF
