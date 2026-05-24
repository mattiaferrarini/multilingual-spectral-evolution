#!/bin/bash
# Submit downstream evaluation jobs for ALL checkpoints of a given model.
#
# Reads checkpoints from the model's RankMe CSV, filters out unrecognised
# entries (e.g. the 'main' HF branch), sorts them by token count, and calls
# submit_eval.sh once per checkpoint.
#
# Usage:
#   ./downstream_evaluation/submit_all.sh [fuxi|apertus] [limit]
#
# Examples:
#   ./downstream_evaluation/submit_all.sh fuxi        # full run, all checkpoints
#   ./downstream_evaluation/submit_all.sh fuxi 5      # smoke test with limit=5
#   ./downstream_evaluation/submit_all.sh apertus

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_KEY="${1:-fuxi}"
LIMIT="${2:-}"

case "${MODEL_KEY}" in
    fuxi)    RANKME_CSV="${REPO_ROOT}/results/fuxi.csv" ;;
    apertus) RANKME_CSV="${REPO_ROOT}/results/apertus.csv" ;;
    *)
        echo "ERROR: unknown model '${MODEL_KEY}'. Choose: fuxi | apertus" >&2
        exit 1
        ;;
esac

if [[ ! -f "${RANKME_CSV}" ]]; then
    echo "ERROR: ${RANKME_CSV} not found." >&2
    exit 1
fi

# Read and sort checkpoints from the CSV using the same logic as the notebook.
CHECKPOINTS=$(python3 - <<PYEOF
import sys
sys.path.insert(0, "${SCRIPT_DIR}")
import pandas as pd
from utils.checkpoints import sort_checkpoints, ckpt_to_tokens

df = pd.read_csv("${RANKME_CSV}")
known = [c for c in df["checkpoint"].unique() if ckpt_to_tokens(c) != float("inf")]
for ckpt in sort_checkpoints(known):
    print(ckpt)
PYEOF
)

TOTAL=$(echo "${CHECKPOINTS}" | wc -l)
echo ">>> Submitting ${TOTAL} eval jobs for model: ${MODEL_KEY}"
echo "    Limit: ${LIMIT:-none (full run)}"
echo ""

COUNT=0
while IFS= read -r CKPT; do
    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] ${CKPT}"
    "${SCRIPT_DIR}/submit_eval.sh" "${CKPT}" "${MODEL_KEY}" ${LIMIT}
done <<< "${CHECKPOINTS}"

echo ""
echo ">>> All ${TOTAL} jobs submitted."
echo "    Monitor: runai training list -p course-cs-552-\${GASPAR:-<gaspar>}"
