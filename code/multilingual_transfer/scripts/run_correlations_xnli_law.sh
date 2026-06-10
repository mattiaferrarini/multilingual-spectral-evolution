#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=code/multilingual_transfer
CONFIGS=$SCRIPTS/configs

for model in xnli_law_apertus.yaml xnli_law_fuxi.yaml; do

    echo "=== xnli law: $model ==="
    python $SCRIPTS/xnli_correlate_law.py \
        --config "$CONFIGS/$model" \
        --analysis-config "$CONFIGS/xnli_law_correlation_analysis.yaml"

done
