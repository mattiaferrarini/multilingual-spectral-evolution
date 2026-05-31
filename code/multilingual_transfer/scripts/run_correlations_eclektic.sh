#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=code/multilingual_transfer
CONFIGS=$SCRIPTS/configs

for model in eclektic_apertus.yaml eclektic_fuxi.yaml; do

    echo "=== eclektic: $model ==="
    python $SCRIPTS/eclektic_correlate.py \
        --config "$CONFIGS/$model" \
        --analysis-config "$CONFIGS/eclektic_correlation_analysis.yaml"

    echo "=== eclektic collapsed: $model ==="
    python $SCRIPTS/eclektic_correlate_collapsed.py \
        --config "$CONFIGS/$model" \
        --analysis-config "$CONFIGS/eclektic_correlation_collapsed_analysis.yaml"
done
