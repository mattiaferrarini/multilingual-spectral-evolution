#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=code/multilingual_transfer
CONFIGS=$SCRIPTS/configs

for model in eclektic_law_apertus.yaml eclektic_law_fuxi.yaml; do

    echo "=== eclektic law plots: $model ==="
    python $SCRIPTS/eclektic_plot_law.py \
        --config "$CONFIGS/$model" \
        --analysis-config "$CONFIGS/eclektic_law_correlation_analysis.yaml"

done
