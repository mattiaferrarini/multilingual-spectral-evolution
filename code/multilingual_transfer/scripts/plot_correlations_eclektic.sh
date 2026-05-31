#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=code/multilingual_transfer
CONFIGS=$SCRIPTS/configs

for model in eclektic_apertus.yaml eclektic_fuxi.yaml; do

    echo "=== eclektic plots: $model ==="
    python $SCRIPTS/eclektic_plot.py \
        --config "$CONFIGS/$model" \
        --analysis-config "$CONFIGS/eclektic_correlation_analysis.yaml"

    echo "=== eclektic collapsed plots: $model ==="
    python $SCRIPTS/eclektic_plot_collapsed.py \
        --config "$CONFIGS/$model" \
        --analysis-config "$CONFIGS/eclektic_correlation_collapsed_analysis.yaml"
done
