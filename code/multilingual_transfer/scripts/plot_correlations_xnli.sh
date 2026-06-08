#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=code/multilingual_transfer
CONFIGS=$SCRIPTS/configs

for model in xnli_apertus.yaml xnli_fuxi.yaml; do

    echo "=== ahead plots: $model ==="
    # python $SCRIPTS/xnli_plot_ahead.py \
    #     --config "$CONFIGS/$model" \
    #     --analysis-config "$CONFIGS/xnli_correlation_ahead_analysis.yaml"

    echo "=== ahead collapsed plots: $model ==="
    python $SCRIPTS/xnli_plot_ahead_collapsed.py \
        --config "$CONFIGS/$model" \
        --analysis-config "$CONFIGS/xnli_correlation_ahead_collapsed_analysis.yaml"
done
