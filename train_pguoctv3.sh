#!/usr/bin/env bash
# DocROI-Former v3: Native Linux Training Script for pguoctv3 on RTX 5080

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/home/ogmailab/miniconda3/envs/docroi/bin/python"

echo "================================================================================"
echo "  DocROI-Former v3: Native Linux Training Launcher (pguoctv3)"
echo "  NVIDIA RTX 5080 | Mixed Precision FP16 | Persistent Workers"
echo "================================================================================"

$PYTHON_BIN train.py \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --head_lr 1.5e-4 \
    --model_name pguoctv3 \
    --resume "$SCRIPT_DIR/checkpoints/best_docroi_former_v2_pgu.pth" \
    "$@"
