#!/bin/bash
# Example training configurations for Swin-UNet on KCL London dataset

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Swin-UNet Training Configuration Examples${NC}"
echo "=========================================="
echo ""

# Configuration 1: Quick Test (5 minutes)
if [ "$1" == "quick" ]; then
    echo -e "${YELLOW}Configuration: QUICK TEST (5 minutes)${NC}"
    echo "Suitable for: Verifying setup, quick validation"
    echo ""
    python train_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --train-split train_data \
        --val-split valid_data \
        --epochs 5 \
        --batch-size 16 \
        --crop-size 224 \
        --lr 1e-4 \
        --device 0 \
        --num-workers 4

# Configuration 2: Standard Training (30-60 minutes)
elif [ "$1" == "standard" ]; then
    echo -e "${YELLOW}Configuration: STANDARD (30-60 minutes)${NC}"
    echo "Suitable for: Good baseline model"
    echo ""
    python train_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --train-split train_data \
        --val-split valid_data \
        --epochs 500 \
        --batch-size 16 \
        --crop-size 256 \
        --lr 1e-4 \
        --weight-decay 1e-5 \
        --device 0 \
        --num-workers 4 \
        --img-size 224 \
        --in-channels 3

# Configuration 3: Large Batch (faster)
elif [ "$1" == "fast" ]; then
    echo -e "${YELLOW}Configuration: FAST TRAINING (20 minutes)${NC}"
    echo "Suitable for: Larger batch size (requires more GPU memory)"
    echo ""
    python train_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --train-split train_data \
        --val-split valid_data \
        --epochs 100 \
        --batch-size 16 \
        --crop-size 256 \
        --lr 5e-4 \
        --weight-decay 1e-4 \
        --device 0 \
        --num-workers 4 \
        --img-size 224

# Configuration 4: High Quality (slow but better accuracy)
elif [ "$1" == "hq" ]; then
    echo -e "${YELLOW}Configuration: HIGH QUALITY (120+ minutes)${NC}"
    echo "Suitable for: Best possible accuracy"
    echo ""
    python train_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --train-split train_data \
        --val-split valid_data \
        --epochs 300 \
        --batch-size 4 \
        --crop-size 384 \
        --lr 1e-4 \
        --weight-decay 1e-5 \
        --device 0 \
        --num-workers 4 \
        --img-size 384 \
        --in-channels 3

# Configuration 5: Grayscale Input
elif [ "$1" == "gray" ]; then
    echo -e "${YELLOW}Configuration: GRAYSCALE INPUT${NC}"
    echo "Suitable for: Single-channel (grayscale) images"
    echo ""
    python train_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --train-split train_data \
        --val-split valid_data \
        --epochs 100 \
        --batch-size 8 \
        --crop-size 256 \
        --lr 1e-4 \
        --device 0 \
        --num-workers 4 \
        --img-size 224 \
        --in-channels 1

# Configuration 6: CPU Training (for debugging)
elif [ "$1" == "cpu" ]; then
    echo -e "${YELLOW}Configuration: CPU TRAINING (for debugging only)${NC}"
    echo "Suitable for: Testing without GPU available"
    echo ""
    python train_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --train-split train_data \
        --val-split valid_data \
        --epochs 2 \
        --batch-size 1 \
        --crop-size 224 \
        --lr 1e-4 \
        --device cpu \
        --num-workers 0

# Configuration 7: Small Model (reduced parameters)
elif [ "$1" == "small" ]; then
    echo -e "${YELLOW}Configuration: SMALL MODEL (reduced parameters)${NC}"
    echo "Suitable for: Limited GPU memory, faster inference"
    echo ""
    python train_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --train-split train_data \
        --val-split valid_data \
        --epochs 100 \
        --batch-size 16 \
        --crop-size 224 \
        --lr 1e-4 \
        --device 0 \
        --num-workers 4 \
        --img-size 160 \
        --in-channels 3

# Configuration 8: Evaluation Only
elif [ "$1" == "eval" ]; then
    if [ -z "$2" ]; then
        echo -e "${RED}Error: Please provide model path as second argument${NC}"
        echo "Usage: bash run_configs.sh eval <model_path>"
        exit 1
    fi
    echo -e "${YELLOW}Configuration: EVALUATION ONLY${NC}"
    echo "Model path: $2"
    echo ""
    python test_swin_unet_kcl.py \
        --data-dir ../TreeFormer/datasets \
        --split test_data \
        --model-path "$2" \
        --device 0 \
        --batch-size 1 \
        --in-channels 3

# Show help
else
    echo -e "${YELLOW}Usage: bash run_configs.sh <config_name>${NC}"
    echo ""
    echo "Available configurations:"
    echo "  ${GREEN}quick${NC}      - Quick test (5 min) - Verify setup"
    echo "  ${GREEN}standard${NC}   - Standard training (30-60 min) - Good baseline"
    echo "  ${GREEN}fast${NC}       - Fast training (20 min) - Large batch size"
    echo "  ${GREEN}hq${NC}         - High quality (120+ min) - Best accuracy"
    echo "  ${GREEN}gray${NC}       - Grayscale input - Single channel"
    echo "  ${GREEN}cpu${NC}        - CPU training - For debugging"
    echo "  ${GREEN}small${NC}      - Small model - Limited GPU memory"
    echo "  ${GREEN}eval${NC}       - Evaluation only - eval <model_path>"
    echo ""
    echo "Examples:"
    echo "  bash run_configs.sh quick"
    echo "  bash run_configs.sh standard"
    echo "  bash run_configs.sh eval ckpts/swin_unet_kcl_20260414-120000/best_mae.pth"
    exit 0
fi

echo ""
echo -e "${GREEN}Training completed!${NC}"
