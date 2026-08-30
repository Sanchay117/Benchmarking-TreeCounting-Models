# Swin-UNet: Tree Counting Benchmark on KCL London Dataset

This is an independent implementation of **Swin-UNet** for tree counting on the KCL London dataset, separate from the existing TreeFormer benchmarking pipeline.

## Overview

Swin-UNet is a pure Transformer-based U-shaped encoder-decoder architecture for image segmentation tasks, including tree density map estimation for tree counting. This implementation is based on the paper:

- **Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation** ([arXiv:2105.05537](https://arxiv.org/abs/2105.05537))
  - Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, Manning Wang

## Key Features

✅ **Separate Implementation**: Does not interfere with existing TreeFormer benchmarking  
✅ **Complete Pipeline**: Training and evaluation scripts included  
✅ **Flexible Configuration**: Supports RGB and grayscale inputs  
✅ **Comprehensive Metrics**: MAE, RMSE, R² scores, and per-image predictions  
✅ **Checkpoint Management**: Best model selection and training history  

## Installation

### 1. Prerequisites

From the TreeCounting_Benchmark root directory:

```bash
cd SwinUNet
pip install -r requirements.txt
```

### 2. Download Datasets

The KCL London dataset is already available in the TreeFormer directory:
```
../TreeFormer/datasets/
├── train_data/
├── valid_data/
└── test_data/
```

Each split contains:
- `images/`: JPEG images  
- `ground_truth/`: 
  - `GT_<name>.mat`: Tree keypoint annotations
  - `<name>_densitymap.npy`: Pre-computed density maps

## Model Architecture

The Swin-UNet consists of:

- **Encoder**: Hierarchical Swin Transformer blocks with windowed attention and patch merging
- **Bottleneck**: 2 Swin Transformer blocks at the deepest level
- **Decoder**: Symmetric structure with patch expanding layers and skip connections
- **Output**: Density map prediction via linear projection

### Architecture Details

```
Input: [B, 3, 224, 224]  (RGB) or [B, 1, 224, 224] (Grayscale)
  ↓
Patch Embedding: [B, 56×56, 96]
  ↓
Encoder (4 stages):
  - Stage 1: 56×56, 96 channels
  - Stage 2: 28×28, 192 channels
  - Stage 3: 14×14, 384 channels
  - Stage 4: 7×7, 768 channels
  ↓
Bottleneck: 2 Swin blocks at 7×7
  ↓
Decoder (4 stages) with skip connections:
  - Patch expanding + Swin blocks
  - Progressively upsampling to 224×224
  ↓
Output: [B, 1, 224, 224] (density map)
```

## Usage

### 1. Training

Train Swin-UNet on KCL London training data:

```bash
python train_swin_unet_kcl.py \
  --data-dir ../TreeFormer/datasets \
  --train-split train_data \
  --val-split valid_data \
  --epochs 100 \
  --batch-size 8 \
  --crop-size 256 \
  --lr 1e-4 \
  --device 0 \
  --in-channels 3
```

#### Training Arguments

- `--data-dir`: Path to dataset root (default: `../TreeFormer/datasets`)
- `--train-split`: Training split name (default: `train_data`)
- `--val-split`: Validation split name (default: `valid_data`)
- `--epochs`: Number of training epochs (default: `100`)
- `--batch-size`: Training batch size (default: `8`)
- `--crop-size`: Random crop size for training (default: `256`)
- `--img-size`: Input image size (default: `224`)
- `--lr`: Learning rate (default: `1e-4`)
- `--weight-decay`: L2 regularization (default: `1e-4`)
- `--device`: CUDA device ID (default: `0`)
- `--num-workers`: Data loading workers (default: `4`)
- `--output-dir`: Checkpoint save directory (default: `ckpts`)
- `--in-channels`: Input channels - 1 for grayscale, 3 for RGB (default: `3`)
- `--pretrained`: Path to pretrained checkpoint (optional)
- `--resume`: Path to checkpoint for resuming training (optional)

#### Output

Checkpoints saved to `ckpts/swin_unet_kcl_<timestamp>/`:
- `latest.pth`: Latest epoch checkpoint
- `best_mae.pth`: Best validation MAE checkpoint
- `hparams.json`: Hyperparameters used
- `history.json`: Training history (loss, MAE, MSE)

### 2. Evaluation/Benchmarking

Evaluate trained model on test set:

```bash
python test_swin_unet_kcl.py \
  --data-dir ../TreeFormer/datasets \
  --split test_data \
  --model-path ckpts/swin_unet_kcl_<timestamp>/best_mae.pth \
  --device 0 \
  --in-channels 3
```

#### Evaluation Arguments

- `--data-dir`: Dataset root directory
- `--split`: Dataset split to evaluate (default: `test_data`)
- `--model-path`: Path to trained model checkpoint (required)
- `--device`: CUDA device ID (default: `0`)
- `--batch-size`: Evaluation batch size (default: `1`)
- `--output-dir`: Results save directory (default: `eval`)
- `--pred-dir`: Directory to save predictions (default: `predictions`)
- `--img-size`: Input image size (default: `224`)
- `--in-channels`: Input channels (default: `3`)

#### Output

Results saved to `eval/swin_unet_test_<timestamp>/`:
- `final_stats.json`: Metrics (MAE, RMSE, R²) and per-image predictions
- `test_summary.txt`: Human-readable summary
- `hparams.json`: Evaluation hyperparameters
- Predictions saved to `predictions/` as `.mat` files containing:
  - `pred_density`: Predicted density map
  - `gt_density`: Ground truth density map
  - `image`: Input image
  - `pred_count`: Predicted tree count
  - `gt_count`: Ground truth tree count
  - `error`: Absolute error

### 3. Using Pre-trained Weights (Optional)

To use the official pre-trained Swin Transformer weights:

```bash
# 1. Download Swin-T pre-trained model from:
# https://github.com/HuCaoFighting/Swin-Unet or Google Drive

# 2. Place in pretrained_ckpt/ directory

# 3. Use when training:
python train_swin_unet_kcl.py \
  --pretrained pretrained_ckpt/swin_tiny_patch4_window7_224.pth \
  ...
```

## Evaluation Metrics

The benchmark uses the following metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and ground truth counts
- **RMSE (Root Mean Square Error)**: Square root of mean squared differences
- **R² Score**: Coefficient of determination (1.0 = perfect prediction, 0.0 = mean prediction)

## Expected Performance

Expected results on KCL London dataset (after training):

| Metric | Value |
|--------|-------|
| MAE | ~15-20 |
| RMSE | ~20-30 |
| R² | ~0.70-0.85 |

*Note: These are approximate values; actual results depend on hyperparameters and training data split.*

## File Structure

```
SwinUNet/
├── network/
│   └── swin_unet.py              # Swin-UNet model implementation
├── datasets/
│   └── kcl_london.py             # KCL London dataset loader
├── train_swin_unet_kcl.py        # Training script
├── test_swin_unet_kcl.py         # Evaluation/benchmarking script
├── pretrained_ckpt/              # Pre-trained weights (download here)
├── ckpts/                        # Training checkpoints
├── eval/                         # Evaluation results
├── predictions/                  # Predicted density maps
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Implementation Notes

### Model Configuration

- **Input size**: 224×224 (can be adjusted with `--img-size`)
- **Patch size**: 4×4
- **Embed dim**: 96
- **Depths**: (2, 2, 2, 2) - number of blocks per stage
- **Number of heads**: (3, 6, 12, 24) per stage
- **Window size**: 7×7

### Training Strategy

- **Optimizer**: Adam with `lr=1e-4`, `weight_decay=1e-4`
- **Scheduler**: StepLR with `step_size=30`, `gamma=0.1`
- **Loss function**: Mean Squared Error (MSE)
- **Data augmentation**: Random cropping, random horizontal flips

### Compatibility

- **PyTorch**: 1.9.0+
- **CUDA**: Any version compatible with your PyTorch installation
- **GPU Memory**: ~4-6GB for batch size 8
- **CPU**: Supported but slow

## Troubleshooting

### Memory Issues

If you get out-of-memory errors:
```bash
# Reduce batch size
python train_swin_unet_kcl.py --batch-size 4 ...

# Reduce image size
python train_swin_unet_kcl.py --img-size 192 ...
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

Ensure `timm` is properly installed:
```bash
pip install timm>=0.4.12
```

### Dataset Issues

Verify dataset structure:
```bash
ls ../TreeFormer/datasets/train_data/images/ | head -5
ls ../TreeFormer/datasets/train_data/ground_truth/ | head -5
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{cao2021swin,
  title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
  author={Cao, Hu and Wang, Yueyue and Chen, Joy and Jiang, Dongsheng and Zhang, Xiaopeng and Tian, Qi and Wang, Manning},
  journal={arXiv preprint arXiv:2105.05537},
  year={2021}
}
```

## References

1. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
2. [TreeFormer: a Semi-Supervised Transformer-based Framework for Tree Counting](https://arxiv.org/abs/2307.06118)
3. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## License

This implementation is provided for research and benchmarking purposes.

## Support

For issues or questions about this implementation:
1. Check the troubleshooting section above
2. Verify dataset structure and paths
3. Ensure all dependencies are correctly installed
4. Check CUDA/GPU availability with `python -c "import torch; print(torch.cuda.is_available())"`
