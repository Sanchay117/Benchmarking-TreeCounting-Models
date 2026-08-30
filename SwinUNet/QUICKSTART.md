# Quick Start Guide for Swin-UNet Tree Counting Benchmark

## 5-Minute Quick Start

### Step 1: Setup Environment

```bash
cd SwinUNet
pip install -r requirements.txt
```

### Step 2: Train Model (5-10 minutes on GPU)

For a quick training run with minimal epochs:

```bash
python train_swin_unet_kcl.py \
  --data-dir ../TreeFormer/datasets \
  --epochs 5 \
  --batch-size 8 \
  --device 0
```

Expected output:
```
Run directory: ckpts/swin_unet_kcl_20260414-120000
Hyperparameters saved to: ckpts/swin_unet_kcl_20260414-120000/hparams.json
Loading datasets...
Training samples: 452
Validation samples: 161
...
Epoch 1/5
  Batch 10/57: Loss=0.0234, MAE=5.23
  Batch 20/57: Loss=0.0198, MAE=4.89
...
```

### Step 3: Evaluate Model

After training completes, evaluate on test set:

```bash
python test_swin_unet_kcl.py \
  --data-dir ../TreeFormer/datasets \
  --split test_data \
  --model-path ckpts/swin_unet_kcl_<timestamp>/best_mae.pth \
  --device 0
```

Expected output:
```
============================================================
Evaluation Results on test_data
============================================================
Model: ckpts/swin_unet_kcl_20260414-120000/best_mae.pth
Metrics:
  MAE (Mean Absolute Error):  18.5432
  RMSE (Root Mean Square Error): 25.6234
  R² Score: 0.7823
  Number of test images: 123
============================================================
```

## Full Training Pipeline

### Option 1: Standard Training (100 epochs)

```bash
python train_swin_unet_kcl.py \
  --data-dir ../TreeFormer/datasets \
  --train-split train_data \
  --val-split valid_data \
  --epochs 100 \
  --batch-size 8 \
  --crop-size 256 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --device 0 \
  --output-dir ckpts \
  --in-channels 3
```

**Expected training time**: ~30-60 minutes on NVIDIA V100 GPU

### Option 2: Quick Validation (10 epochs)

```bash
python train_swin_unet_kcl.py \
  --epochs 10 \
  --batch-size 16 \
  --lr 5e-4 \
  --device 0
```

## Checking Results

### View Training History

```bash
# After training completes, check the history:
cat ckpts/swin_unet_kcl_<timestamp>/history.json | python -m json.tool
```

Output shows loss, MAE, and MSE for each epoch.

### View Evaluation Results

```bash
# After evaluation, check final metrics:
cat eval/swin_unet_test_<timestamp>/final_stats.json | python -m json.tool

# View human-readable summary:
cat eval/swin_unet_test_<timestamp>/test_summary.txt
```

### Inspect Predictions

Predicted density maps are saved as `.mat` files:

```python
import scipy.io as sio
import matplotlib.pyplot as plt

# Load prediction
result = sio.loadmat('predictions/IMG_158.mat')

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(result['image'], cmap='gray')
axes[0].set_title(f"Input Image")
axes[1].imshow(result['gt_density'])
axes[1].set_title(f"Ground Truth (Count={result['gt_count'][0][0]:.0f})")
axes[2].imshow(result['pred_density'])
axes[2].set_title(f"Prediction (Count={result['pred_count'][0][0]:.0f})")
plt.tight_layout()
plt.savefig('comparison.png')
```

## Understanding the Output

### Directory Structure After Training

```
ckpts/
└── swin_unet_kcl_20260414-120000/
    ├── best_mae.pth          # Best model (lowest validation MAE)
    ├── latest.pth            # Latest epoch checkpoint
    ├── hparams.json          # Training configuration
    └── history.json          # Training curves (loss, MAE, MSE)

eval/
└── swin_unet_test_20260414-121500/
    ├── best_mae.pth          # (symlink to best model)
    ├── final_stats.json      # Evaluation metrics + per-image predictions
    ├── test_summary.txt      # Human-readable results
    └── hparams.json          # Evaluation configuration

predictions/
├── IMG_158.mat             # Prediction for image IMG_158
├── IMG_159.mat
└── ...
```

### Key Metrics Explanation

| Metric | Range | Better | Interpretation |
|--------|-------|--------|-----------------|
| MAE | 0-∞ | Lower | Average absolute difference in tree counts |
| RMSE | 0-∞ | Lower | Penalizes large errors more than MAE |
| R² | -∞-1 | Higher | 1.0 = perfect, 0.0 = mean prediction, <0 = worse than mean |

## Advanced Options

### Using Different Input Sizes

```bash
# Larger input for better accuracy (slower)
python train_swin_unet_kcl.py --img-size 384 --batch-size 4

# Smaller input for faster training
python train_swin_unet_kcl.py --img-size 160 --batch-size 16
```

### Grayscale Input

```bash
# Use grayscale images (1 channel)
python train_swin_unet_kcl.py --in-channels 1 
```

### Custom Learning Rate

```bash
# Higher learning rate for faster convergence (may be unstable)
python train_swin_unet_kcl.py --lr 5e-4

# Lower learning rate for more stable training (slower)
python train_swin_unet_kcl.py --lr 5e-5
```

### Resume Training

```bash
# Continue training from checkpoint
python train_swin_unet_kcl.py \
  --resume ckpts/swin_unet_kcl_20260414-120000/latest.pth \
  --epochs 150
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train_swin_unet_kcl.py --batch-size 4

# Reduce image size
python train_swin_unet_kcl.py --img-size 192
```

### No CUDA Available

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU (slow)
python train_swin_unet_kcl.py --device cpu
```

### Dataset Not Found

```bash
# Verify dataset structure
ls ../TreeFormer/datasets/train_data/images/ | wc -l
ls ../TreeFormer/datasets/test_data/ground_truth/ | wc -l
```

## Comparison with Other Models

In the TreeCounting_Benchmark, you can compare Swin-UNet with:

- **TreeFormer** (Transformer-based, semi-supervised)
- **MCNN** (CNN-based baseline)
- **CSRNet** (Dilated CNN)

Check individual READMEs in each model's directory for benchmarking commands.

## Next Steps

1. **Train on full dataset**: Increase `--epochs` to 100-300
2. **Tune hyperparameters**: Adjust `--lr`, `--batch-size`, `--crop-size`
3. **Compare with baselines**: Run other models in the benchmark
4. **Analyze predictions**: Inspect `.mat` files in `predictions/` directory
5. **Visualize results**: Create comparison plots

## Documentation

- Full documentation: See `README.md`
- Model details: `network/swin_unet.py`
- Dataset loader: `datasets/kcl_london.py`
- Training pipeline: `train_swin_unet_kcl.py`
- Evaluation pipeline: `test_swin_unet_kcl.py`
