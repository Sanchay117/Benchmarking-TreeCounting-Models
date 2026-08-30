# Integration with TreeCounting_Benchmark

This document explains how the Swin-UNet implementation integrates with the existing TreeCounting_Benchmark infrastructure while remaining completely separate.

## Directory Structure

```
TreeCounting_Benchmark/
├── AdaTreeFormer/          # Another benchmarked model
├── TreeFormer/             # TreeFormer benchmarking
│   ├── datasets/           # Contains train_data, valid_data, test_data
│   │   ├── train_data/
│   │   │   ├── images/
│   │   │   └── ground_truth/
│   │   ├── valid_data/
│   │   └── test_data/
│   ├── train.py            # TreeFormer training
│   ├── test.py             # TreeFormer testing
│   ├── train_mcnn_kcl.py   # MCNN fine-tuning
│   └── test_mcnn_kcl.py    # MCNN evaluation
├── crowdcount-mcnn/        # MCNN implementation
└── SwinUNet/ (NEW)         # Swin-UNet implementation
    ├── train_swin_unet_kcl.py    # Swin-UNet training
    ├── test_swin_unet_kcl.py     # Swin-UNet evaluation
    ├── network/
    │   └── swin_unet.py          # Model implementation
    ├── datasets/
    │   └── kcl_london.py         # Dataset loader
    └── ...
```

## Key Design Principles

### 1. Separate Implementation
- **No interference**: The SwinUNet module has its own:
  - Model architecture (different from TreeFormer)
  - Dataset loader (compatible with KCL London format)
  - Training pipeline
  - Evaluation pipeline
  
- **Shared dataset**: Uses the same ground truth data in `TreeFormer/datasets/`
  - Both models read from the same `train_data/`, `valid_data/`, `test_data/` splits
  - Same density maps and annotations

### 2. Independent Outputs
Each model maintains its own checkpoint and prediction directories:

```
SwinUNet/
├── ckpts/swin_unet_kcl_<timestamp>/     # Swin-UNet checkpoints
│   ├── best_mae.pth
│   ├── latest.pth
│   └── history.json
├── eval/swin_unet_test_<timestamp>/     # Swin-UNet results
│   ├── final_stats.json
│   └── test_summary.txt
└── predictions/                          # Swin-UNet predictions

TreeFormer/
├── ckpts/                               # TreeFormer checkpoints
├── eval/                                # TreeFormer results
└── predictions/                         # TreeFormer predictions

crowdcount-mcnn/
├── ckpts/mcnn_kcl_<timestamp>/          # MCNN checkpoints
├── predictions_mcnn/                     # MCNN predictions
└── eval/                                 # MCNN results
```

## Comparing Results

### Quick Comparison

To compare all models on the same test set:

```bash
# 1. Run Swin-UNet evaluation
cd SwinUNet
python test_swin_unet_kcl.py \
  --model-path ckpts/swin_unet_kcl_<timestamp>/best_mae.pth

# 2. Check MCNN results in TreeFormer
cd ../TreeFormer
python test_mcnn_kcl.py \
  --model-path ckpts/mcnn_kcl/<timestamp>/best_mae.pth

# 3. Compare metrics
cat eval/swin_unet_test_*/final_stats.json | grep -E "mae|mse|r2"
```

### Results Comparison Script

```python
import json
import os

# Collect results from all models
results = {}

# Swin-UNet
swin_results = json.load(open('SwinUNet/eval/swin_unet_test_latest/final_stats.json'))
results['Swin-UNet'] = {
    'mae': swin_results['mae'],
    'rmse': swin_results['rmse'],
    'r2': swin_results['r2']
}

# MCNN (from TreeFormer)
mcnn_results = json.load(open('TreeFormer/eval/mcnn_test_latest/final_stats.json'))
results['MCNN'] = {
    'mae': mcnn_results['mae'],
    'mse': mcnn_results['mse'],
    'r2': mcnn_results['r2']
}

# Print comparison table
print(f"{'Model':<15} {'MAE':<10} {'RMSE/MSE':<12} {'R²':<8}")
print("-" * 50)
for model, metrics in results.items():
    rmse_val = metrics.get('rmse', metrics.get('mse', 0))
    print(f"{model:<15} {metrics['mae']:<10.4f} {rmse_val:<12.4f} {metrics['r2']:<8.4f}")
```

## Dataset Compatibility

Both Swin-UNet and MCNN use identical dataset format:

### KCL London Dataset Format
```
<split>/
├── images/
│   ├── IMG_158.jpg
│   ├── IMG_159.jpg
│   └── ...
└── ground_truth/
    ├── GT_IMG_158.mat          # Keypoint annotations
    ├── IMG_158_densitymap.npy  # Pre-computed density map
    ├── GT_IMG_159.mat
    ├── IMG_159_densitymap.npy
    └── ...
```

### Dataset Loading
Both models load data identically:
```python
# Swin-UNet
from SwinUNet.datasets.kcl_london import KCLLondonSwinUNetDataset
dataset = KCLLondonSwinUNetDataset(
    root='../TreeFormer/datasets',
    split='train_data'
)

# MCNN  
from TreeFormer.datasets.mcnn_kcl import KCLLondonMCNNDataset
dataset = KCLLondonMCNNDataset(
    root='datasets',
    split='train_data'
)
```

## Evaluation Metrics

Both models report the same metrics:

| Metric | Definition | Usage |
|--------|-----------|-------|
| **MAE** | Mean Absolute Error | Primary metric for tree counting |
| **RMSE** | Root Mean Square Error | Penalizes large errors |
| **R²** | Coefficient of determination | Overall fit quality (0-1) |

### Metric Calculation
```python
# Same formulas used by both models
mae = np.mean(np.abs(predicted - ground_truth))
rmse = np.sqrt(np.mean((predicted - ground_truth) ** 2))
ss_res = np.sum((gt - pred) ** 2)
ss_tot = np.sum((gt - np.mean(gt)) ** 2)
r2 = 1 - (ss_res / ss_tot)
```

## Hyperparameter Comparison

### Swin-UNet Configuration
```json
{
  "img_size": 224,
  "patch_size": 4,
  "embed_dim": 96,
  "depths": [2, 2, 2, 2],
  "num_heads": [3, 6, 12, 24],
  "window_size": 7,
  "learning_rate": 1e-4,
  "batch_size": 8,
  "epochs": 100,
  "crop_size": 256
}
```

### MCNN Configuration  
```json
{
  "input_size": 224,
  "batch_size": 8,
  "learning_rate": 1e-5,
  "epochs": 300
}
```

## Reproducibility

### To reproduce Swin-UNet results:

1. **Environment Setup**
   ```bash
   cd SwinUNet
   pip install -r requirements.txt
   ```

2. **Training**
   ```bash
   python train_swin_unet_kcl.py \
     --data-dir ../TreeFormer/datasets \
     --epochs 100 \
     --batch-size 8 \
     --lr 1e-4
   ```

3. **Evaluation**
   ```bash
   python test_swin_unet_kcl.py \
     --model-path ckpts/swin_unet_kcl_<timestamp>/best_mae.pth
   ```

4. **Results Location**
   - Checkpoints: `ckpts/swin_unet_kcl_<timestamp>/`
   - Predictions: `predictions/<image_name>.mat`
   - Results: `eval/swin_unet_test_<timestamp>/final_stats.json`

## Contributing Results

To add your Swin-UNet results to the benchmark:

1. Run complete evaluation on test set
2. Save results directory structure:
   ```
   eval/swin_unet_test_<timestamp>/
   ├── final_stats.json
   ├── test_summary.txt
   └── hparams.json
   ```

3. Document hyperparameters used
4. Include metrics (MAE, RMSE, R²)

## Known Differences from Official Implementation

1. **Input channels**: Flexible (grayscale or RGB)
2. **Dataset**: Adapted for tree counting (from medical imaging original)
3. **Loss function**: MSE (can be extended to other losses)
4. **Evaluation**: Integrated density map + count regression

## References

- Official Swin-UNet: https://github.com/HuCaoFighting/Swin-Unet
- TreeFormer Benchmark: https://arxiv.org/abs/2307.06118
- This implementation serves as a comparison baseline

## Support

For questions about:
- **Swin-UNet**: See `SwinUNet/README.md`
- **Integration**: See this file
- **Dataset format**: See `TreeFormer/README.md`
