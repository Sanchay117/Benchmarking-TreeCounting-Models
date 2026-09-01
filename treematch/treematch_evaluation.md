# TreeMatch: Counting Trees from Satellite Imagery with Noisy Supervision

This document outlines the dataset, the data splits, the commands to run the models on the dataset, and the model's reported performance from the paper.

## 1. The TINYTREES Dataset

TINYTREES is the first large-scale benchmark for tree counting from satellite imagery. 
- **Coverage**: Spans three continents and three satellite sensors.
- **Area**: Covers over 25,890 km².
- **Annotations**: Contains more than 216 million tree annotations, including 639k manually verified instances.

The dataset is divided into three geographically and ecologically distinct regions:

1. **China (Gaofen-2)**: 
   - **Resolution**: 0.8 m GSD
   - **Environment**: Temperate forests.
   - **Annotations**: Strong labels via expert photo-interpretation. Weak labels (canopy height maps) derived from medium-density Airborne Laser Scanning (ALS).
2. **Rwanda (PlanetScope)**: 
   - **Resolution**: 3.4 - 4.2 m GSD
   - **Environment**: Heterogeneous tropical landscapes.
   - **Annotations**: Strong labels via expert photo-interpretation. Weak labels from semi-automatic national-scale crown segmentation.
3. **France (SPOT-6)**: 
   - **Resolution**: 1.5 m GSD (pansharpened)
   - **Environment**: Temperate forests.
   - **Annotations**: Strong labels from in situ field measurements (15m radius plots). Weak labels from high-density LiDAR-HD ALS canopy height maps.

---

## 2. Dataset Format & Splits

All images in the dataset are extracted as **64 × 64 RGB+NIR patches** (saved as 5-band GeoTIFFs: 4 spectral bands + 1 binary validity mask) at nominal satellite resolution. 

**Label Format**: Point annotations are stored in a single GeoPackage file (`.gpkg`) per split. Each point contains a `tile` column that links the tree location to its corresponding image patch. For the France (SPOT) weak labels, pseudolabels are derived from Open-Canopy Canopy Height Models (CHM).

To prevent spatial leakage, a 1 km buffer separates training and test regions. Each of the three regions is partitioned into three subsets:

- **Train-weak**: Large geographic areas containing automatic (noisy/weak) annotations.
- **Train-strong**: Smaller geographic areas containing manual (strong/clean) annotations.
- **Test**: Held-out areas containing only manual (strong/clean) annotations for evaluation.

---

## 3. Running the Models

To run the models on the dataset, navigate to the `treematch` directory and use the isolated Conda environment we previously set up. 

**Working Directory:** `/home/ashank/TreeCounting_Benchmark/treematch`

### China (Gaofen-2)
```bash
/home/ashank/TreeCounting_Benchmark/treematch_env/bin/python train.py dataset=gf model=treematch
```

### Rwanda (PlanetScope)
```bash
/home/ashank/TreeCounting_Benchmark/treematch_env/bin/python train.py dataset=ps model=treematch
```

### France (SPOT-6)
```bash
/home/ashank/TreeCounting_Benchmark/treematch_env/bin/python train.py dataset=spot model=treematch
```

> **Note**: To evaluate or train with different strong/weak data ratios (e.g., using 80% strong and 20% weak labels), you can append arguments like `train.strong_ratio=0.8` to the commands above.

---

## 4. Model Performances

The table below summarizes the quantitative evaluation of the **TREEMATCH** model on the TINYTREES benchmark as reported in the paper (Table 3). 

Metrics used:
- **RMSE (↓)**: Image-level root mean squared error on tree counts (trees per hectare). Lower is better.
- **R² (↑)**: Coefficient of determination between predicted and ground-truth counts. Higher is better.
- **nMAE (↓)**: Dataset-level normalized mean absolute error. Lower is better.

| Region / Sensor | Supervision Used | RMSE (↓) | R² (↑) | nMAE (↓) |
| :--- | :---: | :---: | :---: | :---: |
| **China / Gaofen-2** | Strong + Weak | 60.6 | 0.60 | 36.6 |
| **Rwanda / PlanetScope** | Strong + Weak | 72.4 | 0.47 | 51.1 |
| **France / SPOT-6** | Strong + Weak | 147.2 | 0.35 | 37.4 |

TREEMATCH consistently outperformed detection-based (YOLOv8, CenterNet), regression-based, and other distribution-matching (DM-Count) baselines across all splits.

### Reproduced Performances (HuggingFace Checkpoints)

Below is the evaluation of the pretrained checkpoints downloaded from `dgominski/TinyTrees` run using `evaluate_pretrained.py`. *(Note: the paper reports an average of 3 runs, whereas these are the exact outputs from the published checkpoints. The nMAE outputs from the script have been converted to percentages to match the paper).*

| Region / Sensor | Supervision Used | RMSE (↓) | R² (↑) | nMAE (↓) |
| :--- | :---: | :---: | :---: | :---: |
| **China / Gaofen-2** | Strong + Weak | 61.8 | 0.57 | 43.0 |
| **Rwanda / PlanetScope** | Strong + Weak | 79.1 | 0.48 | 52.0 |
| **France / SPOT-6** | Strong + Weak | 153.6 | 0.33 | 39.0 |
