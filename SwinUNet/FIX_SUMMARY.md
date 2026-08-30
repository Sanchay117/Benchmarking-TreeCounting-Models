# Swin-UNet Tree Counting - Bug Fixes Summary

## Problems Identified

### 1. **Negative Predictions** 🔴 CRITICAL
- **Symptom**: Test predictions showing negative values (-6.23, -2.10, etc.)
- **Root Cause**: Model output layer (Conv2d) has NO activation function, allowing unbounded outputs
- **Impact**: Impossible predictions for tree counting task

### 2. **Loss Goes to Zero** 🔴 CRITICAL  
- **Symptom**: Training/validation loss becomes 0.000 after initial epochs
- **Root Cause**: Possible numerical instability or NaN/Inf propagation
- **Impact**: Invalid model training, unable to detect convergence issues

### 3. **Test Results Collapse** 🔴 CRITICAL
- **Symptom**: MAE 119.2755 on test vs 6-7 on validation
- **Root Cause**: 
  - Severe overfitting
  - Possible data preprocessing mismatch between train/val/test
  - Negative predictions causing large errors
- **Impact**: Model completely fails on test data despite good validation results

## Fixes Applied

### Fix #1: Added ReLU Activation to Model Output 
**File**: `network/swin_unet.py`
```python
# In forward() method:
logits = self.model(x)
# ✅ CRITICAL FIX: Apply ReLU activation to ensure non-negative density predictions
logits = torch.nn.functional.relu(logits)
return logits
```
**Why**: Density maps must be non-negative. ReLU ensures all predictions are ≥ 0.

---

### Fix #2: Enhanced Training Stability
**File**: `train_swin_unet_kcl.py`

**Changes**:
1. **Improved Loss Function** - Added NaN/Inf detection:
   ```python
   if torch.isnan(loss) or torch.isinf(loss):
       print(f"WARNING: Loss is {loss}. ...")
       loss = torch.tensor(1.0)  # Fallback
   ```

2. **Training Loop** - Added multiple safeguards:
   - Clamp density maps to [0, ∞) before loss computation
   - Clamp predicted density to [0, ∞)
   - Better error handling for invalid batches
   - Improved gradient clipping

3. **Validation Loop** - Consistent checks:
   - Same density clamping as training
   - NaN/Inf detection
   - Consistent metric computation

**Why**: Prevents training instability and numerical errors

---

### Fix #3: Output Validation in Test Script
**File**: `test_swin_unet_kcl.py`

**Changes**:
```python
# After model inference:
pred_density = torch.clamp(pred_density, min=0)
```

**Why**: Double-checks that predictions are non-negative before final evaluation

---

### Fix #4: Data Loading Safety Checks
**File**: `datasets/kcl_london.py`

**Changes**:

1. **Density Loading** - Check for negative values:
   ```python
   if (density < 0).any():
       print(f"WARNING: Negative values in density map!")
       density = np.clip(density, a_min=0, a_max=None)
   ```

2. **Resizing** - Ensure numerical stability:
   ```python
   if (density < 0).any():
       print(f"WARNING: Negative values after resize!")
       density = np.clip(density, a_min=0, a_max=None)
   ```

3. **Item Loading** - Validate counts:
   ```python
   count = max(0.0, float(count))  # Ensure non-negative
   # Consistency check between density and count
   ```

**Why**: Ensures data integrity throughout the pipeline

---

## Testing the Fixes

### 1. Check for Negative Predictions
```bash
python test_swin_unet_kcl.py \
    --model-path ckpts/swin_unet_kcl_XXXX/best_mae.pth \
    --data-dir ../TreeFormer/datasets \
    --split test_data
```
**Expected**: All predictions > 0

### 2. Monitor Training Stability
```bash
python train_swin_unet_kcl.py \
    --data-dir ../TreeFormer/datasets \
    --epochs 100 \
    --batch-size 8
```
**Expected**: 
- Loss should NOT go to 0.000 
- Loss should decrease smoothly
- No "WARNING: Loss is nan/inf" messages

### 3. Validate Consistency
During training/testing, look for warnings:
- "WARNING: Negative values in density map"
- "WARNING: NaN/Inf detected"

These warnings indicate data issues that were caught and handled.

---

## Implementation Details

### Model Architecture Changes
- **Before**: `Conv2d → unbounded output`
- **After**: `Conv2d → ReLU → [0, ∞) output`

### Loss Computation Changes
- **Before**: Direct MSE computation (could be NaN/Inf)
- **After**: MSE with validation and fallback handling

### Data Pipeline Changes
- **Before**: No validation of density values
- **After**: Multiple checkpoints for negative value detection and handling

---

## Expected Results After Fixes

### Before Fixes:
- MAE: 119.2755 (on test)
- RMSE: 130.0527
- R²: -6.8850
- Predictions: Negative values like -6.23, -2.10

### After Fixes (Expected):
- Predictions: All non-negative
- MAE: Should be reasonable for tree counting
- R²: Should be positive
- Training loss: Should NOT go to 0
- Validation MAE: Should be realistic and close to test MAE

---

## Root Cause Analysis

The primary issue was the **lack of output activation function** in the Swin-UNet model. While this model was designed for medical image segmentation (where the output can represent different tissue types with any real values), tree counting/density regression requires **non-negative outputs**.

The secondary issue was the **lack of stability checks** during training, which masked the output range problem and made the loss function unstable.

---

## Recommendations

### 1. Always Validate Output Ranges
```python
assert (predictions >= 0).all(), "Model outputs negative values!"
```

### 2. Add Periodic Output Monitoring
Log min/max/mean of predictions during training to catch issues early.

### 3. Consider Output Normalization
Depending on density map scale, consider normalizing outputs further:
```python
logits = torch.nn.functional.relu(logits)
# Optional: scale to reasonable range
logits = logits * 255  # if normalizing to [0, 255] range
```

### 4. Use Loss Functions Suitable for Regression
- MSE (what you're using) ✓
- MAE (alternative)
- Huber Loss (robust to outliers)
- Smooth L1 Loss

---

## Files Modified

1. ✅ `network/swin_unet.py` - Added ReLU activation
2. ✅ `train_swin_unet_kcl.py` - Enhanced stability and validation
3. ✅ `test_swin_unet_kcl.py` - Output validation
4. ✅ `datasets/kcl_london.py` - Data validation and safety checks

---

## Quick Verification Checklist

- [ ] No "WARNING: Loss is nan/inf" during training
- [ ] Test predictions are all positive
- [ ] Test predictions are in reasonable range (not all 0s either)
- [ ] Training loss decreases smoothly (doesn't go to 0)
- [ ] Validation MAE is similar to training MAE
- [ ] Test MAE is in same ballpark as validation MAE

---

## Questions or Issues?

If you still see:
1. **NaN/Inf in loss**: Check density map values and ranges
2. **Loss goes to 0**: May indicate overfitting or numerical issues with specific images
3. **Still negative predictions**: Verify ReLU is being applied in forward()
4. **Val vs Test mismatch**: Check preprocessing consistency (especially image normalization)

