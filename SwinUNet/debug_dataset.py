"""
Diagnostic script to verify dataset loading and preprocessing.
"""
import torch
from datasets.kcl_london import KCLLondonSwinUNetDataset

print("=" * 80)
print("DEBUGGING DATASET LOADING")
print("=" * 80)

# Test with crop_size=256 (old broken way)
print("\n[1] Dataset with crop_size=256 (BROKEN - random crops):")
dataset_cropped = KCLLondonSwinUNetDataset(
    root="../TreeFormer/datasets",
    split="train_data",
    crop_size=256,
    random_flip=False,
    in_channels=3
)

print(f"    Samples: {len(dataset_cropped)}")
# Get same sample multiple times to see if it changes (it will due to random crop)
sample1 = dataset_cropped[0]
sample2 = dataset_cropped[0]
sample3 = dataset_cropped[0]

count1 = sample1['count'].item()
count2 = sample2['count'].item()
count3 = sample3['count'].item()

print(f"    Image shape: {sample1['image'].shape}")
print(f"    Density shape: {sample1['density'].shape}")
print(f"    Tree counts from same image (3 loads): {count1:.1f}, {count2:.1f}, {count3:.1f}")
if not (abs(count1 - count2) < 0.1 and abs(count2 - count3) < 0.1):
    print("    ⚠️  WARNING: Different counts for same image! (Random cropping)")

# Test with crop_size=None (fixed way)
print("\n[2] Dataset with crop_size=None (FIXED - full images):")
dataset_full = KCLLondonSwinUNetDataset(
    root="../TreeFormer/datasets",
    split="train_data",
    crop_size=None,
    random_flip=False,
    in_channels=3
)

print(f"    Samples: {len(dataset_full)}")
sample1 = dataset_full[0]
sample2 = dataset_full[0]
sample3 = dataset_full[0]

count1 = sample1['count'].item()
count2 = sample2['count'].item()
count3 = sample3['count'].item()

print(f"    Image shape: {sample1['image'].shape}")
print(f"    Density shape: {sample1['density'].shape}")
print(f"    Tree counts from same image (3 loads): {count1:.1f}, {count2:.1f}, {count3:.1f}")
if abs(count1 - count2) < 0.1 and abs(count2 - count3) < 0.1:
    print("    ✅ GOOD: Consistent counts! (Full images, no random cropping)")

# Compare validation dataset processing
print("\n[3] Validation dataset comparison:")
val_cropped = KCLLondonSwinUNetDataset(
    root="../TreeFormer/datasets",
    split="valid_data",
    crop_size=256,
    random_flip=False,
    in_channels=3
)

val_full = KCLLondonSwinUNetDataset(
    root="../TreeFormer/datasets",
    split="valid_data",
    crop_size=None,
    random_flip=False,
    in_channels=3
)

# Get average count from a few samples
counts_cropped = [val_cropped[i]['count'].item() for i in range(min(5, len(val_cropped)))]
counts_full = [val_full[i]['count'].item() for i in range(min(5, len(val_full)))]

avg_cropped = sum(counts_cropped) / len(counts_cropped)
avg_full = sum(counts_full) / len(counts_full)

print(f"    With crop_size=256 (samples={len(val_cropped)}): avg tree count = {avg_cropped:.1f}")
print(f"    With crop_size=None (samples={len(val_full)}): avg tree count = {avg_full:.1f}")

if avg_full > avg_cropped * 1.5:
    print(f"    ⚠️  Note: Full images have ~{(avg_full/avg_cropped):.1f}x more trees (expected due to cropping)")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("✅ crop_size=None should be used for VALIDATION and TEST")
print("✅ crop_size=256 can be used for TRAINING (data augmentation)")
print("⚠️  Never mix: training with crops but validation with full images!")
print("=" * 80)
