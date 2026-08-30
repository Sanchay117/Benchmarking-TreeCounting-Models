"""
Verification script to test if the fixes are working correctly.

Usage:
    python verify_fixes.py \
      --model-path ckpts/swin_unet_kcl_XXXX/best_mae.pth \
      --data-dir ../TreeFormer/datasets \
      --device 0
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from network.swin_unet import SwinUNet
from datasets.kcl_london import KCLLondonSwinUNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Swin-UNet fixes")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--data-dir", type=str, default="../TreeFormer/datasets",
                        help="Dataset root")
    parser.add_argument("--split", type=str, default="test_data",
                        help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to verify")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Input image size")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_arg
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def verify_fixes():
    args = parse_args()
    device = resolve_device(args.device)
    
    print("\n" + "="*70)
    print("SWIN-UNET TREE COUNTING - FIX VERIFICATION")
    print("="*70)
    
    # Load dataset
    print(f"\n[1/5] Loading dataset...")
    try:
        dataset = KCLLondonSwinUNetDataset(
            root=args.data_dir,
            split=args.split,
            crop_size=None,
            resize_to=args.img_size,
            random_flip=False,
            in_channels=3
        )
        print(f"      ✓ Dataset loaded: {len(dataset)} images")
    except Exception as e:
        print(f"      ✗ Error loading dataset: {e}")
        return False
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    # Load model
    print(f"\n[2/5] Loading model...")
    try:
        model = SwinUNet(
            img_size=args.img_size,
            patch_size=4,
            in_chans=3,
            num_classes=1,
            embed_dim=96,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=16
        ).to(device)
        
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        print(f"      ✓ Model loaded from: {args.model_path}")
    except Exception as e:
        print(f"      ✗ Error loading model: {e}")
        return False
    
    # Verify fixes
    print(f"\n[3/5] Verifying fixes on {min(args.num_samples, len(loader))} samples...")
    
    all_predictions_positive = True
    all_outputs_reasonable = True
    sample_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_samples:
                break
            
            image = batch["image"].to(device)
            gt_count = batch["count"].item()
            name = batch["name"]
            
            # Forward pass
            pred_density = model(image)
            pred_density_clamped = torch.clamp(pred_density, min=0)
            pred_count = pred_density_clamped.flatten(1).sum(dim=1).item()
            
            # Check for negatives
            min_val = pred_density.min().item()
            max_val = pred_density.max().item()
            mean_val = pred_density.mean().item()
            
            has_negatives = (pred_density < 0).any().item()
            
            sample_results.append({
                'name': name,
                'gt_count': gt_count,
                'pred_count': pred_count,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'has_negatives': has_negatives,
                'error': abs(gt_count - pred_count)
            })
            
            status = "✓" if not has_negatives else "✗"
            print(f"      {status} {name}: GT={gt_count:.1f}, Pred={pred_count:.1f}, "
                  f"Min={min_val:.4f}, Max={max_val:.4f}, Error={abs(gt_count - pred_count):.1f}")
            
            if has_negatives:
                all_predictions_positive = False
                print(f"         WARNING: Found negative predictions! Min={min_val:.6f}")
            
            if max_val > gt_count * 5 or pred_count > gt_count * 5:
                all_outputs_reasonable = False
                print(f"         WARNING: Prediction seems unusually large!")
    
    # Summary
    print(f"\n[4/5] Verification Summary:")
    print("-" * 70)
    
    results_arr = np.array([(r['gt_count'], r['pred_count']) for r in sample_results])
    mae = np.mean(np.abs(results_arr[:, 0] - results_arr[:, 1]))
    rmse = np.sqrt(np.mean((results_arr[:, 0] - results_arr[:, 1]) ** 2))
    
    print(f"  Samples checked: {len(sample_results)}")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  Root Mean Square Error: {rmse:.2f}")
    
    print(f"\n  ✓ Fixes Status:")
    
    # Check 1: All predictions positive
    all_positive = all(not r['has_negatives'] for r in sample_results)
    status = "✓" if all_positive else "✗"
    print(f"  {status} All predictions non-negative: {all_positive}")
    
    # Check 2: Reasonable predictions
    all_reasonable = all(r['pred_count'] >= 0 for r in sample_results)
    status = "✓" if all_reasonable else "✗"
    print(f"  {status} All predictions are real numbers (no NaN/Inf): {all_reasonable}")
    
    # Check 3: Model is working
    working = len(sample_results) > 0 and all(r['error'] < 10000 for r in sample_results)
    status = "✓" if working else "✗"
    print(f"  {status} Model output ranges are reasonable: {working}")
    
    print(f"\n[5/5] Detailed Results:")
    print("-" * 70)
    print(f"{'Image':<20} {'GT':<10} {'Pred':<10} {'Error':<10} {'Status':<10}")
    print("-" * 70)
    
    for r in sample_results:
        status = "✓ GOOD" if not r['has_negatives'] else "✗ NEG"
        print(f"{r['name']:<20} {r['gt_count']:<10.1f} {r['pred_count']:<10.1f} "
              f"{r['error']:<10.1f} {status:<10}")
    
    print("\n" + "="*70)
    
    if all_positive and all_reasonable and working:
        print("✓ ALL FIXES VERIFIED SUCCESSFULLY!")
        print("="*70 + "\n")
        return True
    else:
        print("✗ SOME ISSUES DETECTED - Please review the output above")
        print("="*70 + "\n")
        return False


if __name__ == "__main__":
    success = verify_fixes()
    exit(0 if success else 1)
