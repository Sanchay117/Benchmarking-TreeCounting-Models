"""
Evaluation/Benchmarking script for Swin-UNet on KCL London tree counting dataset.

Usage:
    python test_swin_unet_kcl.py \
      --data-dir ../TreeFormer/datasets \
      --split test_data \
      --model-path ckpts/swin_unet_kcl_<timestamp>/best_mae.pth \
      --device 0
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from scipy.io import savemat

from network.swin_unet import SwinUNet
from datasets.kcl_london import KCLLondonSwinUNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin-UNet on KCL London dataset")
    parser.add_argument("--data-dir", type=str, default="../TreeFormer/datasets",
                        help="Dataset root")
    parser.add_argument("--split", type=str, default="test_data",
                        help="Dataset split to evaluate")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Evaluation batch size")
    parser.add_argument("--output-dir", type=str, default="eval",
                        help="Output directory for evaluation results")
    parser.add_argument("--pred-dir", type=str, default="predictions",
                        help="Directory for predicted density maps")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--in-channels", type=int, default=3,
                        help="Number of input channels")
    return parser.parse_args()


def prepare_eval_run_dir(eval_root: str, prefix: str) -> str:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(eval_root, f"{prefix}_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_hyperparams(run_dir: str, args):
    hparams_path = os.path.join(run_dir, "hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    return hparams_path


def resolve_device(device_arg: str) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_arg
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    arch_list = set(torch.cuda.get_arch_list())
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    if sm not in arch_list:
        print(
            f"CUDA arch {sm} is not supported by this PyTorch build ({sorted(arch_list)}). "
            "Falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")


def load_checkpoint(model: torch.nn.Module, model_path: str, device: torch.device):
    """Load checkpoint into model"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return int(checkpoint.get("epoch", -1))

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
        return -1

    raise ValueError("Unsupported checkpoint format")


def compute_metrics(pred_counts, gt_counts):
    """Compute evaluation metrics"""
    pred_counts = np.asarray(pred_counts, dtype=np.float32)
    gt_counts = np.asarray(gt_counts, dtype=np.float32)
    
    # MAE
    mae = np.mean(np.abs(pred_counts - gt_counts))
    
    # MSE (RMSE)
    mse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))
    
    # R2 Score
    ss_res = np.sum((gt_counts - pred_counts) ** 2)
    ss_tot = np.sum((gt_counts - np.mean(gt_counts)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return mae, mse, r2


def evaluate(args):
    device = resolve_device(args.device)
    run_dir = prepare_eval_run_dir(args.output_dir, "swin_unet_test")
    hparams_path = save_hyperparams(run_dir, args)
    os.makedirs(args.pred_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Hyperparameters saved to: {hparams_path}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = KCLLondonSwinUNetDataset(
        root=args.data_dir,
        split=args.split,
        crop_size=None,  # ✅ NO CROPPING for test - use full images
        resize_to=args.img_size,
        random_flip=False,
        in_channels=args.in_channels
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    print(f"Evaluating on {len(dataset)} images")
    
    # Create model
    print("\nCreating model...")
    model = SwinUNet(
        img_size=args.img_size,
        patch_size=4,
        in_chans=args.in_channels,
        num_classes=1,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=16
    ).to(device)
    
    load_checkpoint(model, args.model_path, device)
    model.eval()
    
    print(f"Model loaded from: {args.model_path}")
    
    # Evaluation
    all_results = []
    abs_errors = []
    pred_counts = []
    gt_counts = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            gt_density = batch["density"].to(device, non_blocking=True)
            gt_count = batch["count"].to(device, non_blocking=True)
            name = batch["name"]
            
            # Forward pass
            pred_density = model(image)
            
            # Ensure same spatial dimensions as ground truth
            if pred_density.shape != gt_density.shape:
                pred_density = torch.nn.functional.interpolate(
                    pred_density, size=gt_density.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Calculate predicted count
            pred_count = pred_density.flatten(1).sum(dim=1)
            
            # Calculate error
            error = gt_count - pred_count
            abs_errors.append(torch.abs(error))
            
            # Store results
            pred_count_val = float(pred_count.item())
            gt_count_val = float(gt_count.item())
            error_val = gt_count_val - pred_count_val
            
            pred_counts.append(pred_count_val)
            gt_counts.append(gt_count_val)
            all_results.append([name[0] if isinstance(name, (list, tuple)) else name, 
                               gt_count_val, pred_count_val, error_val])
            
            if (batch_idx + 1) % 50 == 0 or batch_idx == len(loader) - 1:
                print(f"  Processed {batch_idx + 1}/{len(loader)}")
            
            # Save predictions
            pred_density_cpu = pred_density[0].detach().cpu().numpy()
            gt_density_cpu = gt_density[0].detach().cpu().numpy()
            image_cpu = image[0].detach().cpu().numpy()
            
            name_str = name[0] if isinstance(name, (list, tuple)) else name
            
            savemat(
                os.path.join(args.pred_dir, f"{name_str}.mat"),
                {
                    "pred_density": np.squeeze(pred_density_cpu),
                    "gt_density": np.squeeze(gt_density_cpu),
                    "image": np.squeeze(image_cpu),
                    "pred_count": pred_count_val,
                    "gt_count": gt_count_val,
                    "error": error_val
                }
            )
    
    # Compute metrics
    print("\nComputing metrics...")
    abs_errors = torch.cat(abs_errors)
    mae = float(abs_errors.mean().item())
    mse = np.sqrt(np.mean((np.array(gt_counts) - np.array(pred_counts)) ** 2))
    r2 = 1.0 - (np.sum((np.array(gt_counts) - np.array(pred_counts)) ** 2) / 
                np.sum((np.array(gt_counts) - np.mean(gt_counts)) ** 2))
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results on {args.split}")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Metrics:")
    print(f"  MAE (Mean Absolute Error):  {mae:.4f}")
    print(f"  RMSE (Root Mean Square Error): {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Number of test images: {len(all_results)}")
    print(f"{'='*60}\n")
    
    # Save results
    final_stats = {
        "model_path": args.model_path,
        "split": args.split,
        "num_images": len(all_results),
        "mae": mae,
        "rmse": mse,
        "r2": r2,
        "predictions": all_results
    }
    
    final_stats_path = os.path.join(run_dir, "final_stats.json")
    with open(final_stats_path, "w") as f:
        json.dump(final_stats, f, indent=2)
    
    # Save text summary
    text_lines = [
        f"Swin-UNet Evaluation Results on {args.split}\n",
        f"{'='*60}\n",
        f"Model: {args.model_path}\n",
        f"Number of images: {len(all_results)}\n",
        f"MAE: {mae:.4f}\n",
        f"RMSE: {mse:.4f}\n",
        f"R2: {r2:.4f}\n",
        f"{'='*60}\n\n",
        f"{'Image':<30} {'GT':<10} {'Pred':<10} {'Error':<10}\n",
        f"{'-'*60}\n"
    ]
    
    for result in all_results:
        name, gt, pred, err = result
        text_lines.append(f"{name:<30} {gt:<10.2f} {pred:<10.2f} {err:<10.2f}\n")
    
    summary_path = os.path.join(run_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(text_lines)
    
    print(f"Saved results:")
    print(f"  Final stats: {final_stats_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Predictions: {args.pred_dir}")
    print(f"  Run directory: {run_dir}")


if __name__ == "__main__":
    evaluate(parse_args())
