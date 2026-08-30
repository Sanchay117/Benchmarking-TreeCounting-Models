"""
Training script for Swin-UNet on KCL London tree counting dataset.

Usage:
    python train_swin_unet_kcl.py \
      --data-dir ../TreeFormer/datasets \
      --train-split train_data \
      --val-split valid_data \
      --epochs 100 \
      --batch-size 8 \
      --crop-size 256 \
      --lr 1e-4 \
      --device 0
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from network.swin_unet import SwinUNet
from datasets.kcl_london import KCLLondonSwinUNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin-UNet on KCL London dataset")
    parser.add_argument("--data-dir", type=str, default="../TreeFormer/datasets", 
                        help="Dataset root")
    parser.add_argument("--train-split", type=str, default="train_data",
                        help="Training split name")
    parser.add_argument("--val-split", type=str, default="valid_data",
                        help="Validation split name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Crop size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="Early stopping patience (epochs with no improvement)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--output-dir", type=str, default="ckpts",
                        help="Output directory for checkpoints")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--in-channels", type=int, default=3,
                        help="Number of input channels (1=grayscale, 3=RGB)")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to pretrained checkpoint")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to resume training from")
    return parser.parse_args()


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


def create_run_dir(output_dir: str, prefix: str) -> str:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, f"{prefix}_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_hparams(run_dir: str, args):
    hparams_path = os.path.join(run_dir, "hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    return hparams_path


def mse_loss(pred, target):
    """Mean squared error loss"""
    return ((pred - target) ** 2).mean()


def mae_loss(pred, target):
    """Mean absolute error loss"""
    return (pred - target).abs().mean()


def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    
    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        density = batch["density"].to(device, non_blocking=True)
        
        # Forward pass
        pred_density = model(image)
        
        # Ensure same spatial dimensions
        if pred_density.shape != density.shape:
            pred_density = torch.nn.functional.interpolate(
                pred_density, size=density.shape[2:], mode='bilinear', align_corners=False
            )
        
        loss = loss_fn(pred_density, density)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate MAE on total count
        with torch.no_grad():
            pred_count = pred_density.flatten(1).sum(dim=1)
            gt_count = density.flatten(1).sum(dim=1)
            mae = (pred_count - gt_count).abs().mean().item()
            total_mae += mae
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_mae = total_mae / (batch_idx + 1)
            print(f"  Batch {batch_idx + 1}/{len(loader)}: Loss={avg_loss:.4f}, MAE={avg_mae:.2f}")
    
    return total_loss / len(loader), total_mae / len(loader)


def validate(model, loader, device):
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            density = batch["density"].to(device, non_blocking=True)
            
            pred_density = model(image)
            
            # Ensure same spatial dimensions
            if pred_density.shape != density.shape:
                pred_density = torch.nn.functional.interpolate(
                    pred_density, size=density.shape[2:], mode='bilinear', align_corners=False
                )
            
            loss = ((pred_density - density) ** 2).mean()
            total_loss += loss.item()
            
            # Calculate metrics on total count
            pred_count = pred_density.flatten(1).sum(dim=1)
            gt_count = density.flatten(1).sum(dim=1)
            
            mae = (pred_count - gt_count).abs().mean().item()
            mse = ((pred_count - gt_count) ** 2).mean().item()
            
            total_mae += mae
            total_mse += mse
    
    return {
        'mae': total_mae / len(loader),
        'mse': np.sqrt(total_mse / len(loader)),
        'loss': total_loss / len(loader)
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    
    # Create run directory
    run_dir = create_run_dir(args.output_dir, "swin_unet_kcl")
    hparams_path = save_hparams(run_dir, args)
    print(f"Run directory: {run_dir}")
    print(f"Hyperparameters saved to: {hparams_path}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = KCLLondonSwinUNetDataset(
        root=args.data_dir,
        split=args.train_split,
        crop_size=args.crop_size,
        random_flip=True,
        in_channels=args.in_channels
    )
    
    val_dataset = KCLLondonSwinUNetDataset(
        root=args.data_dir,
        split=args.val_split,
        crop_size=None,  # Full images for validation
        resize_to=args.img_size,  # Resize to model input size (256x256)
        random_flip=False,
        in_channels=args.in_channels
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
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
        window_size=16,
        drop_path_rate=0.1
    ).to(device)
    
    # Load pretrained if provided
    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Loss function
    loss_fn = mse_loss
    
    # Training loop
    best_mae = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_mse': []
    }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics = validate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_mse'].append(val_metrics['mse'])
        
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.2f}, Val MSE: {val_metrics['mse']:.2f}")
        
        # ✅ FIX: Early stopping based on validation MAE
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            patience_counter = 0  # Reset patience counter
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_mae": val_metrics['mae']
            }
            
            best_path = os.path.join(run_dir, "best_mae.pth")
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved with MAE: {best_mae:.2f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{args.early_stopping_patience} epochs")
            
            if patience_counter >= args.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered! Best validation MAE: {best_mae:.2f}")
                break
        
        # Always save latest checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_mae": val_metrics['mae']
        }
        latest_path = os.path.join(run_dir, "latest.pth")
        torch.save(checkpoint, latest_path)
        
        scheduler.step()
    
    # Save history
    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Checkpoints saved to: {run_dir}")
    print(f"Best validation MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()
