import os
import sys
import argparse
import datetime
import numpy as np

# 1. Import TreeMatch Trainer and datasets FIRST to avoid 'utils' and 'models' collision
treematch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../treematch"))
sys.path.insert(0, treematch_path)
from models.treematch import Trainer
from data.ps import PlanetScopeStrong, PlanetScopeWeak
from data.gf import GaofenStrong, GaofenWeak
from data.spot import SPOTStrong, SPOTWeak
sys.path.remove(treematch_path)

# 2. Clear conflicting top-level modules from sys.modules so AnySat can load its own modules
for mod in list(sys.modules.keys()):
    if mod == 'models' or mod.startswith('models.') or mod == 'utils' or mod.startswith('utils.'):
        del sys.modules[mod]

# 3. Import AnySat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../AnySat")))
from hubconf import AnySat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
import matplotlib.pyplot as plt

# For logging
class SimpleLogger:
    def __init__(self):
        self.history = {'train/total_loss': [], 'train/count_loss': [], 'train/mae': []}
        self.current_epoch = {'train/total_loss': [], 'train/count_loss': [], 'train/mae': []}
        
    def log(self, metrics):
        for k, v in metrics.items():
            if k in self.current_epoch:
                self.current_epoch[k].append(v)
                
    def step_epoch(self):
        for k in self.history.keys():
            if self.current_epoch[k]:
                avg_val = sum(self.current_epoch[k]) / len(self.current_epoch[k])
                self.history[k].append(avg_val)
                self.current_epoch[k] = []

def plot_curves(logger, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(logger.history['train/total_loss'])
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    
    axes[1].plot(logger.history['train/count_loss'])
    axes[1].set_title('Count Loss')
    axes[1].set_xlabel('Epoch')
    
    axes[2].plot(logger.history['train/mae'])
    axes[2].set_title('MAE')
    axes[2].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close(fig)

class AnySatTreematchBackbone(nn.Module):
    def __init__(self, model_size='base', model_variant='anysat'):
        super().__init__()
        self.anysat = AnySat(model_size=model_size, flash_attn=False)
        model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), f"../AnySat/.models/AnySat{'_full' if model_variant == 'anysat_full' else ''}.pth"
        ))
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.anysat.load_state_dict(state_dict, strict=False)
            print(f"Loaded {model_variant} weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}")
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1536, 128, kernel_size=8, stride=8),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        bands = x[:, :4, :, :]
        data = {"naip": bands}
        features = self.anysat(data, patch_size=10, output='dense', output_modality='naip')
        features = features.permute(0, 3, 1, 2)
        out = self.decoder(features)
        return out


def get_dataset(sensor_name, split="train_strong"):
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../tinytrees_dataset/tinytrees/{sensor_name}"))
    if sensor_name == "ps":
        return PlanetScopeWeak(imsize=64, root=os.path.join(root_path, split)) if "weak" in split else PlanetScopeStrong(imsize=64, split=split, root=root_path)
    elif sensor_name == "gf":
        return GaofenWeak(imsize=64, root=os.path.join(root_path, split)) if "weak" in split else GaofenStrong(imsize=64, split=split, root=root_path)
    elif sensor_name == "spot":
        if "weak" in split:
            print("Notice: SPOT-6 weak imagery is not bundled in TinyTrees dataset. Using SPOTStrong for training.")
            return SPOTStrong(imsize=64, split="train_strong", root=root_path)
        return SPOTStrong(imsize=64, split=split, root=root_path)
    else:
        raise ValueError(f"Unknown sensor {sensor_name}")


def main():
    parser = argparse.ArgumentParser(description="Train AnySat on TinyTrees with Treematch UOT Trainer")
    parser.add_argument("--sensor", type=str, required=True, choices=["ps", "gf", "spot"])
    parser.add_argument("--model_variant", type=str, default="anysat", choices=["anysat", "anysat_full"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--strong_ratio", type=float, default=1.0, help="Ratio of strong data to use (1.0 means 100%% strong, 0.8 means 80%% strong / 20%% weak)")
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Data
    strong_batch_size = int(args.batch_size * args.strong_ratio)
    weak_batch_size = args.batch_size - strong_batch_size

    dataset_strong = get_dataset(args.sensor, split="train_strong")
    loader = DataLoader(dataset_strong, batch_size=strong_batch_size, shuffle=True, num_workers=4)

    if weak_batch_size > 0:
        dataset_weak = get_dataset(args.sensor, split="train_weak")
        weak_loader = cycle(DataLoader(dataset_weak, batch_size=weak_batch_size, shuffle=True, num_workers=4, drop_last=True))
    else:
        weak_loader = None

    # Model
    backbone = AnySatTreematchBackbone(model_size='base', model_variant=args.model_variant)
    
    trainer = Trainer(
        imsize=64, 
        downscale_ratio=1, 
        device=device, 
        wc=1, 
        wot=1, 
        reg=0.005, 
        reg_m=0.2,
        num_of_iter_in_ot=100, 
        lr=args.lr, 
        strong_ratio=args.strong_ratio, 
        slack=True, 
        convert_density=False,
        max_epoch=args.epochs, 
        alpha=0.8
    )
    trainer.setup(backbone)

    weak_pct = int(round((1.0 - args.strong_ratio) * 100))
    run_name = f"{args.model_variant}_{args.sensor}_weak{weak_pct}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Starting training {args.model_variant} for {args.sensor} ({args.epochs} epochs, strong_ratio={args.strong_ratio}, lr={args.lr}).")
    logger = SimpleLogger()
    
    for epoch in range(args.epochs):
        trainer.train()
        for step, (inputs, valid, gt_discrete) in enumerate(loader):
            if weak_loader is not None:
                inputs_weak, valid_weak, gt_discrete_weak = next(weak_loader)
                inputs = torch.cat([inputs, inputs_weak], dim=0)
                valid = torch.cat([valid, valid_weak], dim=0)
                gt_discrete = torch.cat([gt_discrete, gt_discrete_weak], dim=0)

            trainer.train_step(inputs, valid, gt_discrete, logger=logger)
            if step % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs} - Step {step}/{len(loader)}")
                
        logger.step_epoch()
        if hasattr(trainer, "scheduler"):
            trainer.scheduler.step()
            
        plot_curves(logger, checkpoint_dir)
            
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, "checkpoint_best.pth"))
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, "latest.pth"))
    
    print(f"Training complete! Saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()
