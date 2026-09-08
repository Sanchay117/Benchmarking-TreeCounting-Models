import os
import sys
import argparse
import datetime
import numpy as np

# 1. Import TreeMatch Trainer and datasets FIRST to avoid 'utils' and 'models' collision
treematch_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "treematch")
sys.path.insert(0, treematch_path)
from models.treematch import Trainer
from data.ps import PlanetScopeStrong
from data.gf import GaofenStrong
from data.spot import SPOTStrong
sys.path.remove(treematch_path)

# 2. Clear conflicting top-level modules from sys.modules so AnySat can load its own modules
for mod in list(sys.modules.keys()):
    if mod == 'models' or mod.startswith('models.') or mod == 'utils' or mod.startswith('utils.'):
        del sys.modules[mod]

# 3. Import AnySat
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AnySat"))
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
        model_path = f"/home/ashank/TreeCounting_Benchmark/AnySat/.models/AnySat{'_full' if model_variant == 'anysat_full' else ''}.pth"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", type=str, required=True, choices=["ps", "gf", "spot", "all"])
    parser.add_argument("--model_variant", type=str, default="anysat", choices=["anysat", "anysat_full"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    root_path = f"../tinytrees_dataset/tinytrees/{args.sensor}"
    if args.sensor == "ps":
        train_dataset = PlanetScopeStrong(imsize=64, split="train_strong", root=root_path)
    elif args.sensor == "gf":
        train_dataset = GaofenStrong(imsize=64, split="train_strong", root=root_path)
    elif args.sensor == "spot":
        train_dataset = SPOTStrong(imsize=64, split="train_strong", root=root_path)
    
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

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
        lr=1e-5, 
        strong_ratio=1.0, 
        slack=True, 
        convert_density=False,
        max_epoch=args.epochs, 
        alpha=0.8
    )
    trainer.setup(backbone)

    run_name = f"{args.model_variant}_{args.sensor}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Starting training AnySat for {args.sensor} data over {args.epochs} epochs.")
    logger = SimpleLogger()
    
    for epoch in range(args.epochs):
        trainer.train()
        for step, (inputs, valid, gt_discrete) in enumerate(loader):
            trainer.train_step(inputs, valid, gt_discrete, logger=logger)
            if step % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs} - Step {step}/{len(loader)}")
                
        logger.step_epoch()
        if hasattr(trainer, "scheduler"):
            trainer.scheduler.step()
            
        plot_curves(logger, checkpoint_dir)
            
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_best.pth"))
    
    print(f"Training complete! Saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()
