import os
import sys
import argparse
import datetime
from pathlib import Path
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

# 1. Import TreeMatch Trainer and datasets FIRST to avoid 'utils' and 'models' namespace collisions
treematch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../treematch"))
sys.path.insert(0, treematch_path)
from models.treematch import Trainer
from data.ps import PlanetScopeStrong, PlanetScopeWeak
from data.gf import GaofenStrong, GaofenWeak
from data.spot import SPOTStrong, SPOTWeak
sys.path.remove(treematch_path)

# 2. Clear conflicting top-level modules from sys.modules
for mod in list(sys.modules.keys()):
    if mod == 'models' or mod.startswith('models.') or mod == 'utils' or mod.startswith('utils.'):
        del sys.modules[mod]

# 3. Import Galileo components
galileo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Galileo"))
sys.path.append(galileo_path)
from single_file_galileo import Encoder, SPACE_TIME_BANDS, SPACE_TIME_BANDS_GROUPS_IDX

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


class GalileoTreematchBackbone(nn.Module):
    def __init__(self, model_variant="galileo_tiny", patch_size=4):
        super().__init__()
        self.model_variant = model_variant
        self.patch_size = patch_size
        
        # Map variant name to folder
        variant_short = model_variant.replace("galileo_", "")
        weights_dir = Path(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../Galileo/data/models", variant_short
        )))
        
        print(f"Loading Galileo encoder from: {weights_dir}")
        self.encoder = Encoder.load_from_folder(weights_dir, device=torch.device("cpu"))
        self.embedding_size = self.encoder.embedding_size
        print(f"Initialized Galileo {model_variant} (embedding_size={self.embedding_size}, patch_size={self.patch_size})")

        # Precompute band and group indices
        self.rgb_indices = [SPACE_TIME_BANDS.index(b) for b in ['B2', 'B3', 'B4']]
        self.nir_index = SPACE_TIME_BANDS.index('B8')
        self.rgb_group = list(SPACE_TIME_BANDS_GROUPS_IDX.keys()).index('S2_RGB')
        self.nir_group = list(SPACE_TIME_BANDS_GROUPS_IDX.keys()).index('S2_NIR_10m')

        # Band counts for empty non-spacetime modalities
        self.num_space_bands = sum(len(v) for v in self.encoder.space_groups.values())
        self.num_space_groups = len(self.encoder.space_groups)
        self.num_time_bands = sum(len(v) for v in self.encoder.time_groups.values())
        self.num_time_groups = len(self.encoder.time_groups)
        self.num_static_bands = sum(len(v) for v in self.encoder.static_groups.values())
        self.num_static_groups = len(self.encoder.static_groups)

        # Fully Convolutional Decoder (upscaling 16x16 -> 32x32 -> 64x64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_size, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        # x is [B, C, H, W] where C=5 (4 bands: Blue, Green, Red, NIR + 1 validity mask)
        bands = x[:, :4, :, :]
        B, C, H, W = bands.shape
        T = 1
        device = x.device
        dtype = x.dtype

        # Construct space-time inputs & masks for Galileo
        s_t_x = torch.zeros((B, H, W, T, len(SPACE_TIME_BANDS)), dtype=dtype, device=device)
        s_t_m = torch.ones((B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX)), dtype=dtype, device=device)

        # Bands in TinyTrees are [Blue, Green, Red, NIR]
        # Blue -> B2, Green -> B3, Red -> B4 (S2_RGB group)
        # NIR -> B8 (S2_NIR_10m group)
        s_t_x[:, :, :, :, self.rgb_indices] = bands[:, 0:3, :, :].permute(0, 2, 3, 1).unsqueeze(3)
        s_t_x[:, :, :, :, [self.nir_index]] = bands[:, 3:4, :, :].permute(0, 2, 3, 1).unsqueeze(3)

        # Unmask only RGB and NIR_10m groups (0 means visible to encoder)
        s_t_m[:, :, :, :, self.rgb_group] = 0
        s_t_m[:, :, :, :, self.nir_group] = 0

        # Empty placeholders for other modalities
        sp_x = torch.zeros((B, H, W, self.num_space_bands), dtype=dtype, device=device)
        sp_m = torch.ones((B, H, W, self.num_space_groups), dtype=dtype, device=device)
        t_x = torch.zeros((B, T, self.num_time_bands), dtype=dtype, device=device)
        t_m = torch.ones((B, T, self.num_time_groups), dtype=dtype, device=device)
        st_x = torch.zeros((B, self.num_static_bands), dtype=dtype, device=device)
        st_m = torch.ones((B, self.num_static_groups), dtype=dtype, device=device)
        months = torch.ones((B, T), device=device, dtype=torch.long) * 6

        output = self.encoder(
            s_t_x, sp_x, t_x, st_x,
            s_t_m, sp_m, t_m, st_m,
            months,
            patch_size=self.patch_size,
            add_layernorm_on_exit=False
        )

        s_t_x_out = output[0]  # [B, H/P, W/P, T, num_groups, D]
        # Average visible tokens across the S2_RGB and S2_NIR_10m groups
        visible_tokens = s_t_x_out[:, :, :, 0, [self.rgb_group, self.nir_group], :].mean(dim=-2)  # [B, H/P, W/P, D]
        spatial_features = visible_tokens.permute(0, 3, 1, 2)  # [B, D, H/P, W/P]

        out = self.decoder(spatial_features)  # [B, 1, 64, 64]
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
    parser = argparse.ArgumentParser(description="Train Galileo on TinyTrees with Treematch UOT Trainer")
    parser.add_argument("--sensor", type=str, required=True, choices=["ps", "gf", "spot"])
    parser.add_argument("--model_variant", type=str, default="galileo_tiny", choices=["galileo_nano", "galileo_tiny", "galileo_base"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--strong_ratio", type=float, default=0.8, help="Ratio of strong data to use (1.0 means 100%% strong, 0.8 means 80%% strong / 20%% weak)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Setup Data
    strong_batch_size = int(args.batch_size * args.strong_ratio)
    weak_batch_size = args.batch_size - strong_batch_size

    dataset_strong = get_dataset(args.sensor, split="train_strong")
    loader = DataLoader(dataset_strong, batch_size=strong_batch_size, shuffle=True, num_workers=4)

    if weak_batch_size > 0:
        dataset_weak = get_dataset(args.sensor, split="train_weak")
        weak_loader = cycle(DataLoader(dataset_weak, batch_size=weak_batch_size, shuffle=True, num_workers=4, drop_last=True))
    else:
        weak_loader = None

    # 2. Setup Galileo Backbone + FCNDecoder
    backbone = GalileoTreematchBackbone(model_variant=args.model_variant, patch_size=args.patch_size)

    # 3. Setup Treematch Trainer
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

    print(f"Starting training {args.model_variant} on {args.sensor} ({args.epochs} epochs, strong_ratio={args.strong_ratio}, lr={args.lr}).")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
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

        # Update training curves and save latest checkpoint
        plot_curves(logger, checkpoint_dir)
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, "checkpoint_best.pth"))
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, "latest.pth"))

    print(f"Training complete! Final model saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
