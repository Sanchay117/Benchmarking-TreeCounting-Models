import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse

# Add treematch and local terratorch to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../treematch")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src/terratorch")))

from models.treematch import Trainer
from data.ps import PlanetScopeStrong, PlanetScopeWeak
from data.gf import GaofenStrong, GaofenWeak
from data.spot import SPOTStrong, SPOTWeak
from torch.utils.data import DataLoader
from itertools import cycle
import huggingface_hub
import datetime
import matplotlib.pyplot as plt
import numpy as np

from terratorch.models import EncoderDecoderFactory

def download_hf_weights(model_name):
    repo_map = {
        "prithvi_vit_300": ("ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "Prithvi_EO_V2_300M.pt"),
        "prithvi_vit_600": ("ibm-nasa-geospatial/Prithvi-EO-2.0-600M", "Prithvi_EO_V2_600M.pt"),
        "prithvi_vit_300_tl": ("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", "Prithvi_EO_V2_300M_TL.pt"),
        "prithvi_vit_600_tl": ("ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL", "Prithvi_EO_V2_600M_TL.pt"),
    }
    if model_name not in repo_map:
        return None
    repo_id, filename = repo_map[model_name]
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {repo_id}...")
        path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".")
        return path
    return filename


class PrithviTreematchBackbone(nn.Module):
    def __init__(self, model_name="prithvi_vit_300"):
        super().__init__()
        # Instantiate the TerraTorch Prithvi model with a segmentation decoder
        # model_name can be "prithvi_vit_300" or "prithvi_vit_600"
        self.model = EncoderDecoderFactory().build_model(
            task="segmentation",
            backbone=model_name,
            decoder="FCNDecoder",
            backbone_pretrained=False, # We load manually
            backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW"], # For PS, GF, SPOT (4 bands)
            num_classes=1,
            backbone_num_frames=1
        )
        
        # Load pretrained weights manually
        ckpt_path = download_hf_weights(model_name)
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            
            # Rename keys to match terratorch's _timm_module wrapping
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder.") and not k.startswith("encoder._timm_module."):
                    new_k = k.replace("encoder.", "encoder._timm_module.", 1)
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            
            # Slice the patch_embed weights from 6 bands to 4 bands
            pe_key = "encoder._timm_module.patch_embed.proj.weight"
            if pe_key in state_dict and state_dict[pe_key].shape[1] == 6:
                state_dict[pe_key] = state_dict[pe_key][:, :4, :, :]
                
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {ckpt_path}. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    
    def forward(self, x):
        # x is (B, C, H, W) where C=5 (4 bands + 1 valid mask from Treematch dataset)
        bands = x[:, :4, :, :]
        
        # Pad from 64x64 to 70x70 so dimensions are divisible by 14 (ViT patch size)
        bands_padded = F.pad(bands, (3, 3, 3, 3), mode='reflect')
        
        # TerraTorch models expect (B, C, T, H, W)
        bands_padded = bands_padded.unsqueeze(2)
        
        out = self.model(bands_padded)
        
        # If TerraTorch returns an object, extract the tensor output
        if hasattr(out, "output"):
            out = out.output
            
        if out.ndim == 3:
            out = out.unsqueeze(1)
            
        # Center crop back from 70x70 to 64x64 to match the density labels
        out_cropped = out[:, :, 3:-3, 3:-3]
        
        return out_cropped

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

def get_dataset(sensor_name, split="train_strong"):
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../tinytrees_dataset/tinytrees/{sensor_name}"))
    if sensor_name == "ps":
        return PlanetScopeWeak(imsize=64, root=os.path.join(root_path, split)) if "weak" in split else PlanetScopeStrong(imsize=64, split=split, root=root_path)
    elif sensor_name == "gf":
        return GaofenWeak(imsize=64, root=os.path.join(root_path, split)) if "weak" in split else GaofenStrong(imsize=64, split=split, root=root_path)
    elif sensor_name == "spot":
        return SPOTWeak(imsize=64, root=os.path.join(root_path, split)) if "weak" in split else SPOTStrong(imsize=64, split=split, root=root_path)
    else:
        raise ValueError(f"Unknown sensor {sensor_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Prithvi-EO-2.0 on TinyTrees with Treematch UOT Trainer")
    parser.add_argument("--model", type=str, default="prithvi_vit_300", choices=["prithvi_vit_300", "prithvi_vit_600", "prithvi_vit_300_tl", "prithvi_vit_600_tl"])
    parser.add_argument("--sensor", type=str, default="ps", choices=["ps", "gf", "spot"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    strong_batch_size = int(args.batch_size * 0.8)
    weak_batch_size = args.batch_size - strong_batch_size
    
    dataset_strong = get_dataset(args.sensor, split="train_strong")
    loader = DataLoader(dataset_strong, batch_size=strong_batch_size, shuffle=True, num_workers=4)
    
    if weak_batch_size > 0:
        dataset_weak = get_dataset(args.sensor, split="train_weak")
        weak_loader = cycle(DataLoader(dataset_weak, batch_size=weak_batch_size, shuffle=True, num_workers=4, drop_last=True))
    else:
        weak_loader = None
    
    # 2. Setup Prithvi Backbone
    backbone = PrithviTreematchBackbone(model_name=args.model)
    
    # 3. Setup Treematch Trainer (reusing Unbalanced Optimal Transport loss)
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
        strong_ratio=0.8, 
        slack=True, 
        convert_density=False,
        max_epoch=args.epochs, 
        alpha=0.8
    )
    trainer.setup(backbone)
    
    print(f"Starting training on {args.sensor} with {args.model}...")
    
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", f"{args.model}_{args.sensor}_{run_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
                
        # Average and save the metrics for this epoch
        logger.step_epoch()
                
        if hasattr(trainer, "scheduler"):
            trainer.scheduler.step()
            
        # Plot training curves at the end of epoch
        plot_curves(logger, checkpoint_dir)
            
        # Keep latest updated
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, f"latest.pth"))
    
    print(f"Training complete! Final model saved to: {checkpoint_dir}")
