import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add treematch and local terratorch to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../treematch")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src/terratorch")))

from models.treematch import Trainer
from data.ps import PlanetScopeStrong
from data.gf import GaofenStrong
from data.spot import SPOTStrong
from train import get_preds, evaluate_at_fixed_scale

# Import the backbone from train script
from train_prithvi import PrithviTreematchBackbone

def get_test_dataset(sensor_name):
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../tinytrees_dataset/tinytrees/{sensor_name}"))
    if sensor_name == "ps":
        return PlanetScopeStrong(imsize=64, split="test", root=root_path)
    elif sensor_name == "gf":
        return GaofenStrong(imsize=64, split="test", root=root_path)
    elif sensor_name == "spot":
        return SPOTStrong(imsize=64, split="test", root=root_path)
    else:
        raise ValueError(f"Unknown sensor {sensor_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Prithvi-EO-2.0 on TinyTrees")
    parser.add_argument("--model", type=str, default="prithvi_vit_300", choices=["prithvi_vit_300", "prithvi_vit_600", "prithvi_vit_300_tl", "prithvi_vit_600_tl"])
    parser.add_argument("--sensor", type=str, default="ps", choices=["ps", "gf", "spot"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth checkpoint file")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    dataset = get_test_dataset(args.sensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    gsd_map = {"ps": 3.0, "gf": 0.8, "spot": 1.5}
    gsd_m = gsd_map.get(args.sensor, 3.0)
    
    # 2. Setup Prithvi Backbone
    print(f"Initializing backbone: {args.model}...")
    backbone = PrithviTreematchBackbone(model_name=args.model)
    
    print(f"Loading checkpoint: {args.checkpoint}...")
    # Load the fine-tuned checkpoint
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)
    
    # 3. Setup Treematch Trainer for evaluation
    trainer = Trainer(
        imsize=64, downscale_ratio=1, device=device, 
        wc=1, wot=1, reg=0.005, reg_m=0.2, num_of_iter_in_ot=100, 
        lr=1e-5, strong_ratio=1, slack=True, convert_density=False,
        max_epoch=1, alpha=0.8
    )
    trainer.setup(backbone)
    trainer.eval()
    
    print(f"Starting evaluation on {args.sensor} test set (Total batches: {len(loader)})...")
    with torch.no_grad():
        test_pred, test_target, test_valid = get_preds(loader, trainer, device)
        test_metrics = evaluate_at_fixed_scale(test_pred, test_target, gsd_m=gsd_m, masks=test_valid, eval_patch_size=64)
        
        print("\n" + "="*30)
        print("=== Test Results ===")
        print("="*30)
        for k, v in test_metrics.items():
            print(f"{k.upper()}: {v:.4f}")
        print("="*30)
