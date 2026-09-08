import os
import sys
import argparse

# 1. Import TreeMatch Trainer, datasets, and evaluation functions FIRST
treematch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../treematch"))
sys.path.insert(0, treematch_path)

from models.treematch import Trainer
from data.ps import PlanetScopeStrong
from data.gf import GaofenStrong
from data.spot import SPOTStrong
from train import get_preds, evaluate_at_fixed_scale
sys.path.remove(treematch_path)

# 2. Clear conflicting top-level modules from sys.modules
for mod in list(sys.modules.keys()):
    if mod == 'models' or mod.startswith('models.') or mod == 'utils' or mod.startswith('utils.'):
        del sys.modules[mod]

import torch
from torch.utils.data import DataLoader

from train_galileo import GalileoTreematchBackbone


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
    parser = argparse.ArgumentParser(description="Test Galileo on TinyTrees")
    parser.add_argument("--sensor", type=str, default="ps", choices=["ps", "gf", "spot"])
    parser.add_argument("--model_variant", type=str, default="galileo_tiny", choices=["galileo_nano", "galileo_tiny", "galileo_base"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth checkpoint file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Test Data
    dataset = get_test_dataset(args.sensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    gsd_map = {"ps": 3.0, "gf": 0.8, "spot": 1.5}
    gsd_m = gsd_map.get(args.sensor, 3.0)

    # 2. Setup Galileo Backbone + FCNDecoder
    print(f"Initializing Galileo backbone ({args.model_variant})...")
    backbone = GalileoTreematchBackbone(model_variant=args.model_variant, patch_size=args.patch_size)

    print(f"Loading checkpoint: {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)

    # 3. Setup Treematch Trainer for evaluation
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
        max_epoch=1,
        alpha=0.8
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
            if k == 'nmae':
                print(f"{k.upper()}: {v*100:.2f}% (raw: {v:.4f})")
            else:
                print(f"{k.upper()}: {v:.4f}")
        print("="*30)
