#!/usr/bin/env python
"""
Comprehensive verification for Swin-UNet implementation
Checks all components are correctly set up and ready for benchmarking
"""

import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_check(status, message):
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {message}")

def check_file_exists(path, description=""):
    """Check if a file exists"""
    desc = f" ({description})" if description else ""
    if os.path.isfile(path):
        size_kb = os.path.getsize(path) / 1024
        print_check(True, f"{os.path.basename(path)}{desc} - {size_kb:.1f} KB")
        return True
    else:
        print_check(False, f"{os.path.basename(path)}{desc} - MISSING")
        return False

def check_dir_exists(path, description=""):
    """Check if a directory exists"""
    desc = f" ({description})" if description else ""
    if os.path.isdir(path):
        print_check(True, f"{os.path.basename(path)}/{desc}")
        return True
    else:
        print_check(False, f"{os.path.basename(path)}/{desc} - MISSING")
        return False

def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SWIN-UNET Implementation Verification".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    all_ok = True
    
    # 1. Check core implementation files
    print_header("1. Core Implementation Files")
    files_to_check = {
        "network/swin_unet.py": "Model wrapper",
        "network/swin_transformer_unet_skip_expand_decoder_sys.py": "Official implementation",
        "network/__init__.py": "Network package",
        "datasets/kcl_london.py": "Dataset loader",
        "datasets/__init__.py": "Dataset package",
        "train_swin_unet_kcl.py": "Training script",
        "test_swin_unet_kcl.py": "Evaluation script",
        "__init__.py": "Main package",
    }
    
    for file_path, desc in files_to_check.items():
        full_path = os.path.join("/home/ashank/TreeCounting_Benchmark/SwinUNet", file_path)
        if not check_file_exists(full_path, desc):
            all_ok = False
    
    # 2. Check documentation files
    print_header("2. Documentation Files")
    docs = {
        "README.md": "Full documentation",
        "QUICKSTART.md": "Quick start guide",
        "IMPLEMENTATION_SUMMARY.md": "Implementation summary",
        "INTEGRATION.md": "Integration guide",
        "requirements.txt": "Dependencies",
    }
    
    for file_path, desc in docs.items():
        full_path = os.path.join("/home/ashank/TreeCounting_Benchmark/SwinUNet", file_path)
        check_file_exists(full_path, desc)
    
    # 3. Check directories
    print_header("3. Directory Structure")
    dirs = {
        "network": "Network module",
        "datasets": "Datasets module",
        "ckpts": "Checkpoints (may be empty)",
        "eval": "Evaluation results (may be empty)",
        "predictions": "Predictions (may be empty)",
        "pretrained_ckpt": "Pre-trained weights (empty)",
    }
    
    for dir_path, desc in dirs.items():
        full_path = os.path.join("/home/ashank/TreeCounting_Benchmark/SwinUNet", dir_path)
        check_dir_exists(full_path, desc)
    
    # 4. Test imports
    print_header("4. Testing Python Imports")
    
    try:
        print("  Testing imports...")
        sys.path.insert(0, "/home/ashank/TreeCounting_Benchmark/SwinUNet")
        
        from network.swin_unet import SwinUNet
        print_check(True, "✓ SwinUNet model class imported")
        
        from datasets.kcl_london import KCLLondonSwinUNetDataset
        print_check(True, "✓ Dataset loader imported")
        
        import torch
        print_check(True, "✓ PyTorch imported")
        
    except ImportError as e:
        print_check(False, f"Import failed: {e}")
        all_ok = False
    
    # 5. Check dataset accessibility
    print_header("5. Dataset Accessibility")
    
    dataset_root = "/home/ashank/TreeCounting_Benchmark/TreeFormer/datasets"
    splits = ["train_data", "valid_data", "test_data"]
    
    for split in splits:
        split_path = os.path.join(dataset_root, split)
        images_dir = os.path.join(split_path, "images")
        gt_dir = os.path.join(split_path, "ground_truth")
        
        if os.path.isdir(images_dir) and os.path.isdir(gt_dir):
            num_images = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
            print_check(True, f"{split}: {num_images} images found")
        else:
            print_check(False, f"{split}: Dataset not found")
            all_ok = False
    
    # 6. Model instantiation test
    print_header("6. Model Instantiation Test")
    
    try:
        print("  Creating SwinUNet model...")
        model = SwinUNet(img_size=256, patch_size=4, in_chans=3, num_classes=1)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_check(True, f"Model created successfully")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("  Testing forward pass...")
        img_size = model.model.patch_embed.img_size[0]
        x = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            y = model(x)
        
        print_check(True, f"Forward pass successful")
        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {y.shape}")
        
    except Exception as e:
        print_check(False, f"Model test failed: {e}")
        all_ok = False
    
    # 7. Configuration files check
    print_header("7. Configuration Files")
    
    config_scripts = {
        "run_configs.sh": "Configuration templates",
        "verify_installation.py": "Installation verification",
        "checklist.py": "Implementation checklist",
    }
    
    for file_path, desc in config_scripts.items():
        full_path = os.path.join("/home/ashank/TreeCounting_Benchmark/SwinUNet", file_path)
        check_file_exists(full_path, desc)
    
    # Summary
    print_header("Summary & Next Steps")
    
    if all_ok:
        print_check(True, "All critical components are in place!")
        print("""
        ✓ Ready to start benchmarking!
        
        Quick start:
        1. Install dependencies:
           cd /home/ashank/TreeCounting_Benchmark/SwinUNet
           pip install -r requirements.txt
        
        2. Quick test (5 minutes):
           python train_swin_unet_kcl.py --epochs 5 --device 0
        
        3. Full training (60 minutes):
           python train_swin_unet_kcl.py --epochs 100 --batch-size 8 --device 0
        
        4. Evaluate:
           python test_swin_unet_kcl.py --model-path ckpts/swin_unet_kcl_*/best_mae.pth
        
        For detailed instructions, see:
        - QUICKSTART.md - Quick start guide
        - README.md - Full documentation
        - INTEGRATION.md - Integration with other models
        """)
    else:
        print_check(False, "Some components are missing. Check items marked with ✗ above.")
    
    print("\n" + "█"*70 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
