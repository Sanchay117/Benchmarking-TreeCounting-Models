"""
Installation verification script for Swin-UNet Tree Counting
"""

import sys
import os

def check_imports():
    """Check if all required packages are available"""
    print("Checking Python packages...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'torch.utils.data': 'PyTorch data utilities',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'timm': 'timm (PyTorch Image Models)',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)
    
    return len(missing) == 0, missing


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA support...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"  ✓ CUDA is available")
            print(f"    Device count: {device_count}")
            print(f"    Device name: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"  ✗ CUDA is NOT available (CPU mode will be used)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def check_dataset():
    """Check if dataset is available"""
    print("\nChecking dataset...")
    
    dataset_path = "../TreeFormer/datasets"
    splits = ["train_data", "valid_data", "test_data"]
    
    all_ok = True
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        images_path = os.path.join(split_path, "images")
        gt_path = os.path.join(split_path, "ground_truth")
        
        if os.path.isdir(images_path) and os.path.isdir(gt_path):
            num_images = len([f for f in os.listdir(images_path) if f.endswith('.jpg')])
            print(f"  ✓ {split}: {num_images} images")
        else:
            print(f"  ✗ {split}: Missing dataset structure")
            all_ok = False
    
    return all_ok


def check_model():
    """Check if model can be instantiated"""
    print("\nChecking model...")
    
    try:
        from network.swin_unet import SwinUNet
        import torch
        
        # Create model
        model = SwinUNet(
            img_size=256,
            patch_size=4,
            in_chans=3,
            num_classes=1
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ Model created successfully")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {y.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error with model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_loader():
    """Check if dataset loader works"""
    print("\nChecking dataset loader...")
    
    try:
        from datasets.kcl_london import KCLLondonSwinUNetDataset
        
        # Try to load dataset
        dataset = KCLLondonSwinUNetDataset(
            root="../TreeFormer/datasets",
            split="train_data",
            crop_size=256,
            random_flip=False,
            in_channels=3
        )
        
        print(f"  ✓ Dataset loader created successfully")
        print(f"    Dataset size: {len(dataset)}")
        
        # Try to get a sample
        sample = dataset[0]
        print(f"  ✓ Sample loaded successfully")
        print(f"    Image shape: {sample['image'].shape}")
        print(f"    Density shape: {sample['density'].shape}")
        print(f"    Count: {sample['count'].item():.1f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error with dataset loader: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("Swin-UNet Installation Verification")
    print("="*60)
    
    results = {
        'imports': check_imports(),
        'cuda': check_cuda(),
        'dataset': check_dataset(),
        'model': check_model(),
        'dataset_loader': check_dataset_loader()
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if isinstance(results['imports'], tuple):
        imports_ok, missing = results['imports']
        if not imports_ok:
            print(f"✗ Missing packages: {', '.join(missing)}")
            print("  Install with: pip install -r requirements.txt")
        else:
            print("✓ All imports OK")
    
    status = {
        'imports': 'OK' if isinstance(results['imports'], tuple) and results['imports'][0] else 'FAILED',
        'cuda': 'OK' if results['cuda'] else 'NOT AVAILABLE',
        'dataset': 'OK' if results['dataset'] else 'FAILED',
        'model': 'OK' if results['model'] else 'FAILED',
        'dataset_loader': 'OK' if results['dataset_loader'] else 'FAILED'
    }
    
    for check_name, status_val in status.items():
        symbol = '✓' if status_val in ['OK'] else '✗'
        print(f"{symbol} {check_name:20} {status_val}")
    
    print("\n" + "="*60)
    
    all_critical_ok = (
        (isinstance(results['imports'], tuple) and results['imports'][0]) and
        results['dataset'] and
        results['model'] and
        results['dataset_loader']
    )
    
    if all_critical_ok:
        print("✓ Installation verification PASSED!")
        print("  You can now run:")
        print("    python train_swin_unet_kcl.py --help")
        print("    python test_swin_unet_kcl.py --help")
        return 0
    else:
        print("✗ Installation verification FAILED!")
        print("  Please check the errors above and install missing packages")
        print("    pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
