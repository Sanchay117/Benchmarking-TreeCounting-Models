"""
Final checklist and summary of Swin-UNet implementation
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.isfile(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {description:<40} ({size_kb:>7.1f} KB)")
        return True
    else:
        print(f"  ✗ {description:<40} MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"  ✓ {description:<40} (directory)")
        return True
    else:
        print(f"  ✗ {description:<40} MISSING")
        return False

def main():
    print("="*70)
    print("Swin-UNet Implementation Checklist")
    print("="*70)
    
    all_ok = True
    
    # Check core files
    print("\n📋 Core Implementation Files:")
    core_files = {
        'network/swin_unet.py': 'Model architecture',
        'datasets/kcl_london.py': 'Dataset loader',
        'train_swin_unet_kcl.py': 'Training script',
        'test_swin_unet_kcl.py': 'Evaluation script',
        '__init__.py': 'Package init',
        'network/__init__.py': 'Network package init',
        'datasets/__init__.py': 'Datasets package init',
    }
    
    for filepath, description in core_files.items():
        if not check_file_exists(filepath, description):
            all_ok = False
    
    # Check documentation
    print("\n📖 Documentation Files:")
    doc_files = {
        'README.md': 'Main documentation',
        'QUICKSTART.md': 'Quick start guide',
        'IMPLEMENTATION_SUMMARY.md': 'Implementation summary',
        'INTEGRATION.md': 'Integration guide',
        'requirements.txt': 'Python dependencies',
    }
    
    for filepath, description in doc_files.items():
        if not check_file_exists(filepath, description):
            all_ok = False
    
    # Check utility files
    print("\n🛠️  Utility Files:")
    util_files = {
        'verify_installation.py': 'Installation verification',
        'run_configs.sh': 'Configuration templates',
        '.gitignore': 'Git ignore file',
    }
    
    for filepath, description in util_files.items():
        if not check_file_exists(filepath, description):
            all_ok = False
    
    # Check directories
    print("\n📁 Required Directories:")
    directories = {
        'network': 'Network package',
        'datasets': 'Datasets package',
        'ckpts': 'Checkpoints (auto-created)',
        'eval': 'Evaluation results (auto-created)',
        'predictions': 'Predictions (auto-created)',
        'pretrained_ckpt': 'Pre-trained weights (empty)',
    }
    
    for dirpath, description in directories.items():
        if not check_directory_exists(dirpath, description):
            # Some directories are auto-created, so not critical
            if dirpath not in ['ckpts', 'eval', 'predictions']:
                all_ok = False
    
    # Calculate total size
    print("\n📊 Statistics:")
    total_size = 0
    for root, dirs, files in os.walk('.'):
        for file in files:
            if not file.startswith('.'):
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)
    
    print(f"  Total code size: {total_size/1024:.1f} KB")
    
    # Count lines of code
    print("\n📝 Code Statistics:")
    total_lines = 0
    
    code_files = {
        'network/swin_unet.py': 'Model',
        'datasets/kcl_london.py': 'Dataset loader',
        'train_swin_unet_kcl.py': 'Training',
        'test_swin_unet_kcl.py': 'Evaluation',
    }
    
    for filepath, description in code_files.items():
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"  {description:<20} {lines:>5} lines")
    
    print(f"  Total: {total_lines:>5} lines")
    
    # Next steps
    print("\n" + "="*70)
    print("✅ NEXT STEPS:")
    print("="*70)
    print("""
1. Verify Installation:
   python verify_installation.py

2. Review Documentation:
   cat README.md
   cat QUICKSTART.md

3. Quick Training Test:
   bash run_configs.sh quick

4. Full Training:
   bash run_configs.sh standard

5. Evaluate Results:
   bash run_configs.sh eval <checkpoint_path>

6. For Integration Details:
   cat INTEGRATION.md
""")
    
    print("="*70)
    
    if all_ok:
        print("✅ All core files are present!")
        return 0
    else:
        print("⚠️  Some files are missing. Please check above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
