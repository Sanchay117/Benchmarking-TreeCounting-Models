"""
Swin-UNet: Tree Counting Benchmark Implementation
Independent implementation for KCL London dataset benchmarking
"""

__version__ = "1.0.0"
__author__ = "TreeCountingBenchmark"
__description__ = "Swin-UNet for tree counting on KCL London dataset"

import sys
import os

# Add module paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network.swin_unet import SwinUNet
from datasets.kcl_london import KCLLondonSwinUNetDataset

__all__ = [
    'SwinUNet',
    'KCLLondonSwinUNetDataset',
    '__version__',
    '__author__',
    '__description__'
]
