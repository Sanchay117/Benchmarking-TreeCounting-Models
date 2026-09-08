import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "AnySat"))
from hubconf import AnySat
import torch

if 'utils' in sys.modules:
    del sys.modules['utils']
    
model = AnySat('base')
data = {"naip": torch.randn(2, 4, 64, 64)}
model(data, patch_size=10, output='dense', output_modality='naip')
print("Forward pass successful!")
