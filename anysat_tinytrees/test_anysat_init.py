import sys
sys.path.append("/home/ashank/TreeCounting_Benchmark/AnySat")
import torch
from hubconf import AnySat

model = AnySat(model_size='base', flash_attn=False)

data = {
    "naip": torch.randn(2, 4, 64, 64)
}
features = model(data, patch_size=10, output='dense', output_modality='naip')
# features is [2, 8, 8, 1536] -> permute to [2, 1536, 8, 8]
features = features.permute(0, 3, 1, 2)
print("Features permuted:", features.shape)

import torch.nn as nn
# Upsample by 8
upsample = nn.ConvTranspose2d(1536, 64, kernel_size=8, stride=8)
conv = nn.Conv2d(64, 1, kernel_size=1)
out = conv(upsample(features))
print("Final out shape:", out.shape)
