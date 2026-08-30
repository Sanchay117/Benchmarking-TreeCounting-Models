"""
Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation
Official implementation from: https://github.com/HuCaoFighting/Swin-Unet

Wrapper for tree counting task on KCL London dataset.
"""

import copy
import logging
import math
import torch
import torch.nn as nn

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUNet(nn.Module):
    """
    Wrapper around SwinTransformerSys for tree counting.
    Converts grayscale to RGB if needed for the model.
    
    This is the official Swin-UNet implementation adapted for tree counting.
    """
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 ape=False, patch_norm=True, use_checkpoint=False, **kwargs):
        """
        Initialize Swin-UNet model for tree counting.
        
        Args:
            img_size (int): Input image size (default: 224)
            patch_size (int): Patch size (default: 4)
            in_chans (int): Number of input channels (default: 3)
            num_classes (int): Number of output classes (default: 1 for density map)
            embed_dim (int): Embedding dimension (default: 96)
            depths (tuple): Depths of each layer (default: (2, 2, 2, 2))
            num_heads (tuple): Number of heads in each layer (default: (3, 6, 12, 24))
            window_size (int): Window size (default: 7)
            mlp_ratio (float): MLP ratio (default: 4.)
            qkv_bias (bool): Use QKV bias (default: True)
            qk_scale (float): QK scale (default: None)
            drop_rate (float): Dropout rate (default: 0.)
            attn_drop_rate (float): Attention dropout rate (default: 0.)
            drop_path_rate (float): Drop path rate (default: 0.1)
            ape (bool): Use absolute position embedding (default: False)
            patch_norm (bool): Use patch norm (default: True)
            use_checkpoint (bool): Use checkpointing (default: False)
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.num_classes = num_classes
        
        # Create the official Swin-UNet model
        self.model = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,  # Always 3 for official model
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C can be 1 or 3
        
        Returns:
            Output density map of shape (B, num_classes, H, W)
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        logits = self.model(x)
        
        # ✅ CRITICAL FIX: Apply ReLU activation to ensure non-negative density predictions
        # This prevents the model from predicting negative tree counts
        logits = torch.nn.functional.relu(logits)
        
        return logits
    
    def load_from(self, pretrained_path):
        """Load pretrained weights from checkpoint."""
        if pretrained_path is None:
            print("No pretrained path provided")
            return
        
        print(f"Loading pretrained weights from: {pretrained_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            if "model" not in pretrained_dict:
                # Handle old format without 'model' key
                print("Loading pretrained model (old format)...")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print(f"Removing key: {k}")
                        del pretrained_dict[k]
                self.model.load_state_dict(pretrained_dict, strict=False)
            else:
                # Handle new format with 'model' key
                print("Loading pretrained Swin Transformer encoder...")
                pretrained_dict = pretrained_dict['model']
                
                model_dict = self.model.state_dict()
                full_dict = copy.deepcopy(pretrained_dict)
                
                # Map encoder weights to decoder
                for k, v in pretrained_dict.items():
                    if "layers." in k:
                        current_layer_num = 3 - int(k[7:8])
                        current_k = "layers_up." + str(current_layer_num) + k[8:]
                        full_dict.update({current_k: v})
                
                # Remove incompatible layers
                for k in list(full_dict.keys()):
                    if k in model_dict:
                        if full_dict[k].shape != model_dict[k].shape:
                            print(f"Removing incompatible: {k} (pretrain: {full_dict[k].shape}, model: {model_dict[k].shape})")
                            del full_dict[k]
                
                self.model.load_state_dict(full_dict, strict=False)
            
            print("Successfully loaded pretrained weights!")
        
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    model = SwinUNet(img_size=224, patch_size=4, in_chans=3, num_classes=1)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test with grayscale
    x_gray = torch.randn(1, 1, 224, 224)
    y_gray = model(x_gray)
    print(f"Grayscale input shape: {x_gray.shape}")
    print(f"Grayscale output shape: {y_gray.shape}")
