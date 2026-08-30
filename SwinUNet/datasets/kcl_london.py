"""
Dataset loader for Swin-UNet on KCL London tree counting dataset.

Expected folder structure:
  <root>/<split>/images/*.jpg
  <root>/<split>/ground_truth/GT_<name>.mat
  <root>/<split>/ground_truth/<name>_densitymap.npy
"""

import os
import random
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset


class KCLLondonSwinUNetDataset(Dataset):
    """
    Dataset for Swin-UNet fine-tuning on the KCL London dataset layout used in TreeFormer.
    
    Expected folder structure:
      <root>/<split>/images/*.jpg
      <root>/<split>/ground_truth/GT_<name>.mat
      <root>/<split>/ground_truth/<name>_densitymap.npy
    """

    def __init__(self, root, split, crop_size=256, resize_to=None, random_flip=False, is_validation=False, in_channels=3):
        """
        Args:
            root: Root directory of dataset
            split: 'train_data', 'valid_data', 'test_data'
            crop_size: Size to crop images to (None = no cropping)
            resize_to: Resize full images to this size when crop_size is None
            random_flip: Whether to randomly flip images during training
            is_validation: If True, use center crop instead of random crop for consistency
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.resize_to = resize_to
        self.random_flip = random_flip
        self.is_validation = is_validation
        self.in_channels = in_channels

        self.image_dir = os.path.join(root, split, "images")
        self.gt_dir = os.path.join(root, split, "ground_truth")
        
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"Ground-truth directory not found: {self.gt_dir}")

        self.im_list = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
        if len(self.im_list) == 0:
            raise RuntimeError(f"No .jpg files found in {self.image_dir}")
        
        print(f"Loaded {len(self.im_list)} images from {self.image_dir}")

    def __len__(self):
        return len(self.im_list)

    def _load_density(self, name):
        """Load pre-computed density map"""
        density_path = os.path.join(self.gt_dir, f"{name}_densitymap.npy")
        if not os.path.isfile(density_path):
            raise FileNotFoundError(f"Density map not found: {density_path}")
        
        density = np.load(density_path).astype(np.float32)
        
        # ✅ FIX: Clamp any negative values (typically from data preparation)
        density = np.clip(density, a_min=0, a_max=None)
        
        return density

    def _load_count(self, name):
        """Load ground truth tree count from .mat file"""
        mat_path = os.path.join(self.gt_dir, f"GT_{name}.mat")
        if not os.path.isfile(mat_path):
            return None
        try:
            keypoints = sio.loadmat(mat_path)["image_info"][0][0][0][0][0]
            return float(len(keypoints))
        except:
            return None

    @staticmethod
    def _resize_pair(image, density, new_w, new_h):
        """Resize image and density map while preserving total count"""
        old_h, old_w = density.shape
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        density = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Preserve total count during resizing
        scale_factor = (old_h * old_w) / float(new_h * new_w)
        density = density * scale_factor
        
        # ✅ FIX: Clamp small negative values (tiny floating-point errors from interpolation)
        # These are typically < 1e-3 and don't affect training
        density = np.clip(density, a_min=0, a_max=None)
        
        return image, density

    def _prepare(self, image, density):
        """Prepare image and density for training/evaluation"""
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]

        if self.crop_size is not None:
            # For training/validation with cropping
            min_side = min(h, w)
            
            # Resize if needed to fit crop size
            if min_side < self.crop_size:
                scale = float(self.crop_size) / float(min_side)
                new_h = int(round(h * scale))
                new_w = int(round(w * scale))
                image, density = self._resize_pair(image, density, new_w, new_h)
                h, w = image.shape[:2] if image.ndim > 2 else image.shape

            # Use center crop for validation, random crop for training
            if self.is_validation:
                # Center crop for consistent validation
                top = max(0, (h - self.crop_size) // 2)
                left = max(0, (w - self.crop_size) // 2)
            else:
                # Random crop for training augmentation
                top = random.randint(0, max(0, h - self.crop_size))
                left = random.randint(0, max(0, w - self.crop_size))
            
            if image.ndim == 2:
                image = image[top : top + self.crop_size, left : left + self.crop_size]
            else:
                image = image[top : top + self.crop_size, left : left + self.crop_size, :]
            
            density = density[top : top + self.crop_size, left : left + self.crop_size]

            if self.random_flip and random.random() > 0.5:
                if image.ndim == 2:
                    image = np.ascontiguousarray(np.fliplr(image))
                else:
                    image = np.ascontiguousarray(np.fliplr(image))
                density = np.ascontiguousarray(np.fliplr(density))
        
        elif self.resize_to is not None:
            # For testing, resize to specified size (e.g., 256x256 for model input)
            if (h != self.resize_to) or (w != self.resize_to):
                image, density = self._resize_pair(image, density, self.resize_to, self.resize_to)

        return image, density

    def __getitem__(self, idx):
        img_path = self.im_list[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        # Load image
        img_pil = Image.open(img_path)
        
        # Convert to appropriate format
        if self.in_channels == 3:
            image = np.array(img_pil.convert('RGB'), dtype=np.float32)
        else:
            image = np.array(img_pil.convert('L'), dtype=np.float32)
        
        # Load density map
        density = self._load_density(name)
        
        # Prepare image and density
        image, density = self._prepare(image, density)

        if density.ndim != 2:
            raise ValueError(f"Density map must be 2D for image {name}")

        # Load count from ground truth
        count = self._load_count(name)
        if count is None:
            count = float(np.sum(density))
        
        # ✅ FIX: Ensure count is non-negative and reasonable
        count = max(0.0, float(count))  # Ensure non-negative
        
        # ✅ Safety check: Verify density and count consistency
        density_sum = float(np.sum(density))
        if density_sum > 0 and count == 0:
            print(f"WARNING: Image {name} has density sum {density_sum:.2f} but count is 0")
            count = density_sum

        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Convert to tensors
        if self.in_channels == 3:
            image_t = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image_t = torch.from_numpy(image).unsqueeze(0)
        
        density_t = torch.from_numpy(density).unsqueeze(0)

        return {
            "image": image_t.float(),
            "density": density_t.float(),
            "count": torch.tensor(count, dtype=torch.float32),
            "name": name,
        }


if __name__ == "__main__":
    # Test the dataset
    dataset = KCLLondonSwinUNetDataset(
        root="../TreeFormer/datasets",
        split="train_data",
        crop_size=256,
        random_flip=True,
        in_channels=3
    )
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Density shape: {sample['density'].shape}")
    print(f"Count: {sample['count'].item()}")
    print(f"Name: {sample['name']}")
