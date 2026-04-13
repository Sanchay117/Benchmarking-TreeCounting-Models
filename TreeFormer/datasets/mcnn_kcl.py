import os
import random
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset


class KCLLondonMCNNDataset(Dataset):
    """
    Dataset for MCNN fine-tuning on the KCL London dataset layout used in TreeFormer.

    Expected folder structure:
      <root>/<split>/images/*.jpg
      <root>/<split>/ground_truth/GT_<name>.mat
      <root>/<split>/ground_truth/<name>_densitymap.npy
    """

    def __init__(self, root, split, crop_size=None, random_flip=False):
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.random_flip = random_flip

        self.image_dir = os.path.join(root, split, "images")
        self.gt_dir = os.path.join(root, split, "ground_truth")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"Ground-truth directory not found: {self.gt_dir}")

        self.im_list = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
        if len(self.im_list) == 0:
            raise RuntimeError(f"No .jpg files found in {self.image_dir}")

    def __len__(self):
        return len(self.im_list)

    def _load_density(self, name):
        density_path = os.path.join(self.gt_dir, f"{name}_densitymap.npy")
        if not os.path.isfile(density_path):
            raise FileNotFoundError(f"Density map not found: {density_path}")
        return np.load(density_path).astype(np.float32)

    def _load_count(self, name):
        mat_path = os.path.join(self.gt_dir, f"GT_{name}.mat")
        if not os.path.isfile(mat_path):
            return None
        keypoints = sio.loadmat(mat_path)["image_info"][0][0][0][0][0]
        return float(len(keypoints))

    @staticmethod
    def _resize_pair(image, density, new_w, new_h):
        old_h, old_w = density.shape
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        density = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        density = density * ((old_h * old_w) / float(new_h * new_w))
        return image, density

    def _prepare(self, image, density):
        h, w = image.shape

        if self.crop_size is not None:
            min_side = min(h, w)
            if min_side < self.crop_size:
                scale = float(self.crop_size) / float(min_side)
                new_h = int(round(h * scale))
                new_w = int(round(w * scale))
                image, density = self._resize_pair(image, density, new_w, new_h)
                h, w = image.shape

            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            image = image[top : top + self.crop_size, left : left + self.crop_size]
            density = density[top : top + self.crop_size, left : left + self.crop_size]

            if self.random_flip and random.random() > 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                density = np.ascontiguousarray(np.fliplr(density))
        else:
            # Keep dimensions divisible by 4 for the two pooling stages in MCNN.
            new_h = max((h // 4) * 4, 4)
            new_w = max((w // 4) * 4, 4)
            if (new_h != h) or (new_w != w):
                image, density = self._resize_pair(image, density, new_w, new_h)

        return image, density

    def __getitem__(self, idx):
        img_path = self.im_list[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        density = self._load_density(name)
        image, density = self._prepare(image, density)

        if density.ndim != 2:
            raise ValueError(f"Density map must be 2D for image {name}")

        count = self._load_count(name)
        if count is None:
            count = float(np.sum(density))

        image_t = torch.from_numpy(image).unsqueeze(0)
        density_t = torch.from_numpy(density).unsqueeze(0)

        return {
            "image": image_t,
            "density": density_t,
            "count": torch.tensor(count, dtype=torch.float32),
            "name": name,
        }
