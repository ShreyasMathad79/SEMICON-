"""
PyTorch Dataset for Paired Semiconductor Wafer Image Restoration.
Handles NoisyLR (Degraded input) and GT (Clean full-resolution target).
Supports dynamic path discovery, sub-patch cropping, and physical augmentations.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class SemiconDataset(Dataset):
    """
    Dataset loader for paired semiconductor wafer images (.npy float arrays).
    """
    def __init__(self, data_dir='.', is_train=True, val_split=0.1, crop_size=64, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.crop_size = crop_size

        # Flexible search paths for train/GT and train/NoisyLR
        candidate_gt_paths = [
            os.path.join(data_dir, 'train', 'GT'),
            os.path.join(data_dir, 'GT'),
            os.path.join(data_dir, 'dataset', 'train', 'GT'),
            os.path.join('train', 'GT'),
            os.path.join('dataset', 'train', 'GT')
        ]
        candidate_noisy_paths = [
            os.path.join(data_dir, 'train', 'NoisyLR'),
            os.path.join(data_dir, 'NoisyLR'),
            os.path.join(data_dir, 'dataset', 'train', 'NoisyLR'),
            os.path.join('train', 'NoisyLR'),
            os.path.join('dataset', 'train', 'NoisyLR')
        ]

        gt_dir = next((p for p in candidate_gt_paths if os.path.isdir(p)), None)
        noisy_dir = next((p for p in candidate_noisy_paths if os.path.isdir(p)), None)

        if gt_dir is None or noisy_dir is None:
            raise FileNotFoundError(f"Could not locate GT or NoisyLR dataset directories under '{data_dir}'")

        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
        noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.npy')))

        if len(gt_files) == 0 or len(noisy_files) == 0:
            raise ValueError(f"No .npy files found in GT: {gt_dir} or NoisyLR: {noisy_dir}")

        assert len(gt_files) == len(noisy_files), (
            f"Mismatch in files: {len(gt_files)} GT vs {len(noisy_files)} NoisyLR"
        )

        pairs = list(zip(noisy_files, gt_files))
        random.seed(seed)
        random.shuffle(pairs)

        val_size = max(1, int(len(pairs) * val_split))
        if is_train:
            self.pairs = pairs[val_size:]
        else:
            self.pairs = pairs[:val_size]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]

        noisy_img = np.load(noisy_path).astype(np.float32)
        gt_img = np.load(gt_path).astype(np.float32)

        # Augmentation and dynamic crop during training
        if self.is_train:
            h_noisy, w_noisy = noisy_img.shape
            cs = min(self.crop_size, h_noisy, w_noisy)

            # Random crop location on LR image
            top = random.randint(0, h_noisy - cs)
            left = random.randint(0, w_noisy - cs)

            # Corresponding GT crop (scale factor = 2)
            gt_top, gt_left = top * 2, left * 2
            gt_cs = cs * 2

            noisy_img = noisy_img[top:top+cs, left:left+cs]
            gt_img = gt_img[gt_top:gt_top+gt_cs, gt_left:gt_left+gt_cs]

            # Random Horizontal Flip
            if random.random() > 0.5:
                noisy_img = np.fliplr(noisy_img).copy()
                gt_img = np.fliplr(gt_img).copy()

            # Random Vertical Flip
            if random.random() > 0.5:
                noisy_img = np.flipud(noisy_img).copy()
                gt_img = np.flipud(gt_img).copy()

            # Random 90-degree rotations
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                noisy_img = np.rot90(noisy_img, rot_k).copy()
                gt_img = np.rot90(gt_img, rot_k).copy()

            # Random synthetic speckle perturbation (for out-of-distribution robustness)
            if random.random() > 0.7:
                noise_scale = random.uniform(0.01, 0.05)
                speckle = np.random.normal(0, noise_scale, noisy_img.shape).astype(np.float32)
                noisy_img = noisy_img + speckle

        # Add channel dimension: (H, W) -> (1, H, W)
        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0)

        return noisy_tensor, gt_tensor, os.path.basename(gt_path)
