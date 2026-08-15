"""
Enhanced Training & Dataset Generation Pipeline for Semiconductor Restoration.
Generates comprehensive synthetic wafer structures and trains SemiconRestorationNet
with the balanced Composite Loss for high PSNR (>33 dB) and low LPIPS (<0.15).
"""

import os
import sys
import glob
import random
import time
import json
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from models.restoration_model import SemiconRestorationNet
from utils.losses import CompositeRestorationLoss
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_edge_gradient_correlation


def generate_synthetic_wafer_dataset(base_dir='train', num_samples=120):
    """
    Generates realistic semiconductor wafer inspection pattern pairs:
    - Line-space periodic gratings at various angles & pitches
    - Contact hole arrays (hexagonal & grid)
    - FinFET gate array channels
    - Crossbar interconnects and logic routing
    """
    gt_dir = os.path.join(base_dir, 'GT')
    noisy_dir = os.path.join(base_dir, 'NoisyLR')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # Check if already populated
    existing = len(glob.glob(os.path.join(gt_dir, '*.npy')))
    if existing >= num_samples:
        print(f" [Dataset] Found {existing} existing paired wafer samples in '{base_dir}'.")
        return

    print(f" [Dataset] Synthesizing {num_samples} realistic semiconductor wafer patterns...")
    np.random.seed(42)

    for i in range(num_samples):
        # 256x256 Ground Truth Synthesis
        pattern_type = i % 5
        gt = np.zeros((256, 256), dtype=np.float32) + 0.15

        if pattern_type == 0:
            # 1. 1D Periodic Line Gratings (DRAM / Pitch Grating)
            pitch = random.choice([8, 12, 16, 24, 32])
            y, x = np.mgrid[0:256, 0:256]
            wave = 0.5 + 0.45 * np.sin(2 * np.pi * x / pitch)
            gt = np.where(wave > 0.5, 0.85, 0.15).astype(np.float32)

        elif pattern_type == 1:
            # 2. Contact Hole Grid / Hexagonal Array
            pitch = random.choice([32, 40, 48, 56])
            radius = random.choice([8, 10, 12, 14])
            offset_x = random.randint(0, pitch // 2)
            offset_y = random.randint(0, pitch // 2)
            for r in range(offset_y, 256, pitch):
                for c in range(offset_x, 256, pitch):
                    cv2.circle(gt, (c, r), radius, 0.85, -1)

        elif pattern_type == 2:
            # 3. FinFET Array (Interleaved Gates and Source/Drain Fins)
            pitch = random.choice([16, 20, 24])
            fin_w = random.choice([4, 6, 8])
            for c in range(8, 256, pitch):
                gt[:, c:c+fin_w] = 0.85
            # Add transverse gate blocks
            for r in range(16, 256, 32):
                gt[r:r+6, :] = np.clip(gt[r:r+6, :] + 0.35, 0.15, 0.85)

        elif pattern_type == 3:
            # 4. Crossbar Interconnect Grid & Logic Routing
            step_x = random.choice([24, 32, 48])
            step_y = random.choice([24, 32, 48])
            for c in range(12, 256, step_x):
                gt[:, c:c+6] = 0.80
            for r in range(12, 256, step_y):
                gt[r:r+6, :] = 0.80
            for r in range(12, 256, step_y):
                for c in range(12, 256, step_x):
                    cv2.circle(gt, (c+3, r+3), 7, 0.90, -1)

        else:
            # 5. Diagonal Grating & Angled Lithography Slits
            pitch = random.choice([16, 24, 32])
            y, x = np.mgrid[0:256, 0:256]
            wave = 0.5 + 0.45 * np.sin(2 * np.pi * (x + y) / (pitch * 1.414))
            gt = np.where(wave > 0.5, 0.85, 0.15).astype(np.float32)

        # Smooth edges slightly to simulate realistic optical photolithography profile
        gt_blur = cv2.GaussianBlur(gt, (3, 3), 0.5)

        # Physical Degradation Model:
        # Step 1: Optical 2x downsampling (256x256 -> 128x128)
        lr = cv2.resize(gt_blur, (128, 128), interpolation=cv2.INTER_AREA)

        # Step 2: Diffraction limit / stage vibration Gaussian blur
        blur_sigma = random.uniform(0.6, 1.0)
        lr = cv2.GaussianBlur(lr, (3, 3), blur_sigma)

        # Step 3: Coherent speckle noise + sensor shot noise
        speckle_noise = np.random.normal(0, random.uniform(0.06, 0.10), lr.shape).astype(np.float32)
        lr_degraded = lr + speckle_noise

        # Step 4: Random out-of-range speckle values
        lr_degraded = np.clip(lr_degraded, 0.0, 1.35).astype(np.float32)

        # Save files
        file_name = f"wafer_{i:04d}.npy"
        np.save(os.path.join(gt_dir, file_name), gt_blur.astype(np.float32))
        np.save(os.path.join(noisy_dir, file_name), lr_degraded.astype(np.float32))

    print(f" [Dataset] Successfully generated {num_samples} wafer pattern pairs.")


class WaferDataset(Dataset):
    def __init__(self, data_dir='train', is_train=True, val_split=0.15, crop_size=64, seed=42):
        gt_files = sorted(glob.glob(os.path.join(data_dir, 'GT', '*.npy')))
        noisy_files = sorted(glob.glob(os.path.join(data_dir, 'NoisyLR', '*.npy')))
        pairs = list(zip(noisy_files, gt_files))

        random.seed(seed)
        random.shuffle(pairs)

        val_size = max(4, int(len(pairs) * val_split))
        if is_train:
            self.pairs = pairs[val_size:]
        else:
            self.pairs = pairs[:val_size]

        self.is_train = is_train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]
        noisy_img = np.load(noisy_path).astype(np.float32)
        gt_img = np.load(gt_path).astype(np.float32)

        if self.is_train:
            h, w = noisy_img.shape
            cs = min(self.crop_size, h, w)
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
            gt_top, gt_left = top * 2, left * 2
            gt_cs = cs * 2

            noisy_img = noisy_img[top:top+cs, left:left+cs]
            gt_img = gt_img[gt_top:gt_top+gt_cs, gt_left:gt_left+gt_cs]

            # Augmentation
            if random.random() > 0.5:
                noisy_img = np.fliplr(noisy_img).copy()
                gt_img = np.fliplr(gt_img).copy()
            if random.random() > 0.5:
                noisy_img = np.flipud(noisy_img).copy()
                gt_img = np.flipud(gt_img).copy()
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                noisy_img = np.rot90(noisy_img, rot_k).copy()
                gt_img = np.rot90(gt_img, rot_k).copy()

        return torch.from_numpy(noisy_img).unsqueeze(0), torch.from_numpy(gt_img).unsqueeze(0)


def train_enhanced(epochs=12, batch_size=16, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 65)
    print(" [TRAIN] High-Performance Wafer Restoration Training")
    print(f" Device    : {device.upper()}")
    print(f" Epochs    : {epochs}")
    print(f" Batch Size: {batch_size}")
    print(f" LR        : {lr}")
    print("=" * 65)

    generate_synthetic_wafer_dataset('train', num_samples=120)

    train_ds = WaferDataset('train', is_train=True, crop_size=64)
    val_ds = WaferDataset('train', is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = SemiconRestorationNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = CompositeRestorationLoss(alpha=0.25, beta=0.15, gamma=0.05, delta=0.10)

    os.makedirs('weights', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss, _, _, _ = criterion(output, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_ds)

        # Validation (Fast Evaluation)
        model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy = noisy.to(device)
                out = model(noisy).cpu().numpy().squeeze(1)
                gts = gt.numpy().squeeze(1)
                for pred_img, gt_img in zip(out, gts):
                    p = calculate_psnr(pred_img, gt_img)
                    s = calculate_ssim(pred_img, gt_img)
                    l = calculate_lpips(pred_img, gt_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    lpips_list.append(l)

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))
        val_lpips = float(np.mean(lpips_list))

        print(f" Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - Loss: {avg_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_lpips = val_lpips
            best_path = os.path.join('weights', 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  --> Best Model Checkpoint Saved -> {best_path} (PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}, LPIPS: {best_lpips:.4f})")

    # Save final model
    final_path = os.path.join('weights', 'semicon_restoration_model.pth')
    torch.save(model.state_dict(), final_path)

    print("=" * 65)
    print(f" [DONE] Training Complete!")
    print(f"  Best PSNR : {best_psnr:.2f} dB")
    print(f"  Best SSIM : {best_ssim:.4f}")
    print(f"  Best LPIPS: {best_lpips:.4f}")
    print("=" * 65)


if __name__ == '__main__':
    train_enhanced(epochs=12, batch_size=16, lr=1e-3)
