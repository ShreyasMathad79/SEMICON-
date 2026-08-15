"""
Complete Training Pipeline for Semiconductor Image Restoration.
Trains SemiconRestorationNet with Composite Multi-Domain Loss (Charbonnier + SSIM + Sobel Edge + 2D FFT).
Includes Cosine Annealing scheduler, validation tracking, metric exports, and checkpoint saving.
"""

import os
import time
import json
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.restoration_model import SemiconRestorationNet
from utils.dataset import SemiconDataset
from utils.losses import CompositeRestorationLoss
from utils.metrics import calculate_psnr, calculate_ssim, calculate_edge_gradient_correlation


def train_model(data_dir='.', epochs=12, batch_size=32, lr=8e-4, crop_size=64, device=None):
    # Set high-performance thread allocation
    cpus = os.cpu_count() or 4
    torch.set_num_threads(cpus)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 65)
    print(f" [TRAIN] Training SemiconRestorationNet")
    print(f" Compute Device : {device.upper()} ({cpus} CPU threads)")
    print(f" Epochs         : {epochs}")
    print(f" Batch Size     : {batch_size}")
    print(f" Learning Rate  : {lr}")
    print(f" Crop Patch Size: {crop_size}x{crop_size}")
    print("=" * 65)

    os.makedirs('weights', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Initialize Datasets
    train_dataset = SemiconDataset(data_dir=data_dir, is_train=True, val_split=0.1, crop_size=crop_size)
    val_dataset = SemiconDataset(data_dir=data_dir, is_train=False, val_split=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=min(16, len(val_dataset)), shuffle=False, num_workers=0)

    print(f" Dataset Ready: {len(train_dataset)} Train samples | {len(val_dataset)} Validation samples.")

    # Initialize Model, Optimizer, Loss, and Scheduler
    model = SemiconRestorationNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CompositeRestorationLoss(alpha=0.20, beta=0.10, gamma=0.05)

    best_val_psnr = 0.0
    best_val_ssim = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_charb = 0.0
        running_ssim_l = 0.0
        running_edge_l = 0.0

        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}")

        for noisy, gt, _ in pbar:
            noisy = noisy.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            output = model(noisy)

            loss, l_charb, l_ssim, l_edge = criterion(output, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            b_size = noisy.size(0)
            running_loss += loss.item() * b_size
            running_charb += l_charb.item() * b_size
            running_ssim_l += l_ssim.item() * b_size
            running_edge_l += l_edge.item() * b_size

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Charb': f"{l_charb.item():.4f}",
                'SSIM_L': f"{l_ssim.item():.4f}"
            })

        scheduler.step()
        epoch_time = time.time() - start_time

        train_loss = running_loss / len(train_dataset)
        train_charb = running_charb / len(train_dataset)

        # Validation Step
        model.eval()
        val_psnr_list = []
        val_ssim_list = []
        val_edge_corr_list = []

        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy = noisy.to(device)
                outputs = model(noisy).cpu().numpy().squeeze(1)
                gts = gt.numpy().squeeze(1)

                for pred_img, gt_img in zip(outputs, gts):
                    p_score = calculate_psnr(pred_img, gt_img)
                    s_score = calculate_ssim(pred_img, gt_img)
                    e_score = calculate_edge_gradient_correlation(pred_img, gt_img)

                    val_psnr_list.append(p_score)
                    val_ssim_list.append(s_score)
                    val_edge_corr_list.append(e_score)

        mean_val_psnr = float(np.mean(val_psnr_list))
        mean_val_ssim = float(np.mean(val_ssim_list))
        mean_val_edge = float(np.mean(val_edge_corr_list))

        print(f" Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - Train Loss: {train_loss:.4f} | Val PSNR: {mean_val_psnr:.2f} dB | Val SSIM: {mean_val_ssim:.4f} | Edge Corr: {mean_val_edge:.4f}")

        # Save Metrics History
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_charb': train_charb,
            'val_psnr': mean_val_psnr,
            'val_ssim': mean_val_ssim,
            'val_edge_corr': mean_val_edge,
            'lr': scheduler.get_last_lr()[0],
            'time_sec': epoch_time
        }
        history.append(epoch_record)

        # Check and Save Best Model Checkpoint
        if mean_val_psnr > best_val_psnr:
            best_val_psnr = mean_val_psnr
            best_val_ssim = mean_val_ssim
            best_path = os.path.join('weights', 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  --> Best Model Checkpoint Saved -> {best_path} (PSNR: {best_val_psnr:.2f} dB, SSIM: {best_val_ssim:.4f})")

    # Save Final Model Checkpoint
    final_path = os.path.join('weights', 'semicon_restoration_model.pth')
    torch.save(model.state_dict(), final_path)

    # Save JSON and CSV Training Logs
    with open('logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    with open('logs/training_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("=" * 65)
    print(f" Training Complete!")
    print(f" Best Val PSNR: {best_val_psnr:.2f} dB | Best Val SSIM: {best_val_ssim:.4f}")
    print(f" Checkpoints  : weights/best_model.pth, weights/semicon_restoration_model.pth")
    print(f" Training Logs: logs/training_history.json, logs/training_history.csv")
    print("=" * 65)

    return best_val_psnr, best_val_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SemiconRestorationNet for Semiconductor Image Restoration')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=12, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=8e-4, help='Initial learning rate')
    parser.add_argument('--crop_size', type=int, default=64, help='Random LR crop size')
    parser.add_argument('--device', type=str, default=None, help='Device to train on (cuda / cpu)')

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        crop_size=args.crop_size,
        device=args.device
    )
