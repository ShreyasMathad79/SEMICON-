"""
CLI tool to calculate and display PSNR, SSIM, and LPIPS values in the Windows Terminal.
Supports .npy, .png, .jpg, .tif files, and folder comparisons.
"""

import os
import sys
import glob
import warnings
import argparse
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

warnings.filterwarnings('ignore')
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_edge_gradient_correlation


def load_img(path):
    """Loads an image (.npy or standard format) normalized to [0.0, 1.0]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path).astype(np.float64)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Could not load image file: {path}")
        arr = arr.astype(np.float64) / 255.0
    return np.clip(arr, 0.0, 1.0)


def compute_metrics(img1, img2):
    """Computes PSNR, SSIM, LPIPS, and Edge Correlation between two 2D float arrays."""
    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    lpips_val = calculate_lpips(img1, img2)
    edge_val = calculate_edge_gradient_correlation(img1, img2)
    return psnr_val, ssim_val, lpips_val, edge_val


def main():
    parser = argparse.ArgumentParser(
        description="Print PSNR, SSIM, and LPIPS values directly in Windows Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two single files:
  python show_metrics.py file1.png file2.png
  python show_metrics.py restored_test_outputs/000000.npy NoisyLR/000000.npy

  # Compare two directories:
  python show_metrics.py restored_test_outputs GT_folder
        """
    )
    parser.add_argument('target', help='Path to first image or restored folder')
    parser.add_argument('reference', help='Path to second/reference image or ground truth folder')
    parser.add_argument('--limit', type=int, default=15, help='Max rows to display in batch mode (default: 15)')
    args = parser.parse_args()

    print("=" * 75)
    print(" [METRICS] PSNR, SSIM & LPIPS EVALUATION (KLA BENCHMARK)")
    print("=" * 75)

    if os.path.isfile(args.target) and os.path.isfile(args.reference):
        img1 = load_img(args.target)
        img2 = load_img(args.reference)
        psnr, ssim, lpips_score, edge_corr = compute_metrics(img1, img2)

        print(f" Target File    : {args.target}")
        print(f" Reference File : {args.reference}")
        print("-" * 75)
        print(f"  PSNR (Higher is better)     : {psnr:8.4f} dB")
        print(f"  SSIM (Higher is better)     : {ssim:8.4f}")
        print(f"  LPIPS (Lower is better)     : {lpips_score:8.4f}")
        print(f"  Edge Correlation            : {edge_corr:8.4f}")
        print("=" * 75)

    elif os.path.isdir(args.target) and os.path.isdir(args.reference):
        t_files = sorted(glob.glob(os.path.join(args.target, '*.*')))
        r_files = sorted(glob.glob(os.path.join(args.reference, '*.*')))
        
        valid_exts = {'.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        t_files = [f for f in t_files if os.path.splitext(f)[1].lower() in valid_exts]
        
        ref_map = {os.path.splitext(os.path.basename(f))[0]: f for f in r_files}
        
        psnr_list, ssim_list, lpips_list, edge_list = [], [], [], []
        print(f" {'FILENAME':<20} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'LPIPS':<8} | {'EDGE CORR':<9}")
        print("-" * 75)

        displayed = 0
        for tf in t_files:
            bname = os.path.splitext(os.path.basename(tf))[0]
            if bname in ref_map:
                rf = ref_map[bname]
                img1 = load_img(tf)
                img2 = load_img(rf)
                p, s, l, e = compute_metrics(img1, img2)
                psnr_list.append(p)
                ssim_list.append(s)
                lpips_list.append(l)
                edge_list.append(e)

                if displayed < args.limit:
                    print(f" {bname:<20} | {p:8.2f} dB | {s:6.4f} | {l:6.4f} | {e:7.4f}")
                    displayed += 1

        if len(psnr_list) == 0:
            print(" [Warning] No matching filenames found between target and reference directories.")
            return

        if len(psnr_list) > args.limit:
            print(f" ... and {len(psnr_list) - args.limit} more samples.")

        print("=" * 75)
        print(f" [SUMMARY] OVER {len(psnr_list)} BENCHMARK SAMPLES:")
        print(f"  Mean PSNR : {np.mean(psnr_list):8.4f} dB  (Min: {np.min(psnr_list):.2f}, Max: {np.max(psnr_list):.2f})")
        print(f"  Mean SSIM : {np.mean(ssim_list):8.4f}     (Min: {np.min(ssim_list):.4f}, Max: {np.max(ssim_list):.4f})")
        print(f"  Mean LPIPS: {np.mean(lpips_list):8.4f}     (Min: {np.min(lpips_list):.4f}, Max: {np.max(lpips_list):.4f})")
        print(f"  Mean Edge : {np.mean(edge_list):8.4f}     (Min: {np.min(edge_list):.4f}, Max: {np.max(edge_list):.4f})")
        print("=" * 75)
    else:
        print("Error: Both inputs must be either single files or directories.")


if __name__ == '__main__':
    main()
