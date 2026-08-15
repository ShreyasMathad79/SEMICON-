"""
KLA Semiconductor Image Restoration Evaluation Script.
Standalone inference script benchmarked by KLA evaluation team on NVIDIA H100 / RTX GPU & CPU.

Accepts:
  - (a) path to test images directory (or single file)
  - (b) path to output directory where restored outputs (.npy) are saved.
  - (c) optional --gt_dir to calculate PSNR, SSIM, and LPIPS benchmark metrics directly.

Supports both positional and keyword CLI invocations:
  python evaluation_script.py /path/to/test_images /path/to/output_dir
  python evaluation_script.py --input_dir /path/to/test_images --output_dir /path/to/output_dir --tta
"""

import os
import sys
import time
import argparse
import glob
import warnings
import numpy as np
import cv2
import torch

warnings.filterwarnings('ignore')
from models.restoration_model import SemiconRestorationNet
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_edge_gradient_correlation


def load_image_file(file_path):
    """
    Loads image from .npy float32 array or standard image file (.png, .jpg, .tif).
    Returns a 2D float32 numpy array.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        img = np.load(file_path).astype(np.float32)
    else:
        # Load grayscale image and normalize to [0, 1]
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image file: {file_path}")
        img = img.astype(np.float32) / 255.0
    return img


def find_weights_file(custom_path=None):
    """Discovers trained model weights from candidate locations."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    candidates.extend([
        'weights/best_model.pth',
        'weights/semicon_restoration_model.pth',
        'best_model.pth',
        'semicon_restoration_model.pth',
        os.path.join(os.path.dirname(__file__), 'weights', 'best_model.pth'),
        os.path.join(os.path.dirname(__file__), 'weights', 'semicon_restoration_model.pth'),
    ])

    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Model weights file not found. Checked candidate paths:\n" + "\n".join(candidates)
    )


def tta_forward(model, x, refine=True):
    """
    8-Fold Geometric Self-Ensemble (Test-Time Augmentation).
    Averages predictions across all 8 dihedral transformations (rotations & flips)
    to cancel residual random noise and boost PSNR and SSIM.
    """
    transforms = [
        (lambda t: t, lambda t: t),
        (lambda t: torch.rot90(t, 1, [2, 3]), lambda t: torch.rot90(t, -1, [2, 3])),
        (lambda t: torch.rot90(t, 2, [2, 3]), lambda t: torch.rot90(t, -2, [2, 3])),
        (lambda t: torch.rot90(t, 3, [2, 3]), lambda t: torch.rot90(t, -3, [2, 3])),
        (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
        (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
        (lambda t: torch.rot90(torch.flip(t, [2]), 1, [2, 3]), lambda t: torch.flip(torch.rot90(t, -1, [2, 3]), [2])),
        (lambda t: torch.rot90(torch.flip(t, [3]), 1, [2, 3]), lambda t: torch.flip(torch.rot90(t, -1, [2, 3]), [3])),
    ]
    outputs = []
    for fwd_fn, inv_fn in transforms:
        pred = model(fwd_fn(x))
        outputs.append(inv_fn(pred))
    
    ensemble = torch.stack(outputs, dim=0).mean(dim=0)
    return ensemble


def run_evaluation(input_path, output_dir, weights_path=None, gt_path=None, device=None, save_png=False, use_tta=False):
    """
    Standalone Evaluation Engine for KLA Benchmarking Team.
    Accepts:
      - input_path: Path to directory containing test images OR path to a single image file (.npy, .png, .jpg)
      - output_dir: Path to directory where restored 256x256 .npy / image files will be saved
      - gt_path: Optional Ground Truth directory or file to evaluate PSNR, SSIM, and LPIPS
      - use_tta: If True, enables 8-fold geometric self-ensemble for maximum PSNR/SSIM boost.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print(" [KLA] Semiconductor Image Restoration Benchmark Engine")
    print(f" Input Path      : {input_path}")
    print(f" Output Directory: {output_dir}")
    print(f" Compute Device  : {device.upper()}")
    print(f" Self-Ensemble   : {'ENABLED (8-Fold TTA)' if use_tta else 'Standard (Fast 1-Pass)'}")
    if gt_path:
        print(f" Ground Truth Ref: {gt_path}")
    if device == 'cuda':
        print(f" GPU Name        : {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Locate and load weights
    actual_weights = find_weights_file(weights_path)
    print(f" Loading model weights from: {actual_weights}")

    model = SemiconRestorationNet().to(device)
    state_dict = torch.load(actual_weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Collect input files
    if os.path.isfile(input_path):
        test_files = [input_path]
    elif os.path.isdir(input_path):
        test_files = sorted(glob.glob(os.path.join(input_path, '*.npy')))
        if len(test_files) == 0:
            # Fallback to image formats
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'):
                test_files.extend(glob.glob(os.path.join(input_path, ext)))
            test_files = sorted(test_files)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if len(test_files) == 0:
        print(f" [Warning] No valid .npy or image files found in '{input_path}'")
        return

    print(f" Restoring {len(test_files)} sample(s)...")

    # Warmup pass (critical for accurate H100 latency benchmarking)
    dummy_input = torch.zeros((1, 1, 128, 128), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    psnr_scores, ssim_scores, lpips_scores = [], [], []

    with torch.no_grad():
        for file_path in test_files:
            img = load_image_file(file_path)

            # Input shape: (1, 1, H, W)
            input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

            # Forward Pass: TTA Self-Ensemble vs Single Pass
            if use_tta:
                output_tensor = tta_forward(model, input_tensor)
            else:
                output_tensor = model(input_tensor)

            # Output array bounded [0.0, 1.0]
            restored_img = output_tensor.cpu().numpy().squeeze(0).squeeze(0)
            restored_img = np.clip(restored_img, 0.0, 1.0).astype(np.float32)

            # Filenames
            filename = os.path.basename(file_path)
            base_name, ext = os.path.splitext(filename)

            # Save restored .npy file for KLA benchmark scoring
            out_npy_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(out_npy_path, restored_img)

            # If GT path specified, compute and record metrics
            if gt_path:
                ref_file = gt_path if os.path.isfile(gt_path) else os.path.join(gt_path, filename)
                if os.path.isfile(ref_file):
                    ref_img = load_image_file(ref_file)
                    p = calculate_psnr(restored_img, ref_img)
                    s = calculate_ssim(restored_img, ref_img)
                    l = calculate_lpips(restored_img, ref_img)
                    psnr_scores.append(p)
                    ssim_scores.append(s)
                    lpips_scores.append(l)

            # Save PNG visual preview for single image or if explicitly requested
            if save_png or len(test_files) == 1 or ext.lower() in ['.png', '.jpg']:
                out_png_path = os.path.join(output_dir, f"{base_name}_restored.png")
                img_uint8 = (restored_img * 255.0).astype(np.uint8)
                cv2.imwrite(out_png_path, img_uint8)

    if device == 'cuda':
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_fps = len(test_files) / max(total_time, 1e-6)
    avg_latency = (total_time / len(test_files)) * 1000.0

    print("=" * 70)
    print(f" Restoration Complete: {len(test_files)} image(s) processed in {total_time:.3f} s")
    print(f" Throughput Speed   : {avg_fps:.1f} FPS")
    print(f" Average Latency    : {avg_latency:.2f} ms per image")
    print(f" Restored Outputs   : {output_dir}")
    if len(psnr_scores) > 0:
        print("-" * 70)
        print(f" [GROUND TRUTH BENCHMARK SUMMARY] ({len(psnr_scores)} pairs)")
        print(f"  Mean PSNR : {np.mean(psnr_scores):.4f} dB")
        print(f"  Mean SSIM : {np.mean(ssim_scores):.4f}")
        print(f"  Mean LPIPS: {np.mean(lpips_scores):.4f}")
    print("=" * 70)


def parse_args():
    """
    Parses CLI arguments supporting both positional and named flag options.
    """
    parser = argparse.ArgumentParser(
        description='Semiconductor Wafer Image Restoration Evaluation CLI (KLA Benchmark Compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation_script.py dataset/NoisyLR restored_test_outputs
  python evaluation_script.py --input_dir dataset/NoisyLR --output_dir restored_test_outputs --tta
  python evaluation_script.py --input_file dataset/NoisyLR/000000.npy --output_dir single_test
        """
    )
    # Positional args (optional if flags provided)
    parser.add_argument('pos_input', nargs='?', default=None, help='Positional: Path to input directory or image file')
    parser.add_argument('pos_output', nargs='?', default=None, help='Positional: Path to output directory')

    # Flag options
    parser.add_argument('--input_dir', '-i', type=str, default=None, help='Path to test images directory')
    parser.add_argument('--input_file', '-f', type=str, default=None, help='Path to single test image file')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='Path to output directory for restored images')
    parser.add_argument('--gt_dir', '-g', type=str, default=None, help='Path to Ground Truth directory/file for metric evaluation')
    parser.add_argument('--weights', '-w', type=str, default=None, help='Path to model checkpoint (.pth)')
    parser.add_argument('--tta', action='store_true', help='Enable 8-fold geometric self-ensemble (boosts PSNR/SSIM/LPIPS)')
    parser.add_argument('--save_png', action='store_true', help='Also save PNG visual previews in batch mode')

    args = parser.parse_args()

    # Resolve target input: --input_file > --input_dir > pos_input > default
    target_input = args.input_file or args.input_dir or args.pos_input
    if target_input is None:
        if os.path.isdir('NoisyLR'):
            target_input = 'NoisyLR'
        elif os.path.isdir('dataset/NoisyLR'):
            target_input = 'dataset/NoisyLR'
        else:
            target_input = 'NoisyLR'

    # Resolve target output: --output_dir > pos_output > default
    target_output = args.output_dir or args.pos_output or 'restored_test_outputs'

    return target_input, target_output, args.weights, args.gt_dir, args.save_png, args.tta


if __name__ == '__main__':
    inp, outp, weights, gt_dir, save_png, use_tta = parse_args()
    run_evaluation(inp, outp, weights, gt_path=gt_dir, save_png=save_png, use_tta=use_tta)
