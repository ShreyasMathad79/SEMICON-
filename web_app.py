"""
KLA Semiconductor Image Restoration & Wafer Metrology Visualizer Studio.
Interactive Web Dashboard featuring single/paired image restoration, 1D line profile cross-sections,
live HUD metrology metrics (PSNR, SSIM, LPIPS, Edge Correlation, Latency), and Drift-Sense navigation recovery.
"""

import os
import sys
import glob
import io
import json
import base64
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
from PIL import Image
import torch
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

from models.restoration_model import SemiconRestorationNet
from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    calculate_edge_gradient_correlation,
    calculate_speckle_attenuation,
    calculate_tenengrad_sharpness,
    calculate_cnr,
    calculate_estimated_snr_gain
)
from drift_sense import DriftSenseRecovery

# Global Initialization
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = SemiconRestorationNet().to(DEVICE)

# Discovered Weights
WEIGHTS_PATHS = [
    'weights/best_model.pth',
    'weights/semicon_restoration_model.pth',
    'best_model.pth',
    'semicon_restoration_model.pth'
]
for wp in WEIGHTS_PATHS:
    if os.path.exists(wp):
        try:
            MODEL.load_state_dict(torch.load(wp, map_location=DEVICE))
            print(f" Loaded model weights from {wp}")
            break
        except Exception as e:
            print(f" Could not load {wp}: {e}")
MODEL.eval()

# Pre-warm model
dummy = torch.zeros((1, 1, 128, 128), dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    for _ in range(2):
        _ = MODEL(dummy)

# Generate Synthetic Benchmark Paired Samples if not present
def setup_benchmark_pairs():
    os.makedirs('benchmark_pairs/degraded', exist_ok=True)
    os.makedirs('benchmark_pairs/gt', exist_ok=True)

    # 1. FinFET Periodic Array
    f1_gt = 'benchmark_pairs/gt/01_finfet_array.npy'
    f1_deg = 'benchmark_pairs/degraded/01_finfet_array.npy'
    if not os.path.exists(f1_gt):
        y, x = np.mgrid[0:256, 0:256]
        gt1 = 0.5 + 0.4 * np.sin(2 * np.pi * x / 16.0)
        gt1 = np.where(gt1 > 0.5, 0.85, 0.15).astype(np.float32)
        lr1 = cv2.resize(gt1, (128, 128), interpolation=cv2.INTER_AREA)
        lr1 = cv2.GaussianBlur(lr1, (3, 3), 0.8)
        noise1 = np.random.normal(0, 0.10, lr1.shape).astype(np.float32)
        lr1 = np.clip(lr1 + noise1, 0.0, 1.4).astype(np.float32)
        np.save(f1_gt, gt1)
        np.save(f1_deg, lr1)

    # 2. Contact Hole Array
    f2_gt = 'benchmark_pairs/gt/02_contact_holes.npy'
    f2_deg = 'benchmark_pairs/degraded/02_contact_holes.npy'
    if not os.path.exists(f2_gt):
        gt2 = np.zeros((256, 256), dtype=np.float32) + 0.18
        for row in range(32, 256, 48):
            for col in range(32, 256, 48):
                cv2.circle(gt2, (col, row), 14, 0.88, -1)
        lr2 = cv2.resize(gt2, (128, 128), interpolation=cv2.INTER_AREA)
        lr2 = cv2.GaussianBlur(lr2, (3, 3), 0.8)
        noise2 = np.random.normal(0, 0.10, lr2.shape).astype(np.float32)
        lr2 = np.clip(lr2 + noise2, 0.0, 1.4).astype(np.float32)
        np.save(f2_gt, gt2)
        np.save(f2_deg, lr2)

setup_benchmark_pairs()

def tta_infer(model, x):
    """8-Fold Dihedral Self-Ensemble."""
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
    return torch.stack(outputs, dim=0).mean(dim=0)

def get_sample_catalog():
    """Builds unified catalog of paired benchmarks and unlabelled test images."""
    catalog = []
    
    # 1. Paired Benchmarks
    pair_degs = sorted(glob.glob('benchmark_pairs/degraded/*.npy'))
    for p in pair_degs:
        bname = os.path.basename(p)
        gt_p = os.path.join('benchmark_pairs', 'gt', bname)
        if os.path.exists(gt_p):
            catalog.append({
                'name': f"[PAIRED BENCHMARK] {bname}",
                'type': 'paired',
                'noisy_path': p,
                'gt_path': gt_p
            })

    # 2. Check if dataset/train or train has paired GT
    train_noisy = sorted(glob.glob('train/NoisyLR/*.npy') + glob.glob('dataset/train/NoisyLR/*.npy'))
    for p in train_noisy:
        bname = os.path.basename(p)
        gt_cand1 = os.path.join(os.path.dirname(os.path.dirname(p)), 'GT', bname)
        if os.path.exists(gt_cand1):
            catalog.append({
                'name': f"[TRAIN PAIRED] {bname}",
                'type': 'paired',
                'noisy_path': p,
                'gt_path': gt_cand1
            })

    # 3. Test Blind NoisyLR dataset
    noisy_files = sorted(glob.glob('NoisyLR/*.npy'))
    for p in noisy_files:
        bname = os.path.basename(p)
        # Check if GT exists in any GT folder
        gt_cand = os.path.join('GT', bname)
        if os.path.exists(gt_cand):
            catalog.append({
                'name': f"[PAIRED] {bname}",
                'type': 'paired',
                'noisy_path': p,
                'gt_path': gt_cand
            })
        else:
            catalog.append({
                'name': f"[TEST WAFER] {bname}",
                'type': 'blind',
                'noisy_path': p,
                'gt_path': None
            })

    return catalog

SAMPLE_CATALOG = get_sample_catalog()

def ndarray_to_b64png(arr):
    """Converts 2D float array [0.0, 1.0] to base64 PNG string."""
    arr_norm = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr_norm * 255.0).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_edge_map_b64(img):
    """Generates Sobel edge energy map for blind sample reference card."""
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = mag / (np.percentile(mag, 99.5) + 1e-6)
    return ndarray_to_b64png(np.clip(mag_norm, 0.0, 1.0))


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KLA Semiconductor Metrology & AI Restoration Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070B12;
            --panel-bg: #0F172A;
            --card-bg: #1E293B;
            --cyan-accent: #06B6D4;
            --emerald-accent: #10B981;
            --gold-accent: #F59E0B;
            --rose-accent: #F43F5E;
            --white-text: #F8FAFC;
            --muted-text: #94A3B8;
            --border-color: #334155;
            --accent-glow: rgba(6, 182, 212, 0.25);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-dark);
            color: var(--white-text);
            padding: 20px 28px;
            min-height: 100vh;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 18px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        .brand { display: flex; align-items: center; gap: 14px; }
        .logo-icon {
            width: 40px; height: 40px; background: linear-gradient(135deg, var(--cyan-accent), var(--emerald-accent));
            border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 22px; color: #070B12;
            box-shadow: 0 0 20px var(--accent-glow);
        }
        h1 { font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }
        .badge {
            font-size: 11px; background: rgba(6, 182, 212, 0.12); color: var(--cyan-accent);
            padding: 5px 12px; border-radius: 20px; border: 1px solid var(--cyan-accent); font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge.green {
            background: rgba(16, 185, 129, 0.12); color: var(--emerald-accent); border-color: var(--emerald-accent);
        }
        .main-layout { display: grid; grid-template-columns: 380px 1fr; gap: 24px; }
        
        .sidebar {
            background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 22px;
            display: flex; flex-direction: column; gap: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .section-title { font-size: 13px; font-weight: 800; color: var(--cyan-accent); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        label { font-size: 12px; color: var(--muted-text); margin-bottom: 6px; display: block; font-weight: 600; }
        select, input[type="file"], button, input[type="range"] {
            width: 100%; padding: 11px 14px; background: var(--bg-dark); color: var(--white-text); border: 1px solid var(--border-color);
            border-radius: 8px; font-size: 13px; font-family: inherit; cursor: pointer; outline: none; transition: all 0.2s;
        }
        select:focus, input[type="file"]:focus { border-color: var(--cyan-accent); box-shadow: 0 0 0 2px var(--accent-glow); }
        
        .toggle-box {
            display: flex; align-items: center; justify-content: space-between; background: var(--bg-dark);
            padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border-color); margin-top: 8px;
        }
        .toggle-switch { position: relative; width: 44px; height: 24px; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background-color: #334155; transition: .3s; border-radius: 24px;
        }
        .toggle-slider:before {
            position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px;
            background-color: white; transition: .3s; border-radius: 50%;
        }
        input:checked + .toggle-slider { background-color: var(--emerald-accent); }
        input:checked + .toggle-slider:before { transform: translateX(20px); }

        button.btn-primary {
            background: linear-gradient(135deg, var(--cyan-accent), #0284c7); color: #070B12; font-weight: 800; border: none; margin-top: 8px;
            display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        button.btn-primary:hover { opacity: 0.92; box-shadow: 0 0 20px var(--accent-glow); transform: translateY(-1px); }

        .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border-color); padding-bottom: 12px; }
        .tab-btn {
            background: transparent; border: 1px solid var(--border-color); padding: 8px 12px; font-size: 12px;
            border-radius: 6px; font-weight: 600; color: var(--muted-text);
        }
        .tab-btn.active {
            background: rgba(6, 182, 212, 0.15); color: var(--cyan-accent); border-color: var(--cyan-accent);
        }

        .upload-area {
            border: 2px dashed var(--border-color); border-radius: 8px; padding: 14px; text-align: center;
            background: rgba(15, 23, 42, 0.6); transition: all 0.2s;
        }
        .upload-area:hover { border-color: var(--cyan-accent); background: rgba(6, 182, 212, 0.05); }

        .spec-box {
            background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; padding: 14px; font-size: 12px; line-height: 1.6; color: var(--muted-text);
        }
        .spec-box strong { color: var(--white-text); }

        .viewer-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
        .image-card {
            background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 16px; text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        }
        .image-card.degraded h3 { color: var(--cyan-accent); }
        .image-card.restored h3 { color: var(--emerald-accent); }
        .image-card.gt h3 { color: #E2E8F0; }
        .image-card h3 { font-size: 13px; font-weight: 800; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .img-canvas-wrap {
            background: #000; border-radius: 8px; overflow: hidden; height: 280px; display: flex; align-items: center; justify-content: center; position: relative;
            border: 1px solid #1E293B;
        }
        .img-canvas-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: pixelated; }
        .card-meta { font-size: 11px; margin-top: 10px; color: var(--muted-text); font-family: 'JetBrains Mono', monospace; }

        .metrics-hud {
            margin-top: 20px; background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 18px 22px;
            display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .stat-item { text-align: center; border-right: 1px solid var(--border-color); padding: 0 8px; }
        .stat-item:last-child { border-right: none; }
        .stat-label { font-size: 10px; color: var(--muted-text); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; font-weight: 700; }
        .stat-val { font-size: 18px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
        .stat-val.green { color: var(--emerald-accent); }
        .stat-val.cyan { color: var(--cyan-accent); }
        .stat-val.gold { color: var(--gold-accent); }
        .stat-sub { font-size: 10px; color: var(--muted-text); margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

        .profile-container {
            margin-top: 20px; background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 18px 22px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .profile-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .slider-wrap { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--muted-text); }
        .slider-wrap input[type="range"] { width: 140px; }
        canvas#lineProfileCanvas { width: 100%; height: 130px; background: #000; border-radius: 8px; border: 1px solid #1E293B; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="logo-icon">⚡</div>
            <div>
                <h1>KLA Semiconductor Metrology & AI Restoration Studio</h1>
                <div style="font-size: 12px; color: var(--muted-text); margin-top: 2px;">Sub-Nanometer Wafer Image Restoration, Live PSNR/SSIM/LPIPS Telemetry & Drift Recovery</div>
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="badge" id="hudModeBadge">FULL REFERENCE METROLOGY</span>
            <span class="badge green">PYTORCH 2.0+ ACTIVE</span>
        </div>
    </header>

    <div class="main-layout">
        <div class="sidebar">
            <div class="tabs">
                <button class="tab-btn active" id="tabBtnDataset" onclick="switchTab('dataset')">1. Catalog Inspection</button>
                <button class="tab-btn" id="tabBtnUpload" onclick="switchTab('upload')">2. Custom Dual-Upload</button>
            </div>

            <!-- Tab 1: Dataset Samples -->
            <div id="panelDataset">
                <div class="section-title">
                    <span>Wafer Sample Select</span>
                    <span id="sampleTypeBadge" class="badge" style="font-size: 10px; padding: 2px 8px;">PAIRED</span>
                </div>
                <label for="sampleSelect">Select Test Sample or Paired Benchmark:</label>
                <select id="sampleSelect" onchange="loadSampleInference()"></select>

                <div class="toggle-box">
                    <div>
                        <div style="font-size: 12px; font-weight: 700; color: var(--white-text);">⚡ 8-Fold TTA Self-Ensemble</div>
                        <div style="font-size: 10px; color: var(--muted-text);">Boosts PSNR (+0.5 dB), SSIM & LPIPS</div>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="ttaToggle" onchange="loadSampleInference()" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <button class="btn-primary" onclick="loadSampleInference()">⚡ Run Metrology Restoration</button>
            </div>

            <!-- Tab 2: Custom Upload -->
            <div id="panelUpload" style="display: none;">
                <div class="section-title">
                    <span>Custom Pair / Single Upload</span>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <label>1. Degraded Low-Res Image (.npy / .png / .jpg):</label>
                    <div class="upload-area">
                        <input type="file" id="fileDegraded" accept=".npy,.png,.jpg,.jpeg,.tif">
                    </div>
                </div>

                <div style="margin-bottom: 12px;">
                    <label>2. Ground Truth Reference (Optional for PSNR/SSIM/LPIPS):</label>
                    <div class="upload-area">
                        <input type="file" id="fileGT" accept=".npy,.png,.jpg,.jpeg,.tif">
                    </div>
                </div>

                <div class="toggle-box">
                    <div>
                        <div style="font-size: 12px; font-weight: 700; color: var(--white-text);">⚡ 8-Fold TTA Self-Ensemble</div>
                        <div style="font-size: 10px; color: var(--muted-text);">Maximum Precision Averaging</div>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="ttaToggleUpload" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <button class="btn-primary" onclick="uploadCustomPair()">⚡ Upload & Benchmark</button>
            </div>

            <div class="spec-box">
                <strong>SemiconRestorationNet Architecture:</strong><br>
                • Gated Non-Linear Activation (NAFNet SimpleGate)<br>
                • 2× PixelShuffle Sub-Pixel Super-Resolution Head<br>
                • Multi-Domain Loss (Charbonnier + SSIM + Sobel + FFT + LPIPS)<br>
                • Dynamic Range Soft Clamping & Adaptive Refinement<br>
                • Parameter Size: 1.02 MB (321,793 params)
            </div>
        </div>

        <div class="main-content">
            <div class="viewer-container">
                <div class="image-card degraded">
                    <h3>1. Degraded Input (128×128)</h3>
                    <div class="img-canvas-wrap"><img id="imgDegraded" src="" alt="Degraded"></div>
                    <div class="card-meta" id="metaDegraded">Range: --</div>
                </div>

                <div class="image-card restored">
                    <h3>2. AI Restored Output (256×256)</h3>
                    <div class="img-canvas-wrap"><img id="imgRestored" src="" alt="Restored"></div>
                    <div class="card-meta" style="color: var(--emerald-accent);" id="metaRestored">Denoised + 2x Super-Resolved</div>
                </div>

                <div class="image-card gt">
                    <h3 id="gtCardTitle">3. Ground Truth Reference (256×256)</h3>
                    <div class="img-canvas-wrap"><img id="imgGT" src="" alt="Ground Truth"></div>
                    <div class="card-meta" id="metaGT">Reference High-SNR Ground Truth</div>
                </div>
            </div>

            <!-- Full Telemetry Metrics HUD -->
            <div class="metrics-hud">
                <div class="stat-item">
                    <div class="stat-label">PSNR METRIC</div>
                    <div class="stat-val green" id="hudPSNR">-- dB</div>
                    <div class="stat-sub" id="hudPSNRSub">Gain: -- dB</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">SSIM SCORE</div>
                    <div class="stat-val green" id="hudSSIM">--</div>
                    <div class="stat-sub" id="hudSSIMSub">Structural Fidelity</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">LPIPS DISTANCE</div>
                    <div class="stat-val cyan" id="hudLPIPS">--</div>
                    <div class="stat-sub">Lower is Better (0.0=Exact)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">EDGE CORRELATION</div>
                    <div class="stat-val cyan" id="hudEdge">--</div>
                    <div class="stat-sub" id="hudEdgeSub">Cosine Sharpness</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">INFERENCE LATENCY</div>
                    <div class="stat-val gold" id="hudLatency">-- ms</div>
                    <div class="stat-sub" id="hudSpeed">-- FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">BASELINE COMPARISON</div>
                    <div class="stat-val" id="hudBicubic">-- dB</div>
                    <div class="stat-sub" id="hudBicubicSub">Bicubic Baseline</div>
                </div>
            </div>

            <div class="profile-container">
                <div class="profile-header">
                    <div class="section-title" style="margin: 0;">1D Horizontal Line Profile Cross-Section</div>
                    <div class="slider-wrap">
                        <span>Cross-Section Row:</span>
                        <input type="range" id="rowSlider" min="0" max="255" value="128" oninput="updateRowSlider(this.value)">
                        <span id="rowLabel" style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--cyan-accent);">#128</span>
                    </div>
                    <div style="font-size: 11px; color: var(--muted-text);">
                        <span style="color: #06B6D4;">■ Degraded</span> &nbsp;
                        <span style="color: #10B981;">■ Restored AI</span> &nbsp;
                        <span style="color: #FFFFFF;" id="legendGT">■ Ground Truth</span>
                    </div>
                </div>
                <canvas id="lineProfileCanvas" width="800" height="130"></canvas>
            </div>
        </div>
    </div>

    <script>
        let currentFullData = null;

        function switchTab(tab) {
            if (tab === 'dataset') {
                document.getElementById('tabBtnDataset').classList.add('active');
                document.getElementById('tabBtnUpload').classList.remove('active');
                document.getElementById('panelDataset').style.display = 'block';
                document.getElementById('panelUpload').style.display = 'none';
            } else {
                document.getElementById('tabBtnUpload').classList.add('active');
                document.getElementById('tabBtnDataset').classList.remove('active');
                document.getElementById('panelUpload').style.display = 'block';
                document.getElementById('panelDataset').style.display = 'none';
            }
        }

        async function initPage() {
            const res = await fetch('/api/samples');
            const samples = await res.json();
            const select = document.getElementById('sampleSelect');
            select.innerHTML = '';
            samples.forEach((s, idx) => {
                const opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = s.name;
                select.appendChild(opt);
            });
            if (samples.length > 0) {
                loadSampleInference();
            }
        }

        async function loadSampleInference() {
            const idx = document.getElementById('sampleSelect').value || 0;
            const useTTA = document.getElementById('ttaToggle').checked;
            const row = document.getElementById('rowSlider').value || 128;

            const res = await fetch(`/api/restore?idx=${idx}&tta=${useTTA}&row=${row}`);
            const data = await res.json();
            renderResultData(data);
        }

        async function uploadCustomPair() {
            const fileDegraded = document.getElementById('fileDegraded').files[0];
            const fileGT = document.getElementById('fileGT').files[0];
            const useTTA = document.getElementById('ttaToggleUpload').checked;
            const row = document.getElementById('rowSlider').value || 128;

            if (!fileDegraded) {
                alert("Please select at least a degraded input image file.");
                return;
            }

            const formData = new FormData();
            formData.append('degraded', fileDegraded);
            if (fileGT) formData.append('gt', fileGT);
            formData.append('tta', useTTA);
            formData.append('row', row);

            const res = await fetch('/api/restore_pair', { method: 'POST', body: formData });
            const data = await res.json();
            renderResultData(data);
        }

        function renderResultData(data) {
            currentFullData = data;

            document.getElementById('imgDegraded').src = 'data:image/png;base64,' + data.b64_degraded;
            document.getElementById('imgRestored').src = 'data:image/png;base64,' + data.b64_restored;
            document.getElementById('imgGT').src = 'data:image/png;base64,' + data.b64_gt;

            document.getElementById('metaDegraded').textContent = `Input Range: [${data.min_val.toFixed(2)}, ${data.max_val.toFixed(2)}] (128x128)`;
            document.getElementById('metaRestored').textContent = `Restored [0.0, 1.0] (256x256 Super-Resolved)`;

            if (data.is_paired) {
                document.getElementById('hudModeBadge').textContent = 'FULL REFERENCE METROLOGY';
                document.getElementById('hudModeBadge').style.borderColor = 'var(--emerald-accent)';
                document.getElementById('hudModeBadge').style.color = 'var(--emerald-accent)';
                document.getElementById('sampleTypeBadge').textContent = 'PAIRED GT';
                document.getElementById('sampleTypeBadge').style.color = 'var(--emerald-accent)';
                document.getElementById('gtCardTitle').textContent = '3. Ground Truth Reference (256×256)';
                document.getElementById('metaGT').textContent = 'High-SNR Clean Reference';
                document.getElementById('legendGT').style.display = 'inline';

                document.getElementById('hudPSNR').textContent = `${data.psnr_model.toFixed(2)} dB`;
                const gain = data.psnr_model - data.psnr_bicubic;
                document.getElementById('hudPSNRSub').textContent = `Net Gain: +${gain.toFixed(2)} dB`;

                document.getElementById('hudSSIM').textContent = `${data.ssim_model.toFixed(4)}`;
                document.getElementById('hudSSIMSub').textContent = `Target: > 0.8800`;

                document.getElementById('hudLPIPS').textContent = `${data.lpips_model.toFixed(4)}`;
                document.getElementById('hudEdge').textContent = `${data.edge_corr.toFixed(4)}`;
                document.getElementById('hudEdgeSub').textContent = `Gradient Alignment`;

                document.getElementById('hudBicubic').textContent = `${data.psnr_bicubic.toFixed(2)} dB`;
                document.getElementById('hudBicubicSub').textContent = `Bicubic Baseline`;
            } else {
                document.getElementById('hudModeBadge').textContent = 'BLIND TEST METROLOGY (ESTIMATED)';
                document.getElementById('hudModeBadge').style.borderColor = 'var(--gold-accent)';
                document.getElementById('hudModeBadge').style.color = 'var(--gold-accent)';
                document.getElementById('sampleTypeBadge').textContent = 'BLIND TEST';
                document.getElementById('sampleTypeBadge').style.color = 'var(--gold-accent)';
                document.getElementById('gtCardTitle').textContent = '3. Edge Energy Analysis (Blind Mode)';
                document.getElementById('metaGT').textContent = 'Sobel High-Frequency Energy Map';
                document.getElementById('legendGT').style.display = 'none';

                document.getElementById('hudPSNR').textContent = `+${data.est_snr_gain.toFixed(2)} dB`;
                document.getElementById('hudPSNRSub').textContent = `Est. SNR Gain (Blind)`;

                document.getElementById('hudSSIM').textContent = `${data.speckle_att.toFixed(1)}%`;
                document.getElementById('hudSSIMSub').textContent = `Speckle Noise Attenuation`;

                document.getElementById('hudLPIPS').textContent = `${data.lpips_model.toFixed(4)}`;
                document.getElementById('hudEdge').textContent = `${data.tenengrad_score.toFixed(1)}`;
                document.getElementById('hudEdgeSub').textContent = `Tenengrad Edge Energy`;

                document.getElementById('hudBicubic').textContent = `CNR ${data.cnr_score.toFixed(1)}`;
                document.getElementById('hudBicubicSub').textContent = `Contrast-to-Noise Ratio`;
            }

            document.getElementById('hudLatency').textContent = `${data.latency_ms.toFixed(1)} ms`;
            const fps = 1000.0 / Math.max(data.latency_ms, 1e-3);
            document.getElementById('hudSpeed').textContent = `${fps.toFixed(0)} FPS Throughput`;

            drawLineProfile(data.profile);
        }

        async function updateRowSlider(val) {
            document.getElementById('rowLabel').textContent = `#${val}`;
            if (!currentFullData) return;
            const idx = document.getElementById('sampleSelect').value || 0;
            const useTTA = document.getElementById('ttaToggle').checked;
            const res = await fetch(`/api/restore?idx=${idx}&tta=${useTTA}&row=${val}`);
            const data = await res.json();
            currentFullData.profile = data.profile;
            drawLineProfile(data.profile);
        }

        function drawLineProfile(profile) {
            if (!profile) return;
            const canvas = document.getElementById('lineProfileCanvas');
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;

            ctx.fillStyle = '#070B12';
            ctx.fillRect(0, 0, w, h);

            // Draw grid lines
            ctx.strokeStyle = '#1E293B';
            ctx.lineWidth = 1;
            for (let y = 0; y < h; y += 25) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            function drawCurve(arr, color, lineWidth) {
                if (!arr || arr.length === 0) return;
                ctx.strokeStyle = color;
                ctx.lineWidth = lineWidth;
                ctx.beginPath();
                const step = w / (arr.length - 1);
                arr.forEach((v, i) => {
                    const x = i * step;
                    const y = h - (Math.max(0, Math.min(1.2, v)) * (h - 20) / 1.2 + 10);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            if (profile.gt) drawCurve(profile.gt, '#FFFFFF', 1.8);
            if (profile.degraded) drawCurve(profile.degraded, '#06B6D4', 1.0);
            if (profile.restored) drawCurve(profile.restored, '#10B981', 2.0);
        }

        window.onload = initPage;
    </script>
</body>
</html>
"""

class VisualizerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif path == '/api/samples':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            samples_info = [{'idx': i, 'name': s['name'], 'type': s['type']} for i, s in enumerate(SAMPLE_CATALOG)]
            self.wfile.write(json.dumps(samples_info).encode('utf-8'))
        elif path == '/api/restore':
            idx = int(query.get('idx', [0])[0])
            use_tta = query.get('tta', ['true'])[0].lower() in ['true', '1', 'yes']
            row_coord = int(query.get('row', [128])[0])
            row_coord = max(0, min(255, row_coord))

            if idx < 0 or idx >= len(SAMPLE_CATALOG):
                idx = 0

            item = SAMPLE_CATALOG[idx]
            noisy_img = np.load(item['noisy_path']).astype(np.float32)
            gt_img = np.load(item['gt_path']).astype(np.float32) if item['gt_path'] and os.path.exists(item['gt_path']) else None

            start = time.time()
            input_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if use_tta:
                    restored_tensor = tta_infer(MODEL, input_tensor)
                else:
                    restored_tensor = MODEL(input_tensor)
            latency_ms = (time.time() - start) * 1000.0

            restored_img = restored_tensor.cpu().numpy().squeeze(0).squeeze(0)
            restored_img = np.clip(restored_img, 0.0, 1.0).astype(np.float32)

            bicubic_img = cv2.resize(noisy_img, (256, 256), interpolation=cv2.INTER_CUBIC)
            bicubic_img = np.clip(bicubic_img, 0.0, 1.0)

            is_paired = gt_img is not None
            if is_paired:
                psnr_model = calculate_psnr(restored_img, gt_img)
                ssim_model = calculate_ssim(restored_img, gt_img)
                lpips_model = calculate_lpips(restored_img, gt_img)
                edge_corr = calculate_edge_gradient_correlation(restored_img, gt_img)
                psnr_bicubic = calculate_psnr(bicubic_img, gt_img)
                b64_gt = ndarray_to_b64png(gt_img)
                speckle_att = calculate_speckle_attenuation(noisy_img, restored_img)
                est_snr_gain = psnr_model - psnr_bicubic
                tenengrad_score = calculate_tenengrad_sharpness(restored_img)
                cnr_score = calculate_cnr(restored_img)
            else:
                psnr_model = 0.0
                ssim_model = 0.0
                lpips_model = calculate_lpips(restored_img, bicubic_img)
                edge_corr = 0.0
                psnr_bicubic = 0.0
                b64_gt = generate_edge_map_b64(restored_img)
                speckle_att = calculate_speckle_attenuation(noisy_img, restored_img)
                est_snr_gain = calculate_estimated_snr_gain(noisy_img, restored_img)
                tenengrad_score = calculate_tenengrad_sharpness(restored_img)
                cnr_score = calculate_cnr(restored_img)

            # Extract 1D cross-section profile at requested row
            lr_row = int(row_coord // 2)
            profile_deg = cv2.resize(noisy_img[lr_row:lr_row+1, :], (256, 1), interpolation=cv2.INTER_NEAREST)[0].tolist()
            profile_res = restored_img[row_coord, :].tolist()
            profile_gt = gt_img[row_coord, :].tolist() if is_paired else None

            resp = {
                'is_paired': is_paired,
                'b64_degraded': ndarray_to_b64png(noisy_img),
                'b64_restored': ndarray_to_b64png(restored_img),
                'b64_gt': b64_gt,
                'min_val': float(noisy_img.min()),
                'max_val': float(noisy_img.max()),
                'psnr_model': float(psnr_model),
                'ssim_model': float(ssim_model),
                'lpips_model': float(lpips_model),
                'edge_corr': float(edge_corr),
                'psnr_bicubic': float(psnr_bicubic),
                'est_snr_gain': float(est_snr_gain),
                'speckle_att': float(speckle_att),
                'tenengrad_score': float(tenengrad_score),
                'cnr_score': float(cnr_score),
                'latency_ms': float(latency_ms),
                'profile': {
                    'degraded': profile_deg,
                    'restored': profile_res,
                    'gt': profile_gt
                }
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ['/api/restore_pair', '/api/restore_single']:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            try:
                # Parse multipart boundaries
                boundary = None
                content_type = self.headers.get('Content-Type', '')
                if 'boundary=' in content_type:
                    boundary = content_type.split('boundary=')[1].encode('utf-8')

                noisy_img = None
                gt_img = None
                use_tta = True
                row_coord = 128

                if boundary and boundary in body:
                    parts = body.split(b'--' + boundary)
                    for p in parts:
                        if b'name="degraded"' in p:
                            file_bytes = p.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n', 1)[0]
                            try:
                                noisy_img = np.load(io.BytesIO(file_bytes)).astype(np.float32)
                            except Exception:
                                nparr = np.frombuffer(file_bytes, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                                noisy_img = img.astype(np.float32) / 255.0
                        elif b'name="gt"' in p:
                            file_bytes = p.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n', 1)[0]
                            try:
                                gt_img = np.load(io.BytesIO(file_bytes)).astype(np.float32)
                            except Exception:
                                nparr = np.frombuffer(file_bytes, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    gt_img = img.astype(np.float32) / 255.0
                        elif b'name="tta"' in p:
                            val_str = p.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n', 1)[0].decode('utf-8', errors='ignore')
                            use_tta = val_str.lower() in ['true', '1', 'yes']
                        elif b'name="row"' in p:
                            try:
                                row_str = p.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n', 1)[0].decode('utf-8', errors='ignore')
                                row_coord = int(row_str)
                            except Exception:
                                row_coord = 128

                if noisy_img is None:
                    # Fallback single file payload
                    file_bytes = body
                    if b'\r\n\r\n' in body:
                        file_bytes = body.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n--', 1)[0]
                    try:
                        noisy_img = np.load(io.BytesIO(file_bytes)).astype(np.float32)
                    except Exception:
                        nparr = np.frombuffer(file_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            img = np.zeros((128, 128), dtype=np.float32)
                        noisy_img = img.astype(np.float32) / 255.0

                start = time.time()
                input_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    if use_tta:
                        restored_tensor = tta_infer(MODEL, input_tensor)
                    else:
                        restored_tensor = MODEL(input_tensor)
                latency_ms = (time.time() - start) * 1000.0

                restored_img = restored_tensor.cpu().numpy().squeeze(0).squeeze(0)
                restored_img = np.clip(restored_img, 0.0, 1.0).astype(np.float32)

                bicubic_img = cv2.resize(noisy_img, (256, 256), interpolation=cv2.INTER_CUBIC)
                bicubic_img = np.clip(bicubic_img, 0.0, 1.0)

                is_paired = gt_img is not None
                if is_paired:
                    psnr_model = calculate_psnr(restored_img, gt_img)
                    ssim_model = calculate_ssim(restored_img, gt_img)
                    lpips_model = calculate_lpips(restored_img, gt_img)
                    edge_corr = calculate_edge_gradient_correlation(restored_img, gt_img)
                    psnr_bicubic = calculate_psnr(bicubic_img, gt_img)
                    b64_gt = ndarray_to_b64png(gt_img)
                    speckle_att = calculate_speckle_attenuation(noisy_img, restored_img)
                    est_snr_gain = psnr_model - psnr_bicubic
                    tenengrad_score = calculate_tenengrad_sharpness(restored_img)
                    cnr_score = calculate_cnr(restored_img)
                else:
                    psnr_model = 0.0
                    ssim_model = 0.0
                    lpips_model = calculate_lpips(restored_img, bicubic_img)
                    edge_corr = 0.0
                    psnr_bicubic = 0.0
                    b64_gt = generate_edge_map_b64(restored_img)
                    speckle_att = calculate_speckle_attenuation(noisy_img, restored_img)
                    est_snr_gain = calculate_estimated_snr_gain(noisy_img, restored_img)
                    tenengrad_score = calculate_tenengrad_sharpness(restored_img)
                    cnr_score = calculate_cnr(restored_img)

                row_coord = max(0, min(255, row_coord))
                lr_row = int(row_coord // 2)
                profile_deg = cv2.resize(noisy_img[lr_row:lr_row+1, :], (256, 1), interpolation=cv2.INTER_NEAREST)[0].tolist()
                profile_res = restored_img[row_coord, :].tolist()
                profile_gt = gt_img[row_coord, :].tolist() if is_paired else None

                resp = {
                    'is_paired': is_paired,
                    'b64_degraded': ndarray_to_b64png(noisy_img),
                    'b64_restored': ndarray_to_b64png(restored_img),
                    'b64_gt': b64_gt,
                    'min_val': float(noisy_img.min()),
                    'max_val': float(noisy_img.max()),
                    'psnr_model': float(psnr_model),
                    'ssim_model': float(ssim_model),
                    'lpips_model': float(lpips_model),
                    'edge_corr': float(edge_corr),
                    'psnr_bicubic': float(psnr_bicubic),
                    'est_snr_gain': float(est_snr_gain),
                    'speckle_att': float(speckle_att),
                    'tenengrad_score': float(tenengrad_score),
                    'cnr_score': float(cnr_score),
                    'latency_ms': float(latency_ms),
                    'profile': {
                        'degraded': profile_deg,
                        'restored': profile_res,
                        'gt': profile_gt
                    }
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))


def run_server(port=8050):
    server = HTTPServer(('127.0.0.1', port), VisualizerHandler)
    print("=" * 65)
    print(f" [KLA Studio] AI Metrology & Restoration Studio Running")
    print(f" URL: http://127.0.0.1:{port}")
    print("=" * 65)
    server.serve_forever()


if __name__ == '__main__':
    run_server()
