"""
Composite Loss Functions for Semiconductor Wafer Image Restoration.
Combines Charbonnier (Smooth L1), Differentiable Structural Similarity (SSIM),
Multi-Scale Sobel Edge Gradients, 2D Fourier Frequency Domain Loss, and Perceptual Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import SSIMLoss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: A smooth, differentiable variant of L1 loss.
    L(x, y) = sqrt((x - y)^2 + eps^2)
    More robust to severe speckle noise spikes and intensity outliers than standard L1/L2.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class SobelEdgeLoss(nn.Module):
    """
    Computes Multi-Scale Sobel Gradient Magnitude Loss.
    Enforces sharp structural boundaries on semiconductor lines, contact holes, and FinFET gates.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        if self.sobel_x.device != pred.device:
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)

        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        pred_mag = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)

        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        target_mag = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)

        return F.l1_loss(pred_mag, target_mag)


class FrequencyDomainLoss(nn.Module):
    """
    2D Fast Fourier Transform (FFT) Magnitude Loss.
    Penalizes discrepancies in spatial frequency spectrum, preserving periodic pitch
    in DRAM memory arrays, line-space gratings, and logic gates.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Compute real 2D FFT along spatial dimensions
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Log magnitude loss to prevent low-frequency dominance
        pred_log = torch.log1p(pred_mag)
        target_log = torch.log1p(target_mag)

        return F.l1_loss(pred_log, target_log)


class PerceptualLoss(nn.Module):
    """
    Perceptual Feature Consistency Loss using Multi-Scale Gradient Projections.
    Reduces perceptual artifact distance (LPIPS) while preserving edge structures.
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        pred_s1 = self.pool(pred)
        target_s1 = self.pool(target)
        l2 = F.l1_loss(pred_s1, target_s1)
        pred_s2 = self.pool(pred_s1)
        target_s2 = self.pool(target_s1)
        l3 = F.l1_loss(pred_s2, target_s2)
        return l1 * 0.5 + l2 * 0.3 + l3 * 0.2


class CompositeRestorationLoss(nn.Module):
    """
    Full Composite Multi-Domain Loss for Semiconductor Restoration:
    L_total = L_charbonnier + alpha * L_SSIM + beta * L_Sobel + gamma * L_FFT + delta * L_Perceptual
    """
    def __init__(self, alpha=0.20, beta=0.10, gamma=0.05, delta=0.10):
        super().__init__()
        self.charbonnier_loss = CharbonnierLoss(eps=1e-3)
        self.ssim_loss = SSIMLoss()
        self.sobel_loss = SobelEdgeLoss()
        self.fft_loss = FrequencyDomainLoss()
        self.perceptual_loss = PerceptualLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, pred, target):
        l_charb = self.charbonnier_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_edge = self.sobel_loss(pred, target)
        l_fft = self.fft_loss(pred, target)
        l_perc = self.perceptual_loss(pred, target)

        total_loss = l_charb + (self.alpha * l_ssim) + (self.beta * l_edge) + (self.gamma * l_fft) + (self.delta * l_perc)
        return total_loss, l_charb, l_ssim, l_edge
