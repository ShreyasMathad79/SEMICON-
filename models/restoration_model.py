import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    """
    Non-Linear Activation Free Gate (NAFNet component).
    Splits feature channels into two halves and performs elementwise multiplication.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D spatial feature maps.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

class NAFBlock(nn.Module):
    """
    NAFNet-inspired Residual Block with SimpleGate and Simplified Channel Attention.
    Optimized for semiconductor image restoration (denoising + sharpening + super-resolution).
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        
        # Simplified Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1)
        )
        
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0)
        self.norm1 = LayerNorm2d(c)

        # Feed-Forward Network (FFN)
        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0)
        self.norm2 = LayerNorm2d(c)

        self.gamma1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        # Spatial Attention Branch
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.ca(y)
        y = self.conv3(y)
        x = x + y * self.gamma1

        # FFN Branch
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma2
        return x

class SemiconRestorationNet(nn.Module):
    """
    SemiconRestorationNet Architecture:
    Multi-Scale Gated Residual UNet with PixelShuffle 2x Super-Resolution.
    Designed specifically for simultaneous Speckle Noise removal, Gaussian Denoising,
    and 2x Super-Resolution for semiconductor wafer pattern inspection.
    """
    def __init__(self, in_channels=1, out_channels=1, width=24, num_blocks=[1, 1, 2, 1]):
        super().__init__()
        self.width = width

        # Input projection
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        
        # Encoder Level 1
        self.enc1 = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(width, width * 2, kernel_size=2, stride=2)

        # Encoder Level 2
        self.enc2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(width * 2, width * 4, kernel_size=2, stride=2)

        # Bottleneck Level 3
        self.middle = nn.Sequential(*[NAFBlock(width * 4) for _ in range(num_blocks[2])])

        # Decoder Level 2
        self.up2 = nn.Sequential(
            nn.Conv2d(width * 4, width * 4, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.reduce2 = nn.Conv2d(width * 6, width * 2, kernel_size=1)
        self.dec2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(num_blocks[1])])

        # Decoder Level 1
        self.up1 = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.reduce1 = nn.Conv2d(width * 3, width, kernel_size=1)
        self.dec1 = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks[3])])

        # Super-Resolution Upsampling Head (2x scale expansion)
        # Converts 128x128 spatial features to 256x256 high-resolution output
        self.sr_conv = nn.Conv2d(width, width * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.out_conv = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (B, 1, H, W) e.g., (B, 1, 128, 128)
        # Precompute Bicubic baseline upsampling (B, 1, 2*H, 2*W)
        bicubic_baseline = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        
        # Encoder
        feat = self.intro(x)
        e1 = self.enc1(feat)
        
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        
        d2 = self.down2(e2)
        m = self.middle(d2)

        # Decoder
        u2 = self.up2(m)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.reduce2(u2)
        d2_feat = self.dec2(u2)

        u1 = self.up1(d2_feat)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.reduce1(u1)
        d1_feat = self.dec1(u1)

        # Super-resolution pixel shuffle output
        sr_feat = self.pixel_shuffle(self.sr_conv(d1_feat))
        res_output = self.out_conv(sr_feat)

        # Combine residual prediction with bicubic baseline (Dynamic Range Clamping)
        output = torch.clamp(res_output + bicubic_baseline, 0.0, 1.0)
        return output

if __name__ == '__main__':
    model = SemiconRestorationNet()
    test_input = torch.randn(2, 1, 128, 128)
    out = model(test_input)
    print("Model Input Shape:", test_input.shape)
    print("Model Output Shape:", out.shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
