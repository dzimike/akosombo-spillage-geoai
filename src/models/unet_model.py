from __future__ import annotations
"""
U-Net architecture for flood inundation semantic segmentation.

Implements a standard encoder-decoder U-Net (Ronneberger et al., 2015) in pure
PyTorch without torchvision, accepting an arbitrary number of input channels.

Architecture
------------
Encoder  : 4 × (DoubleConv → MaxPool2d)
Bottleneck: DoubleConv
Decoder  : 4 × (bilinear Upsample + skip concat + DoubleConv)
Head     : Conv2d(64 → 1) + Sigmoid

Input  : (B, C_in, 256, 256)  float32, normalised to [0, 1]
Output : (B, 1, 256, 256)     float32 probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two successive Conv2d-BatchNorm-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """DoubleConv followed by 2×2 max-pooling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class Up(nn.Module):
    """Bilinear upsampling, skip-connection concatenation, DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if the skip and upsampled tensors differ by one pixel (odd input)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Four-level U-Net for binary flood segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input spectral/feature bands.
    base_filters : int
        Number of feature maps in the first encoder block; doubled at each
        subsequent level (default 32 gives 32 → 64 → 128 → 256 → 512).
    """

    def __init__(self, in_channels: int = 28, base_filters: int = 32):
        super().__init__()
        f = base_filters
        # Encoder
        self.down1 = Down(in_channels, f)       # 256 → 128
        self.down2 = Down(f,           f * 2)   # 128 → 64
        self.down3 = Down(f * 2,       f * 4)   # 64  → 32
        self.down4 = Down(f * 4,       f * 8)   # 32  → 16
        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)   # 16 × 16
        # Decoder (in_ch = bottleneck + skip)
        self.up1 = Up(f * 16 + f * 8,  f * 8)
        self.up2 = Up(f * 8  + f * 4,  f * 4)
        self.up3 = Up(f * 4  + f * 2,  f * 2)
        self.up4 = Up(f * 2  + f,       f)
        # Output head — raw logits; sigmoid applied externally in loss/inference
        self.head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)
        b      = self.bottleneck(x4)
        x      = self.up1(b,  s4)
        x      = self.up2(x,  s3)
        x      = self.up3(x,  s2)
        x      = self.up4(x,  s1)
        return self.head(x)

    @staticmethod
    def param_count(model: "UNet") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """
    Weighted combination of Dice loss and Binary Cross-Entropy.

    Dice loss addresses class imbalance by optimising the overlap coefficient
    directly; BCE provides stable gradient behaviour in the early training phase.

    Parameters
    ----------
    dice_weight : float
        Weighting for the Dice component (BCE weight = 1 - dice_weight).
    smooth : float
        Laplace smoothing constant to avoid division by zero.
    pos_weight : float or None
        Positive-class weight passed to BCE to further address imbalance.
    """

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0,
                 pos_weight: float | None = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth      = smooth
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """targets: (B,1,H,W) labels, or (B,2,H,W) = [label, valid]; pixels with
        valid == 0 (permanent water) contribute to neither loss term."""
        if targets.shape[1] == 2:
            targets, valid = targets[:, :1], targets[:, 1:]
        else:
            valid = torch.ones_like(targets)
        probs = torch.sigmoid(logits)

        # Dice component
        p_flat = (probs * valid).reshape(-1)
        t_flat = (targets * valid).reshape(-1)
        inter  = (p_flat * t_flat).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
            p_flat.sum() + t_flat.sum() + self.smooth)

        # BCE component (operates on logits for numerical stability)
        bce_loss = (self.bce(logits, targets) * valid).sum() / valid.sum().clamp(min=1.0)

        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss
