"""
U-Net for satellite land-cover semantic segmentation.

Classes (5):
    0 = Urban
    1 = Forest
    2 = Water
    3 = Agriculture
    4 = Bare land

Works with RGB (3-channel) or multispectral (e.g. RGB+NIR, 4-channel) input —
just change `in_channels`. Using a 4th NIR channel lets the network learn a
proper NDVI-like signal for vegetation/agriculture instead of relying on RGB
color alone, which is strongly recommended for real satellite data.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["Urban", "Forest", "Water", "Agriculture", "Bare land"]
CLASS_COLORS = {
    0: (230, 25, 75),    # Urban - red
    1: (60, 180, 75),    # Forest - green
    2: (0, 130, 200),    # Water - blue
    3: (255, 225, 25),   # Agriculture - yellow
    4: (170, 110, 40),   # Bare land - brown
}


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad in case of odd input dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net. Input: (B, in_channels, H, W). Output logits: (B, num_classes, H, W)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5, base_ch: int = 64, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch * 8, base_ch * 16 // factor)

        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear)
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear)
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear)
        self.up4 = Up(base_ch * 2, base_ch, bilinear)
        self.outc = OutConv(base_ch, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    # quick shape sanity check
    net = UNet(in_channels=3, num_classes=5)
    dummy = torch.randn(2, 3, 256, 256)
    out = net(dummy)
    print("Output shape:", out.shape)  # expect (2, 5, 256, 256)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params:,}")
