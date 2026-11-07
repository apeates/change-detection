"""A small U-Net for six-channel pairs of RGB images."""

import torch
from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
    )


class EarlyFusionUNet(nn.Module):
    """Map (N, 6, H, W) RGB pairs to (N, 1, H, W) change logits."""

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([conv_block(6, 8), conv_block(8, 16), conv_block(16, 32)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 64)
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
        ])
        self.decoders = nn.ModuleList([conv_block(64, 32), conv_block(32, 16), conv_block(16, 8)])
        self.head = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 6 or any(size % 8 or size == 0 for size in inputs.shape[-2:]):
            raise ValueError("Expected (N, 6, H, W) with positive H and W divisible by 8.")

        skips = []
        features = inputs
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)

        features = self.bottleneck(features)
        for upsample, decode, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            features = decode(torch.cat((upsample(features), skip), dim=1))
        return self.head(features)
