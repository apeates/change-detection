"""Small U-Net variants for pairs of RGB images."""

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


class _ChangeUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoders = nn.ModuleList([conv_block(in_channels, 8), conv_block(8, 16), conv_block(16, 32)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 64)
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
        ])
        self.decoders = nn.ModuleList([conv_block(64, 32), conv_block(32, 16), conv_block(16, 8)])
        self.head = nn.Conv2d(8, 1, kernel_size=1)

    def encode(self, features):
        skips = []
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)
        return self.bottleneck(features), skips

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 6 or any(size % 8 or size == 0 for size in inputs.shape[-2:]):
            raise ValueError("Expected (N, 6, H, W) with positive H and W divisible by 8.")

        features, skips = self.encode_pair(inputs)
        for upsample, decode, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            features = decode(torch.cat((upsample(features), skip), dim=1))
        return self.head(features)


class EarlyFusionUNet(_ChangeUNet):
    """Encode the six input channels jointly."""

    architecture = "early_fusion_unet"

    def __init__(self):
        super().__init__(in_channels=6)

    def encode_pair(self, inputs):
        return self.encode(inputs)


class SiameseUNet(_ChangeUNet):
    """Decode absolute differences between two shared-weight encodings."""

    architecture = "siamese_unet"

    def __init__(self):
        super().__init__(in_channels=3)

    def encode_pair(self, inputs):
        before, after = inputs.split(3, dim=1)
        before_features, before_skips = self.encode(before)
        after_features, after_skips = self.encode(after)
        skips = [(first - second).abs() for first, second in zip(before_skips, after_skips, strict=True)]
        return (before_features - after_features).abs(), skips


UNET_MODELS = {model.architecture: model for model in (EarlyFusionUNet, SiameseUNet)}
