"""BIT with a binary head for the LEVIR-CD trainer.

Architecture reference: Chen, Qi and Shi, https://arxiv.org/abs/2103.00208.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1, dilation=1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(outgoing),
            nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(outgoing),
        )
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False), nn.BatchNorm2d(outgoing)
        )

    def forward(self, image):
        return F.relu(self.convs(image) + self.skip(image))


class AttentionBlock(nn.Module):
    def __init__(self, head_width):
        super().__init__()
        self.heads = 8
        width = self.heads * head_width
        self.norm = nn.LayerNorm(32)
        self.query = nn.Linear(32, width, bias=False)
        self.key_value = nn.Linear(32, 2 * width, bias=False)
        self.project = nn.Linear(width, 32)
        self.feed_forward = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, features, memory=None):
        normalized = self.norm(features)
        memory = normalized if memory is None else self.norm(memory)
        key, value = self.key_value(memory).chunk(2, dim=-1)
        query, key, value = (tensor.unflatten(-1, (self.heads, -1)).transpose(1, 2)
                             for tensor in (self.query(normalized), key, value))
        attended = F.scaled_dot_product_attention(query, key, value, scale=32 ** -0.5)
        features = features + self.project(attended.transpose(1, 2).flatten(2))
        return features + self.feed_forward(features)


class BIT(nn.Module):
    """Tokenize each date, mix both token sequences, then decode pixel features."""

    architecture = "bit"
    variant = "BIT_S4_dd8_dedim8"

    def __init__(self):
        super().__init__()
        # ResNet-18 through layer3. Dilation retains stride 8 in layer3.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(64, 64), ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128),
            ResidualBlock(128, 256), ResidualBlock(256, 256, dilation=2),
        )
        self.features = nn.Conv2d(256, 32, 3, padding=1)
        self.token_weights = nn.Conv2d(32, 4, 1, bias=False)
        self.positions = nn.Parameter(torch.randn(1, 8, 32))
        self.encoder = AttentionBlock(head_width=64)
        self.decoder = nn.ModuleList(AttentionBlock(head_width=8) for _ in range(8))
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(), nn.Conv2d(32, 1, 3, padding=1),
        )

    def encode_date(self, image):
        features = self.features(F.interpolate(self.backbone(image), scale_factor=2, mode="nearest"))
        weights = self.token_weights(features).flatten(2).softmax(dim=-1)
        pixels = features.flatten(2).transpose(1, 2)
        return pixels, weights @ pixels, features.shape[-2:]

    def decode_date(self, pixels, tokens):
        for layer in self.decoder:
            pixels = layer(pixels, tokens)
        return pixels

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 6 or any(size == 0 or size % 8 for size in inputs.shape[-2:]):
            raise ValueError("BIT expects (N, 6, H, W) with positive spatial dimensions divisible by 8.")
        before, after = inputs.chunk(2, dim=1)
        first, first_tokens, shape = self.encode_date(before)
        second, second_tokens, _ = self.encode_date(after)
        tokens = self.encoder(torch.cat((first_tokens, second_tokens), dim=1) + self.positions)
        first_tokens, second_tokens = tokens.chunk(2, dim=1)
        change = (self.decode_date(first, first_tokens) - self.decode_date(second, second_tokens)).abs()
        change = change.transpose(1, 2).unflatten(2, shape)
        change = F.interpolate(change, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(change)
