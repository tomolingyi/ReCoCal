# -*- coding: utf-8 -*-
"""F3-assisted Progressive Decoder for pixel-level forgery localization.

The decoder consumes the calibrated RLGC feature from F4 and a lateral F3 skip
used only for spatial-detail recovery. It preserves a simple-decoder base logit
and learns a progressive residual refinement initialized to zero.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LayerNorm2d


class ResidualRefineBlock(nn.Module):
    """Lightweight residual refinement block with LayerNorm2d."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = LayerNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return residual + x


class F3AssistedProgressiveDecoder(nn.Module):
    """Progressive decoder with an F3 lateral skip used only for spatial recovery.

    Inputs
    ------
    reason_feature : [B, C_r, H/32, W/32]
        Calibrated RLGC feature.
    f3_feature : [B, C_3, H/16, W/16]
        MiT-B2 stage-3 feature used only to restore spatial structure.

    The module starts from a simple-decoder base logit on the RLGC feature and
    adds a learned progressive correction initialized to zero. This makes the
    initial behaviour close to the Simple Decoder while allowing gradual
    refinement with F3 details.
    """
    def __init__(
        self,
        reason_channels: int,
        f3_channels: int,
        decoder_dim: int = 160,
        dropout_p: float = 0.10,
    ) -> None:
        super().__init__()
        decoder_dim = int(decoder_dim)
        self.base_head = nn.Conv2d(int(reason_channels), 1, kernel_size=1, bias=True)

        self.reason_projection = nn.Sequential(
            nn.Conv2d(int(reason_channels), decoder_dim, kernel_size=1, bias=False),
            LayerNorm2d(decoder_dim),
        )
        self.f3_projection = nn.Sequential(
            nn.Conv2d(int(f3_channels), decoder_dim, kernel_size=1, bias=False),
            LayerNorm2d(decoder_dim),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.stage16 = nn.Sequential(
            ResidualRefineBlock(decoder_dim),
            nn.Dropout2d(float(dropout_p)),
        )
        self.stage8 = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(decoder_dim),
            nn.GELU(),
            ResidualRefineBlock(decoder_dim),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(decoder_dim),
            nn.GELU(),
            ResidualRefineBlock(decoder_dim),
        )
        self.correction_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, 1, kernel_size=1, bias=True),
        )
        # Start from the simple-decoder behaviour and let the progressive branch
        # learn only a residual correction.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, reason_feature: torch.Tensor, f3_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if reason_feature.ndim != 4 or f3_feature.ndim != 4:
            raise RuntimeError('F3AssistedProgressiveDecoder expects BCHW features')
        base_logit = self.base_head(reason_feature)
        x = F.interpolate(reason_feature, size=f3_feature.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reason_projection(x) + self.f3_projection(f3_feature)
        x = self.fuse(x)
        x = self.stage16(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.stage8(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.stage4(x)
        correction = self.correction_head(x)
        base_native = F.interpolate(base_logit, size=correction.shape[-2:], mode='bilinear', align_corners=False)
        native_logit = base_native + correction
        return x, native_logit
