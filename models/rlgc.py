# -*- coding: utf-8 -*-
"""Reliability-Aware Local-Global Calibration (RLGC).

Local REC calibrates contextual verification evidence using reference dispersion
and consolidates the correction into a single feature. Global calibration then
uses frozen CLIP CLS12/24 descriptors for bounded channel-wise modulation.
RLGC returns one feature to the decoder; no evidence bypass leaves this module.
"""
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn


class ReliabilityAwareEvidenceCalibration(nn.Module):
    """Local REC: reliability-aware evidence calibration and feature consolidation."""
    def __init__(self, dim: int = 160) -> None:
        super().__init__()
        if dim < 8 or dim % 8:
            raise ValueError("dim must be >=8 and divisible by 8")
        self.consistency_gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, 1, 1),
        )
        self.correction_net = nn.Sequential(
            nn.Conv2d(3 * dim, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        # Identity at initialization: the added REC stage starts from the
        # already-validated SCDV representation.
        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)
        self.feature_adapter = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.dirac_(self.feature_adapter.weight)

    def forward(self, feature: torch.Tensor, evidence: torch.Tensor,
                reference_variance: torch.Tensor) -> Dict[str, torch.Tensor]:
        if feature.shape != evidence.shape or feature.shape != reference_variance.shape:
            raise ValueError("feature, evidence, and reference variance must share [B,C,H,W]")
        if not torch.isfinite(reference_variance).all():
            raise FloatingPointError("Reference variance contains non-finite values")
        log_var = torch.log1p(reference_variance.float().clamp_min(0)).to(evidence.dtype)
        reliability = torch.sigmoid(self.consistency_gate(torch.cat((evidence, log_var), dim=1)))
        evidence_delta = reliability * self.correction_net(
            torch.cat((feature, evidence, log_var), dim=1)
        )
        calibrated_evidence = evidence + evidence_delta
        rec_feature = feature + self.feature_adapter(evidence_delta)
        return {
            "rec_feature": rec_feature,
            "calibrated_evidence": calibrated_evidence,  # diagnostic/internal only
            "reference_reliability_gate": reliability,
            "evidence_correction": evidence_delta,
        }


class GlobalSemanticCalibration(nn.Module):
    def __init__(self, clip_dim: int, feature_dim: int,
                 selected_layers: Sequence[int] = (12, 24), hidden_dim: int = 512,
                 dropout_p: float = 0.10, scale_limit: float = 0.25,
                 bias_limit: float = 0.25) -> None:
        super().__init__()
        self.selected_layers = tuple(int(v) for v in selected_layers)
        if len(self.selected_layers) != 2 or self.selected_layers[0] == self.selected_layers[1]:
            raise ValueError("Global calibration needs two distinct CLIP layers")
        self.scale_limit = float(scale_limit)
        self.bias_limit = float(bias_limit)
        input_dim = 3 * int(clip_dim)
        self.cls_norms = nn.ModuleDict({str(layer): nn.LayerNorm(int(clip_dim)) for layer in self.selected_layers})
        self.global_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout_p)),
            nn.Linear(int(hidden_dim), 2 * int(feature_dim)),
        )
        nn.init.zeros_(self.global_adapter[-1].weight)
        nn.init.zeros_(self.global_adapter[-1].bias)

    def forward(self, feature: torch.Tensor,
                clip_features: Dict[str, Dict[int, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not isinstance(clip_features, dict) or not isinstance(clip_features.get("cls_features"), dict):
            raise KeyError("Global calibration requires cls_features")
        cls_features = clip_features["cls_features"]
        l1, l2 = self.selected_layers
        if l1 not in cls_features or l2 not in cls_features:
            raise KeyError("Missing required CLIP CLS layers")
        c1 = self.cls_norms[str(l1)](cls_features[l1])
        c2 = self.cls_norms[str(l2)](cls_features[l2])
        descriptor = torch.cat((c1, c2, torch.abs(c1 - c2)), dim=1)
        scale, bias = self.global_adapter(descriptor).chunk(2, dim=1)
        scale = (self.scale_limit * torch.tanh(scale)).unsqueeze(-1).unsqueeze(-1)
        bias = (self.bias_limit * torch.tanh(bias)).unsqueeze(-1).unsqueeze(-1)
        calibrated = feature * (1.0 + scale) + bias
        return {
            "global_feature": calibrated,
            "global_scale_mean": scale.abs().mean(dim=(1, 2, 3)),
            "global_bias_mean": bias.abs().mean(dim=(1, 2, 3)),
        }


class ReliabilityAwareLocalGlobalCalibration(nn.Module):
    """Complete RLGC = local REC + Global Semantic Calibration."""
    def __init__(self, clip_dim: int, feature_dim: int,
                 selected_layers: Sequence[int] = (12, 24), global_hidden_dim: int = 512,
                 dropout_p: float = 0.10, global_scale_limit: float = 0.25,
                 global_bias_limit: float = 0.25) -> None:
        super().__init__()
        self.rec = ReliabilityAwareEvidenceCalibration(dim=feature_dim)
        self.global_calibration = GlobalSemanticCalibration(
            clip_dim=clip_dim, feature_dim=feature_dim, selected_layers=selected_layers,
            hidden_dim=global_hidden_dim, dropout_p=dropout_p,
            scale_limit=global_scale_limit, bias_limit=global_bias_limit,
        )

    def forward(self, feature: torch.Tensor, evidence: torch.Tensor,
                reference_variance: torch.Tensor,
                clip_features: Dict[str, Dict[int, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        local = self.rec(feature, evidence, reference_variance)
        global_out = self.global_calibration(local["rec_feature"], clip_features)
        outputs = {
            "calibrated_feature": global_out["global_feature"],
            "rec_feature": local["rec_feature"],
            "reference_reliability_gate": local["reference_reliability_gate"],
            "global_scale_mean": global_out["global_scale_mean"],
            "global_bias_mean": global_out["global_bias_mean"],
        }
        return outputs
