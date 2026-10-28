# -*- coding: utf-8 -*-
"""Semantic Contextual Discrepancy Verification (SCDV).

Serial-design version used by the final ablation family.
SCDV never writes a mask logit and has no post-decoder branch. It produces a
verified forensic representation plus contextual evidence for downstream RLGC.
"""
from __future__ import annotations

from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ContextualDiscrepancyStage(nn.Module):
    """F4 projection + contextual discrepancy representation construction."""
    def __init__(self, in_channels: int = 512, dim: int = 160,
                 topk: int = 8, temperature: float = 0.10,
                 reference_seed: int = 42, gamma_init: float = 0.10) -> None:
        super().__init__()
        if topk < 1 or temperature <= 0:
            raise ValueError("topk >= 1 and temperature > 0 are required")
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.reference_seed = int(reference_seed)
        self.dim = int(dim)
        self.task_projection = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.context_predictor = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.discrepancy_encoder = nn.Sequential(
            nn.Conv2d(2 * dim + 1, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.update_gate = nn.Conv2d(2 * dim, dim, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self._indices_cache: Dict[tuple, torch.Tensor] = {}

    def _reference_indices(self, n: int, device: torch.device) -> torch.Tensor:
        if n <= self.topk:
            raise ValueError(f"F4 has {n} positions; need n > topk={self.topk}")
        key = (n, str(device))
        if key not in self._indices_cache:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.reference_seed + n)
            scores = torch.rand(n, n, generator=gen)
            scores.fill_diagonal_(-1.0)
            ids = torch.topk(scores, self.topk, dim=-1, sorted=False).indices
            self._indices_cache[key] = ids.to(device=device, dtype=torch.long)
        return self._indices_cache[key]

    def forward(self, f4: torch.Tensor, clip_patch: torch.Tensor) -> Dict[str, torch.Tensor]:
        if clip_patch is None:
            raise ValueError("CLIP layer-24 patch features are required by CDM")
        z = self.task_projection(f4)
        b, c, h, w = z.shape
        n = h * w
        indices = self._reference_indices(n, z.device).unsqueeze(0).expand(b, -1, -1)
        k = indices.shape[-1]
        patch = F.interpolate(clip_patch.detach().float(), (h, w), mode="bilinear", align_corners=False)
        semantic = F.normalize(patch.flatten(2), dim=1).transpose(1, 2)
        affinity = torch.bmm(semantic, semantic.transpose(1, 2))
        semantic_logits = affinity.gather(dim=2, index=indices) / self.temperature
        z_flat = z.flatten(2).transpose(1, 2)
        refs = z_flat.gather(
            1, indices.reshape(b, -1).unsqueeze(-1).expand(-1, -1, c)
        ).reshape(b, n, k, c)
        weights = semantic_logits.softmax(dim=-1)
        reference = (weights.to(refs.dtype).unsqueeze(-1) * refs).sum(dim=2)
        reference = reference.transpose(1, 2).reshape(b, c, h, w)
        prediction = self.context_predictor(reference)
        delta = z - prediction
        zero_channel = z.new_zeros((b, 1, h, w))
        discrepancy = self.discrepancy_encoder(torch.cat((delta, delta.abs(), zero_channel), dim=1))
        gate = torch.sigmoid(self.update_gate(torch.cat((z, discrepancy), dim=1)))
        contextual_feature = z + self.gamma * gate * discrepancy
        return {
            "projected_feature": z,
            "context_prediction": prediction,
            "contextual_discrepancy": discrepancy,
            "contextual_feature": contextual_feature,
            "reference_indices": indices,
            "reference_weights": weights,
        }


class _ConditionalContextPredictor(nn.Module):
    """Context predictor for FV.

    ``use_semantic_conditioning=False`` is the FV ablation: the same non-local
    support is used, but all eligible references receive uniform weights and the
    added CLIP12/24 condition is exactly zero. CDM still keeps its original
    layer-24 CLIP weighting.
    """
    def __init__(self, feature_dim: int = 160, clip_dim: int = 1024,
                 temperature: float = 0.10, exclusion_radius: int = 1,
                 use_semantic_conditioning: bool = True) -> None:
        super().__init__()
        if temperature <= 0 or exclusion_radius < 0:
            raise ValueError("temperature must be positive and exclusion_radius nonnegative")
        if feature_dim % 8:
            raise ValueError("feature_dim must be divisible by 8")
        self.temperature = float(temperature)
        self.exclusion_radius = int(exclusion_radius)
        self.use_semantic_conditioning = bool(use_semantic_conditioning)
        self.feature_dim = int(feature_dim)
        if self.use_semantic_conditioning:
            self.semantic_mix_logit = nn.Parameter(torch.tensor(0.0))
            self.local_condition = nn.Conv2d(clip_dim, feature_dim, 1, bias=False)
            self.deep_condition = nn.Conv2d(clip_dim, feature_dim, 1, bias=False)
        self.context_predictor = nn.Sequential(
            nn.Conv2d(3 * feature_dim, feature_dim, 1, bias=False),
            nn.GroupNorm(8, feature_dim), nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim, bias=False),
            nn.GroupNorm(8, feature_dim), nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        self._valid_cache: Dict[tuple, torch.Tensor] = {}

    def _allowed_references(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device), self.exclusion_radius)
        if key not in self._valid_cache:
            yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            pos = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
            diff = (pos[:, None, :] - pos[None, :, :]).abs()
            exclude = ((diff[..., 0] <= self.exclusion_radius) &
                       (diff[..., 1] <= self.exclusion_radius))
            valid = ~exclude
            if not valid.any(dim=1).all():
                raise ValueError(f"No eligible context for F4 grid={h}x{w}; reduce exclusion_radius")
            self._valid_cache[key] = valid
        return self._valid_cache[key]

    def forward(self, z: torch.Tensor, clip_mid: torch.Tensor | None,
                clip_deep: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        b, c, h, w = z.shape
        valid = self._allowed_references(h, w, z.device)
        values = z.detach().flatten(2).transpose(1, 2)

        if self.use_semantic_conditioning:
            if clip_mid is None or clip_deep is None:
                raise ValueError("Semantic Conditioning requires frozen CLIP layer-12/24 patch maps")
            mid = F.interpolate(clip_mid.detach().float(), (h, w), mode="bilinear", align_corners=False)
            deep = F.interpolate(clip_deep.detach().float(), (h, w), mode="bilinear", align_corners=False)
            if (mid.shape[1] != self.local_condition.in_channels or
                    deep.shape[1] != self.deep_condition.in_channels):
                raise ValueError("CLIP feature channels mismatch: check clip_feature_dim")
            mid_norm = F.normalize(mid.flatten(2), dim=1).transpose(1, 2)
            deep_norm = F.normalize(deep.flatten(2), dim=1).transpose(1, 2)
            with torch.autocast(device_type=z.device.type, enabled=False):
                sim_mid = torch.bmm(mid_norm, mid_norm.transpose(1, 2))
                sim_deep = torch.bmm(deep_norm, deep_norm.transpose(1, 2))
                mix = self.semantic_mix_logit.sigmoid()
                logits = ((1.0 - mix) * sim_mid + mix * sim_deep) / self.temperature
                weights = logits.masked_fill(
                    ~valid.unsqueeze(0), torch.finfo(logits.dtype).min
                ).softmax(-1)
            condition = self.local_condition(mid.to(z.dtype)) + self.deep_condition(deep.to(z.dtype))
        else:
            uniform = valid.to(dtype=z.dtype)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True)
            weights = uniform.unsqueeze(0).expand(b, -1, -1)
            mix = z.new_tensor(0.0)
            condition = z.new_zeros((b, c, h, w))

        context = torch.bmm(weights.to(values.dtype), values)
        with torch.autocast(device_type=z.device.type, enabled=False):
            w32 = weights.float()
            v32 = values.float()
            mean32 = torch.bmm(w32, v32)
            second32 = torch.bmm(w32, v32.square())
            variance = (second32 - mean32.square()).clamp_min_(0.0)
        reference_variance = variance.transpose(1, 2).reshape(b, c, h, w).to(z.dtype)
        context = context.transpose(1, 2).reshape(b, c, h, w)
        prediction = self.context_predictor(
            torch.cat((context, condition, context * condition.sigmoid()), dim=1)
        )
        return {
            "semantic_context_prediction": prediction,
            "semantic_condition": condition,
            "reference_context": context,
            "reference_variance": reference_variance,
            "semantic_mix": mix,
        }


class _FeatureVerifier(nn.Module):
    """Convert context mismatch into a verified *feature*, never a mask bypass."""
    def __init__(self, dim: int = 160) -> None:
        super().__init__()
        self.evidence_encoder = nn.Sequential(
            nn.Conv2d(3 * dim, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
        )
        self.verification_gate = nn.Conv2d(2 * dim, 1, 1)
        nn.init.zeros_(self.verification_gate.weight)
        nn.init.zeros_(self.verification_gate.bias)
        self.verification_gain_logit = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))

    def forward(self, z: torch.Tensor, prediction: torch.Tensor,
                condition: torch.Tensor, contextual_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        delta = z - prediction
        evidence = self.evidence_encoder(torch.cat((delta, delta.abs(), condition), dim=1))
        reliability = torch.sigmoid(self.verification_gate(torch.cat((contextual_feature, evidence), dim=1)))
        gain = self.verification_gain_logit.sigmoid()
        verified_feature = contextual_feature + gain * reliability * evidence
        return {
            "verification_evidence": evidence,
            "verification_reliability": reliability,
            "verification_gain": gain,
            "verified_feature": verified_feature,
        }


class SemanticContextualDiscrepancyVerification(nn.Module):
    """CDM + FV, optionally with the final CLIP Semantic Conditioning path."""
    def __init__(self, in_channels: int, feature_dim: int, clip_dim: int,
                 topk: int, reference_temperature: float, reference_seed: int,
                 semantic_temperature: float, exclusion_radius: int,
                 use_semantic_conditioning: bool = True) -> None:
        super().__init__()
        self.use_semantic_conditioning = bool(use_semantic_conditioning)
        self.discrepancy = _ContextualDiscrepancyStage(
            in_channels=in_channels, dim=feature_dim, topk=topk,
            temperature=reference_temperature, reference_seed=reference_seed,
        )
        self.context = _ConditionalContextPredictor(
            feature_dim=feature_dim, clip_dim=clip_dim,
            temperature=semantic_temperature, exclusion_radius=exclusion_radius,
            use_semantic_conditioning=self.use_semantic_conditioning,
        )
        self.verifier = _FeatureVerifier(dim=feature_dim)

    def forward(self, f4: torch.Tensor, reference_patch: torch.Tensor,
                clip_mid: torch.Tensor | None, clip_deep: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        outputs = self.discrepancy(f4, reference_patch)
        context = self.context(outputs["projected_feature"], clip_mid, clip_deep)
        verification = self.verifier(
            outputs["projected_feature"],
            context["semantic_context_prediction"],
            context["semantic_condition"],
            outputs["contextual_feature"],
        )
        outputs.update(context)
        outputs.update(verification)
        return outputs
