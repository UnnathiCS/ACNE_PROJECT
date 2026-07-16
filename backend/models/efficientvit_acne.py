"""
EfficientViT-based acne lesion classifier with multi-scale feature fusion and CBAM.

Architecture modifications (backbone remains pretrained EfficientViT from timm):
  1. Extract intermediate + final stage feature maps from the EfficientViT stages.
  2. Resize intermediate features to match the final stage spatial resolution.
  3. Concatenate aligned maps and compress channels with a 1x1 convolution.
  4. Apply lightweight CBAM (channel + spatial attention) on fused conv features.
  5. Global average pool -> dropout -> linear head for 5 acne classes.

The last convolutional feature tensor (post-CBAM) is exposed for Grad-CAM.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover - timm is required at runtime
    timm = None


# ---------------------------------------------------------------------------
# Lightweight CBAM (Convolutional Block Attention Module)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze channel descriptors and recalibrate per-channel responses."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        # Shared MLP for avg- and max-pooled channel descriptors.
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_desc = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_desc = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg_desc) + self.mlp(max_desc))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Highlight lesion-relevant spatial locations using pooled channel stats."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention (lightweight, no extra downsampling)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------------
# Multi-scale EfficientViT classifier
# ---------------------------------------------------------------------------

class EfficientViTAcneClassifier(nn.Module):
    """
    EfficientViT backbone + multi-scale fusion head for 5-class acne classification.

    Grad-CAM compatibility:
      - `self.gradcam_features` stores the final conv feature map (post-CBAM).
      - `forward(..., return_feature_maps=True)` returns logits and a feature dict.
    """

    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = "efficientvit_b0",
        pretrained: bool = True,
        fusion_channels: int = 128,
        dropout: float = 0.4,
        intermediate_stage_idx: Optional[int] = None,
        final_stage_idx: Optional[int] = None,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required. Install with `pip install timm`.")

        if not (0.3 <= dropout <= 0.5):
            raise ValueError("dropout must be between 0.3 and 0.5 for this architecture.")

        # Load pretrained EfficientViT; stem + stages weights are transferred below.
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Modification 1: keep only the convolutional EfficientViT encoder (no timm head).
        self.stem = backbone.stem
        self.stages = backbone.stages
        self.model_name = model_name
        self.num_classes = num_classes

        num_stages = len(self.stages)
        # Modification 2: pick an intermediate stage and the final stage for multi-scale fusion.
        self.intermediate_stage_idx = (
            intermediate_stage_idx if intermediate_stage_idx is not None else num_stages - 2
        )
        self.final_stage_idx = (
            final_stage_idx if final_stage_idx is not None else num_stages - 1
        )

        if not (0 <= self.intermediate_stage_idx < self.final_stage_idx < num_stages):
            raise ValueError(
                f"Invalid stage indices: intermediate={self.intermediate_stage_idx}, "
                f"final={self.final_stage_idx}, num_stages={num_stages}"
            )

        # Channel widths are read from timm feature metadata for correct 1x1 fusion sizing.
        feature_info = getattr(backbone, "feature_info", None) or []
        if feature_info:
            self.intermediate_channels = feature_info[self.intermediate_stage_idx]["num_chs"]
            self.final_channels = feature_info[self.final_stage_idx]["num_chs"]
        else:
            # Fallback for unexpected timm variants.
            self.intermediate_channels = backbone.num_features // 2
            self.final_channels = backbone.num_features

        fused_in_channels = self.intermediate_channels + self.final_channels

        # Modification 3: 1x1 conv fuses concatenated multi-scale features into a compact tensor.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_in_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.Hardswish(inplace=True),
        )

        # Modification 4: CBAM refines fused features for fine-grained lesion discrimination.
        self.cbam = CBAM(fusion_channels, reduction=max(fusion_channels // 16, 8))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.classifier = nn.Linear(fusion_channels, num_classes)

        # Populated on every forward pass; used by Grad-CAM visualizations.
        self.gradcam_features: Optional[torch.Tensor] = None

        del backbone

    def _forward_backbone_stages(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run stem + stages and capture intermediate/final spatial feature maps."""
        x = self.stem(x)
        feat_intermediate = None
        feat_final = None

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx == self.intermediate_stage_idx:
                feat_intermediate = x
            if stage_idx == self.final_stage_idx:
                feat_final = x

        if feat_intermediate is None or feat_final is None:
            raise RuntimeError("Failed to extract required EfficientViT stage features.")

        return feat_intermediate, feat_final

    def _fuse_multiscale(
        self,
        feat_intermediate: torch.Tensor,
        feat_final: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modification 5: align spatial resolutions, concatenate, then compress channels.

        The intermediate map is resized (upsample or downsample) to match the final stage,
        which keeps the deepest semantics and a compact spatial grid for efficiency.
        """
        target_h, target_w = feat_final.shape[-2:]
        if feat_intermediate.shape[-2:] != (target_h, target_w):
            feat_intermediate = F.interpolate(
                feat_intermediate,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.cat([feat_intermediate, feat_final], dim=1)
        return self.fusion_conv(fused)

    def forward(
        self,
        x: torch.Tensor,
        return_feature_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feat_intermediate, feat_final = self._forward_backbone_stages(x)
        fused = self._fuse_multiscale(feat_intermediate, feat_final)
        attended = self.cbam(fused)

        # Last conv feature map before pooling — primary Grad-CAM target.
        self.gradcam_features = attended

        pooled = self.pool(attended).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_feature_maps:
            return logits, {
                "feat_fused": attended,
                "feat_intermediate": feat_intermediate,
                "feat_final": feat_final,
            }
        return logits

    def get_classifier(self) -> nn.Module:
        """Expose the final linear layer for phased head-only fine-tuning."""
        return self.classifier


def build_efficientvit_acne_classifier(
    num_classes: int = 5,
    model_name: str = "efficientvit_b0",
    pretrained: bool = True,
    fusion_channels: int = 128,
    dropout: float = 0.4,
) -> EfficientViTAcneClassifier:
    """Factory used by training and inference entry points."""
    return EfficientViTAcneClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        fusion_channels=fusion_channels,
        dropout=dropout,
    )


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> None:
    """Print a concise architecture summary and parameter count at startup."""
    device = next(model.parameters()).device
    dummy = torch.zeros(input_size, device=device)

    model.eval()
    with torch.no_grad():
        if hasattr(model, "forward") and "return_feature_maps" in model.forward.__code__.co_varnames:
            out = model(dummy, return_feature_maps=True)
            logits, feat_maps = out
            feat_shape = tuple(feat_maps["feat_fused"].shape)
        else:
            logits = model(dummy)
            feat_shape = None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n=== EfficientViT Acne Classifier (multi-scale + CBAM) ===")
    print(f"Model class: {model.__class__.__name__}")
    if hasattr(model, "model_name"):
        print(f"Backbone: {model.model_name}")
    if hasattr(model, "intermediate_stage_idx"):
        print(
            "Multi-scale stages: "
            f"intermediate=stage.{model.intermediate_stage_idx} "
            f"({getattr(model, 'intermediate_channels', '?')} ch), "
            f"final=stage.{model.final_stage_idx} "
            f"({getattr(model, 'final_channels', '?')} ch)"
        )
    print("Fusion: concat -> 1x1 conv -> CBAM -> GAP -> dropout -> linear")
    if feat_shape is not None:
        print(f"Grad-CAM feature map shape (post-CBAM): {feat_shape}")
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print("========================================================\n")
