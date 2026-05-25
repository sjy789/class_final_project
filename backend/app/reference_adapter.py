from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .image_utils import compose_image


REFERENCE_NOTES = {
    "OPA": (
        "bcmi/Object-Placement-Assessment-Dataset-OPA: SimOPA uses a "
        "composite RGB image plus a foreground mask as a 4-channel input "
        "and predicts whether object placement is reasonable."
    ),
    "TopNet": (
        "bcmi/TopNet-Object-Placement: object placement is organized as "
        "multi-location and multi-scale scoring, then high-score locations "
        "are selected as generated candidates."
    ),
    "libcom": (
        "bcmi/libcom: image composition pipeline integrates paste/blending, "
        "OPA-style placement scoring, FOPA heatmaps and color transfer tools."
    ),
}


@dataclass(frozen=True)
class OPAStyleSignals:
    opa_rgb_mask_score: float
    opa_mask_area: float
    opa_mask_bottom: float
    opa_mask_center_x: float


def build_opa_style_input(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
    input_size: int = 128,
) -> torch.Tensor:
    """Build the RGB+mask tensor used by the OPA/SimOPA reference interface.

    OPA's dataset/model code concatenates the composite RGB image and a
    foreground mask into a 4-channel tensor. This project keeps that same
    data contract, while resizing to 128 for faster CPU interaction.
    """

    x = max(0, min(int(round(x)), max(0, background.width - foreground.width)))
    y = max(0, min(int(round(y)), max(0, background.height - foreground.height)))
    composite = compose_image(background, foreground, x, y, harmonize=False).convert("RGB")
    mask = Image.new("L", background.size, 0)
    mask.paste(foreground.getchannel("A"), (x, y))

    composite = composite.resize((input_size, input_size), Image.Resampling.BILINEAR)
    mask = mask.resize((input_size, input_size), Image.Resampling.BILINEAR)
    rgb_np = np.asarray(composite).astype(np.float32) / 255.0
    mask_np = np.asarray(mask).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)
    mask_channel = torch.from_numpy(mask_np).unsqueeze(0)
    return torch.cat([rgb, mask_channel], dim=0)


def opa_style_reference_forward(img_cat: torch.Tensor) -> OPAStyleSignals:
    """A lightweight local adapter inspired by SimOPA's 4-channel forward pass.

    This is not the pretrained SimOPA checkpoint. It is a CPU-safe adaptation
    of the reference interface that extracts placement signals from the same
    RGB+mask tensor shape used by OPA: Bx4xHxW or 4xHxW.
    """

    if img_cat.ndim == 3:
        img_cat = img_cat.unsqueeze(0)
    if img_cat.shape[1] != 4:
        raise ValueError(f"Expected RGB+mask tensor with 4 channels, got {img_cat.shape}")

    rgb = img_cat[:, :3].float()
    mask = img_cat[:, 3:4].float().clamp(0.0, 1.0)
    eps = torch.tensor(1e-6, dtype=rgb.dtype, device=rgb.device)
    b, _, h, w = img_cat.shape

    ys = torch.linspace(0.0, 1.0, h, dtype=rgb.dtype, device=rgb.device).view(1, 1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, dtype=rgb.dtype, device=rgb.device).view(1, 1, 1, w)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp_min(float(eps))

    area = mask.mean(dim=(2, 3)).view(b)
    center_x = (mask * xs).sum(dim=(2, 3), keepdim=True).div(mask_sum).view(b)
    center_y = (mask * ys).sum(dim=(2, 3), keepdim=True).div(mask_sum).view(b)
    height_est = area.sqrt().clamp_min(0.02)
    bottom = (center_y + height_est * 0.55).clamp(0.0, 1.0)

    luma_weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    luma = (rgb * luma_weights).sum(dim=1, keepdim=True)
    fg_luma = (luma * mask).sum(dim=(2, 3), keepdim=True).div(mask_sum)
    bg_weight = (1.0 - mask).sum(dim=(2, 3), keepdim=True).clamp_min(float(eps))
    bg_luma = (luma * (1.0 - mask)).sum(dim=(2, 3), keepdim=True).div(bg_weight)
    brightness_match = (1.0 - torch.abs(fg_luma - bg_luma).view(b) / 0.55).clamp(0.0, 1.0)

    bottom_prior = (1.0 - torch.abs(bottom - 0.78) / 0.35).clamp(0.0, 1.0)
    center_prior = (1.0 - torch.abs(center_x - 0.5) / 0.58).clamp(0.0, 1.0)
    size_prior = torch.where(
        area < 0.015,
        area / 0.015 * 0.55,
        1.0 - torch.abs(area - 0.08) / 0.18,
    ).clamp(0.0, 1.0)

    score = (0.34 * bottom_prior + 0.22 * center_prior + 0.24 * size_prior + 0.20 * brightness_match).clamp(0.0, 1.0)
    return OPAStyleSignals(
        opa_rgb_mask_score=round(float(score.mean().item()), 4),
        opa_mask_area=round(float(area.mean().item()), 4),
        opa_mask_bottom=round(float(bottom.mean().item()), 4),
        opa_mask_center_x=round(float(center_x.mean().item()), 4),
    )
