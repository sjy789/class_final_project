from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from .image_utils import clamp
from .reference_adapter import build_opa_style_input, opa_style_reference_forward
from .trained_scorer import score_with_trained_opa


FEATURE_ORDER = [
    "opa_rgb_mask_score",
    "margin",
    "lower_half",
    "center_balance",
    "size_reasonable",
    "background_clean",
    "brightness_match",
    "color_match",
    "support",
    "mask_quality",
]


@dataclass
class PlacementCandidate:
    id: str
    x: int
    y: int
    w: int
    h: int
    nx: float
    ny: float
    nw: float
    nh: float
    score: float
    label: str
    reason: str
    features: Dict[str, float]
    preview: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class PlacementAssessmentModel(nn.Module):
    """A small CPU-friendly scoring head for RGB + mask + candidate features."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(len(FEATURE_ORDER), 1)
        weights = torch.tensor(
            [[0.92, 0.92, 0.78, 0.54, 0.94, 0.78, 0.64, 0.48, 0.82, 0.42]],
            dtype=torch.float32,
        )
        bias = torch.tensor([-4.15], dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.copy_(weights)
            self.head.bias.copy_(bias)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(features))


MODEL = PlacementAssessmentModel().eval()


def _safe_patch(background: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    x = max(0, min(x, background.width - 1))
    y = max(0, min(y, background.height - 1))
    right = max(x + 1, min(x + w, background.width))
    bottom = max(y + 1, min(y + h, background.height))
    return background.crop((x, y, right, bottom)).convert("RGB")


def _edge_density(patch: Image.Image) -> float:
    gray = np.asarray(patch.convert("L"))
    if gray.size == 0:
        return 1.0
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))


def _foreground_stats(foreground: Image.Image) -> tuple[np.ndarray, float]:
    rgb = np.asarray(foreground.convert("RGB")).astype(np.float32)
    alpha = np.asarray(foreground.getchannel("A")).astype(np.float32) / 255.0
    active = alpha > 0.05
    if int(active.sum()) < 8:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32), 0.0
    return rgb[active].mean(axis=0), float(active.mean())


def extract_features(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
    mask_quality: float,
) -> Dict[str, float]:
    bg_w, bg_h = background.size
    w, h = foreground.size
    x = int(round(x))
    y = int(round(y))
    opa_tensor = build_opa_style_input(background, foreground, x, y)
    opa_signals = opa_style_reference_forward(opa_tensor)

    margin_px = min(max(0, x), max(0, y), max(0, bg_w - x - w), max(0, bg_h - y - h))
    margin = clamp(margin_px / max(1.0, min(bg_w, bg_h) * 0.08))

    bottom = (y + h) / max(1.0, bg_h)
    lower_half = clamp(1.0 - abs(bottom - 0.78) / 0.34)

    center_x = (x + w / 2.0) / max(1.0, bg_w)
    center_balance = clamp(1.0 - abs(center_x - 0.5) / 0.55)

    area_ratio = (w * h) / max(1.0, bg_w * bg_h)
    if area_ratio < 0.018:
        size_reasonable = clamp(area_ratio / 0.018 * 0.62)
    elif area_ratio > 0.24:
        size_reasonable = clamp(1.0 - (area_ratio - 0.24) / 0.2)
    else:
        size_reasonable = clamp(1.0 - abs(area_ratio - 0.085) / 0.16)

    patch = _safe_patch(background, x, y, w, h)
    density = _edge_density(patch)
    background_clean = clamp(1.0 - max(0.0, density - 0.045) / 0.18)

    fg_mean, mask_ratio = _foreground_stats(foreground)
    patch_rgb = np.asarray(patch.resize(foreground.size).convert("RGB")).astype(np.float32)
    patch_mean = patch_rgb.mean(axis=(0, 1))
    fg_luma = float(np.dot(fg_mean, [0.299, 0.587, 0.114]))
    bg_luma = float(np.dot(patch_mean, [0.299, 0.587, 0.114]))
    brightness_match = clamp(1.0 - abs(fg_luma - bg_luma) / 120.0)
    color_match = clamp(1.0 - float(np.linalg.norm(fg_mean - patch_mean)) / 245.0)

    below_top = min(bg_h - 1, y + h)
    below_bottom = min(bg_h, below_top + max(4, h // 8))
    if below_bottom > below_top + 1:
        support_patch = background.crop((x, below_top, min(x + w, bg_w), below_bottom)).convert("L")
        support_edges = _edge_density(support_patch.convert("RGB"))
        surface_stability = clamp(1.0 - max(0.0, support_edges - 0.06) / 0.18)
    else:
        surface_stability = 0.35
    support = clamp(0.65 * lower_half + 0.35 * surface_stability)

    mask_shape = clamp(1.0 - abs(mask_ratio - 0.46) / 0.58)
    mask_score = clamp(mask_quality * 0.68 + mask_shape * 0.32)

    return {
        "opa_rgb_mask_score": round(opa_signals.opa_rgb_mask_score, 4),
        "margin": round(margin, 4),
        "lower_half": round(lower_half, 4),
        "center_balance": round(center_balance, 4),
        "size_reasonable": round(size_reasonable, 4),
        "background_clean": round(background_clean, 4),
        "brightness_match": round(brightness_match, 4),
        "color_match": round(color_match, 4),
        "support": round(support, 4),
        "mask_quality": round(mask_score, 4),
        "opa_mask_area": opa_signals.opa_mask_area,
        "opa_mask_bottom": opa_signals.opa_mask_bottom,
        "opa_mask_center_x": opa_signals.opa_mask_center_x,
    }


def _label(score: float) -> str:
    if score >= 0.74:
        return "推荐"
    if score >= 0.52:
        return "可接受"
    return "不推荐"


def _reason(features: Dict[str, float], score: float) -> str:
    weak = sorted(((name, features[name]) for name in FEATURE_ORDER), key=lambda item: item[1])[:2]
    if score >= 0.74:
        return "位置比例、支撑区域和局部背景较协调"
    messages = {
        "opa_rgb_mask_score": "OPA风格RGB+mask评估分偏低",
        "margin": "靠近边界",
        "lower_half": "垂直位置不够自然",
        "center_balance": "水平位置偏离视觉中心",
        "size_reasonable": "物体尺寸比例不理想",
        "background_clean": "背景局部纹理较复杂",
        "brightness_match": "亮度差异较明显",
        "color_match": "颜色协调度较低",
        "support": "底部支撑感不足",
        "mask_quality": "前景 mask 质量有限",
    }
    return "、".join(messages.get(name, name) for name, _ in weak)


def score_candidate(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
    mask_quality: float,
    candidate_id: str,
) -> PlacementCandidate:
    bg_w, bg_h = background.size
    w, h = foreground.size
    x = max(0, min(int(round(x)), max(0, bg_w - w)))
    y = max(0, min(int(round(y)), max(0, bg_h - h)))
    features = extract_features(background, foreground, x, y, mask_quality)
    feature_tensor = torch.tensor([[features[name] for name in FEATURE_ORDER]], dtype=torch.float32)
    with torch.no_grad():
        fallback_score = float(MODEL(feature_tensor).item())

    trained = score_with_trained_opa(background, foreground, x, y)
    features["fallback_score"] = round(fallback_score, 4)
    features["trained_opa_available"] = 1.0 if trained.available else 0.0
    features["trained_opa_score"] = trained.score if trained.score is not None else None
    features["trained_opa_message"] = trained.message
    if trained.available and trained.score is not None:
        score = 0.72 * trained.score + 0.28 * fallback_score
    else:
        score = fallback_score
    return PlacementCandidate(
        id=candidate_id,
        x=x,
        y=y,
        w=w,
        h=h,
        nx=round(x / max(1, bg_w), 5),
        ny=round(y / max(1, bg_h), 5),
        nw=round(w / max(1, bg_w), 5),
        nh=round(h / max(1, bg_h), 5),
        score=round(score, 4),
        label=_label(score),
        reason=_reason(features, score),
        features=features,
    )


def rank_candidates(candidates: List[PlacementCandidate]) -> List[PlacementCandidate]:
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
