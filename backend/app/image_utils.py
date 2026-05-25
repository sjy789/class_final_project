from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass
class ProcessedForeground:
    image: Image.Image
    mask_source: str
    mask_quality: float
    original_size: Tuple[int, int]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def load_rgba(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGBA")


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _mask_from_rgb_heuristic(image: Image.Image) -> tuple[np.ndarray, str, float]:
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    height, width = rgb.shape[:2]
    border = max(3, min(height, width) // 28)
    samples = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(samples, axis=0)
    distance = np.linalg.norm(rgb - background_color, axis=2)
    threshold = max(22.0, float(np.percentile(distance, 68)))
    mask = np.where(distance > threshold, 255, 0).astype(np.uint8)

    kernel_size = max(3, min(height, width) // 80)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _largest_component(mask)

    coverage = float(mask.mean() / 255.0)
    if coverage < 0.03 or coverage > 0.94:
        mask = np.full((height, width), 255, dtype=np.uint8)
        return mask, "rectangle", 0.35
    quality = clamp(0.42 + (1.0 - abs(coverage - 0.38)) * 0.34, 0.35, 0.76)
    return mask, "border-color", quality


def process_foreground(image: Image.Image) -> ProcessedForeground:
    original_size = image.size
    alpha = np.asarray(image.getchannel("A")).astype(np.uint8)
    transparent_ratio = float(np.mean(alpha < 250))

    if transparent_ratio > 0.01:
        mask = alpha
        source = "alpha"
        quality = clamp(0.74 + transparent_ratio * 0.8, 0.74, 0.98)
    else:
        mask, source, quality = _mask_from_rgb_heuristic(image)

    ys, xs = np.where(mask > 8)
    if len(xs) == 0 or len(ys) == 0:
        mask = np.full(mask.shape, 255, dtype=np.uint8)
        ys, xs = np.where(mask > 8)
        source = "rectangle"
        quality = 0.3

    left = max(int(xs.min()) - 4, 0)
    top = max(int(ys.min()) - 4, 0)
    right = min(int(xs.max()) + 5, image.width)
    bottom = min(int(ys.max()) + 5, image.height)

    cropped = image.crop((left, top, right, bottom)).convert("RGBA")
    cropped_mask = Image.fromarray(mask[top:bottom, left:right], mode="L")
    cropped.putalpha(cropped_mask)
    return ProcessedForeground(
        image=cropped,
        mask_source=source,
        mask_quality=quality,
        original_size=original_size,
    )


def resize_foreground(foreground: ProcessedForeground, target_width: int) -> Image.Image:
    width = max(16, int(target_width))
    ratio = width / max(1, foreground.image.width)
    height = max(16, int(round(foreground.image.height * ratio)))
    return foreground.image.resize((width, height), Image.Resampling.LANCZOS)


def harmonize_foreground(foreground: Image.Image, background_patch: Image.Image) -> Image.Image:
    if foreground.width != background_patch.width or foreground.height != background_patch.height:
        background_patch = background_patch.resize(foreground.size, Image.Resampling.BICUBIC)

    fg_rgb = np.asarray(foreground.convert("RGB")).astype(np.float32)
    bg_rgb = np.asarray(background_patch.convert("RGB")).astype(np.float32)
    alpha = np.asarray(foreground.getchannel("A")).astype(np.float32) / 255.0
    active = alpha > 0.05
    if int(active.sum()) < 16:
        return foreground

    fg_mean = fg_rgb[active].mean(axis=0)
    bg_mean = bg_rgb[active].mean(axis=0)
    channel_gain = np.power((bg_mean + 8.0) / (fg_mean + 8.0), 0.32)
    channel_gain = np.clip(channel_gain, 0.78, 1.24)

    adjusted = fg_rgb * channel_gain.reshape(1, 1, 3)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    result = Image.fromarray(adjusted, mode="RGB").convert("RGBA")
    result.putalpha(foreground.getchannel("A"))

    fg_luma = float(np.dot(fg_mean, [0.299, 0.587, 0.114]))
    bg_luma = float(np.dot(bg_mean, [0.299, 0.587, 0.114]))
    gain = clamp(bg_luma / max(1.0, fg_luma), 0.82, 1.18)
    result = ImageEnhance.Brightness(result).enhance(gain)
    return result


def compose_image(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
    harmonize: bool = True,
) -> Image.Image:
    canvas = background.convert("RGBA").copy()
    x = int(round(x))
    y = int(round(y))
    x = max(0, min(x, max(0, canvas.width - foreground.width)))
    y = max(0, min(y, max(0, canvas.height - foreground.height)))

    fg = foreground
    if harmonize:
        patch = canvas.crop((x, y, x + foreground.width, y + foreground.height)).convert("RGB")
        fg = harmonize_foreground(foreground, patch)

    canvas.alpha_composite(fg, (x, y))
    return canvas


def transparent_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    normalized = heatmap.astype(np.float32)
    if float(normalized.max()) > 0:
        normalized = normalized / float(normalized.max())
    normalized = cv2.resize(normalized, size, interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(normalized, 0, 1)

    values = (normalized * 255).astype(np.uint8)
    color = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    alpha = np.where(normalized > 0.03, normalized * 185, 0).astype(np.uint8)
    rgba = np.dstack([color, alpha])
    return Image.fromarray(rgba, mode="RGBA")

