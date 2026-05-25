from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from .image_utils import (
    compose_image,
    image_to_data_url,
    load_rgba,
    process_foreground,
    resize_foreground,
    transparent_heatmap,
)
from .model import PlacementCandidate, rank_candidates, score_candidate
from .reference_adapter import REFERENCE_NOTES
from .trained_scorer import get_trained_opa_status


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _clamp_scale(scale: float) -> float:
    return max(0.08, min(0.42, float(scale)))


def _prepare_images(background_bytes: bytes, foreground_bytes: bytes):
    background = load_rgba(background_bytes)
    foreground_raw = load_rgba(foreground_bytes)
    processed = process_foreground(foreground_raw)
    return background, processed


def _target_foreground(processed, background: Image.Image, scale: float) -> Image.Image:
    scale = _clamp_scale(scale)
    target_width = int(round(background.width * scale))
    return resize_foreground(processed, target_width)


def _candidate_positions(background: Image.Image, foreground: Image.Image) -> Iterable[tuple[int, int]]:
    bg_w, bg_h = background.size
    w, h = foreground.size
    x_slots = [0.08, 0.24, 0.40, 0.56, 0.72, 0.86]
    bottom_slots = [0.54, 0.66, 0.78, 0.9]
    seen: set[tuple[int, int]] = set()
    for xf in x_slots:
        for bf in bottom_slots:
            x = int(round((bg_w - w) * xf))
            y = int(round(bg_h * bf - h))
            x = max(0, min(x, max(0, bg_w - w)))
            y = max(0, min(y, max(0, bg_h - h)))
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            yield key


def generate_candidates(background: Image.Image, processed, scale: float) -> List[tuple[str, Image.Image, int, int]]:
    variants = [scale * 0.86, scale, scale * 1.14]
    generated: list[tuple[str, Image.Image, int, int]] = []
    for scale_index, variant in enumerate(variants):
        foreground = _target_foreground(processed, background, variant)
        for pos_index, (x, y) in enumerate(_candidate_positions(background, foreground)):
            generated.append((f"c{scale_index + 1}-{pos_index + 1}", foreground, x, y))
    return generated


def _attach_previews(
    background: Image.Image,
    candidate_sources: list[tuple[str, Image.Image, int, int]],
    ranked: list[PlacementCandidate],
    harmonize: bool,
    limit: int,
) -> None:
    source_map = {candidate_id: foreground for candidate_id, foreground, _, _ in candidate_sources}
    for candidate in ranked[:limit]:
        foreground = source_map[candidate.id]
        composite = compose_image(background, foreground, candidate.x, candidate.y, harmonize=harmonize)
        preview = composite.copy()
        preview.thumbnail((520, 360), Image.Resampling.LANCZOS)
        candidate.preview = image_to_data_url(preview)


def _candidate_source(candidate_sources, candidate_id: str) -> Image.Image:
    for source_id, foreground, _, _ in candidate_sources:
        if source_id == candidate_id:
            return foreground
    raise KeyError(candidate_id)


def _make_heatmap(
    background: Image.Image,
    foreground: Image.Image,
    candidate: PlacementCandidate,
    mask_quality: float,
    grid_x: int = 12,
    grid_y: int = 8,
) -> Image.Image:
    base_score = candidate.score
    bg_rgb = background.convert("RGB")
    bg_array = np.asarray(bg_rgb).copy()
    average_color = bg_array.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    heat = np.zeros((grid_y, grid_x), dtype=np.float32)
    cell_w = max(1, background.width // grid_x)
    cell_h = max(1, background.height // grid_y)

    for gy in range(grid_y):
        for gx in range(grid_x):
            x1 = gx * cell_w
            y1 = gy * cell_h
            x2 = background.width if gx == grid_x - 1 else min(background.width, x1 + cell_w)
            y2 = background.height if gy == grid_y - 1 else min(background.height, y1 + cell_h)
            occluded = bg_array.copy()
            occluded[y1:y2, x1:x2] = average_color
            occluded_image = Image.fromarray(occluded, mode="RGB").convert("RGBA")
            changed = score_candidate(
                occluded_image,
                foreground,
                candidate.x,
                candidate.y,
                mask_quality,
                f"occ-{gx}-{gy}",
            )
            heat[gy, gx] = max(0.0, base_score - changed.score)

    return transparent_heatmap(heat, background.size)


def _write_run_outputs(result: dict, composite: Image.Image, heatmap: Image.Image | None) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    composite.save(run_dir / "top_composite.png")
    if heatmap is not None:
        heatmap.save(run_dir / "explanation_heatmap.png")
    with (run_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return str(run_dir.relative_to(PROJECT_ROOT))


def analyze_auto(
    background_bytes: bytes,
    foreground_bytes: bytes,
    top_k: int = 3,
    scale: float = 0.22,
    harmonize: bool = True,
    explain: bool = True,
) -> dict:
    top_k = max(1, min(5, int(top_k)))
    background, processed = _prepare_images(background_bytes, foreground_bytes)
    sources = generate_candidates(background, processed, _clamp_scale(scale))

    scored: list[PlacementCandidate] = []
    for candidate_id, foreground, x, y in sources:
        scored.append(score_candidate(background, foreground, x, y, processed.mask_quality, candidate_id))
    ranked = rank_candidates(scored)
    _attach_previews(background, sources, ranked, harmonize, limit=max(top_k, 5))

    best = ranked[0]
    best_foreground = _candidate_source(sources, best.id)
    composite = compose_image(background, best_foreground, best.x, best.y, harmonize=harmonize)
    heatmap = _make_heatmap(background, best_foreground, best, processed.mask_quality) if explain else None

    result = {
        "mode": "auto",
        "references": REFERENCE_NOTES,
        "trained_opa": get_trained_opa_status(),
        "background": {"width": background.width, "height": background.height},
        "foreground": {
            "width": best_foreground.width,
            "height": best_foreground.height,
            "mask_source": processed.mask_source,
            "mask_quality": round(processed.mask_quality, 4),
            "original_width": processed.original_size[0],
            "original_height": processed.original_size[1],
        },
        "top": [candidate.to_dict() for candidate in ranked[:top_k]],
        "candidates": [candidate.to_dict() for candidate in ranked[: min(18, len(ranked))]],
        "composite": image_to_data_url(composite),
        "heatmap": image_to_data_url(heatmap) if heatmap is not None else None,
    }
    result["output_dir"] = _write_run_outputs(result, composite, heatmap)
    return result


def evaluate_manual(
    background_bytes: bytes,
    foreground_bytes: bytes,
    x_norm: float,
    y_norm: float,
    scale: float = 0.22,
    harmonize: bool = True,
    explain: bool = False,
) -> dict:
    background, processed = _prepare_images(background_bytes, foreground_bytes)
    foreground = _target_foreground(processed, background, _clamp_scale(scale))
    x = int(round(float(x_norm) * background.width))
    y = int(round(float(y_norm) * background.height))
    candidate = score_candidate(background, foreground, x, y, processed.mask_quality, "manual")
    composite = compose_image(background, foreground, candidate.x, candidate.y, harmonize=harmonize)
    heatmap = _make_heatmap(background, foreground, candidate, processed.mask_quality) if explain else None
    candidate.preview = image_to_data_url(composite)
    return {
        "mode": "manual",
        "references": REFERENCE_NOTES,
        "trained_opa": get_trained_opa_status(),
        "background": {"width": background.width, "height": background.height},
        "foreground": {
            "width": foreground.width,
            "height": foreground.height,
            "mask_source": processed.mask_source,
            "mask_quality": round(processed.mask_quality, 4),
        },
        "candidate": candidate.to_dict(),
        "composite": image_to_data_url(composite),
        "heatmap": image_to_data_url(heatmap) if heatmap is not None else None,
    }
