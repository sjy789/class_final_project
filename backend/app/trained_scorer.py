from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .opa_cnn import create_tiny_opa_model
from .reference_adapter import build_opa_style_input


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "backend" / "checkpoints" / "opa_tiny.pt"


@dataclass
class TrainedOPAScore:
    available: bool
    score: float | None
    checkpoint: str | None
    message: str


class TrainedOPAScorer:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "cpu") -> None:
        env_path = os.environ.get("OPA_MODEL_CHECKPOINT")
        self.checkpoint_path = Path(checkpoint_path or env_path or DEFAULT_CHECKPOINT)
        self.device = torch.device(device)
        self.model = None
        self.image_size = 128
        self.width = 32
        self.dropout = 0.15
        self.message = "checkpoint not loaded"
        self._loaded = False

    def _load(self) -> None:
        if self._loaded and self.model is not None:
            return
        if self._loaded and not self.checkpoint_path.exists():
            return
        self._loaded = True
        if not self.checkpoint_path.exists():
            self.message = f"checkpoint not found: {self.checkpoint_path}"
            return

        checkpoint: dict[str, Any] = torch.load(self.checkpoint_path, map_location=self.device)
        self.image_size = int(checkpoint.get("image_size", self.image_size))
        self.width = int(checkpoint.get("width", self.width))
        self.dropout = float(checkpoint.get("dropout", self.dropout))
        model = create_tiny_opa_model(width=self.width, dropout=self.dropout)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        self.message = f"loaded checkpoint: {self.checkpoint_path}"

    def score(self, background: Image.Image, foreground: Image.Image, x: int, y: int) -> TrainedOPAScore:
        self._load()
        if self.model is None:
            return TrainedOPAScore(
                available=False,
                score=None,
                checkpoint=str(self.checkpoint_path),
                message=self.message,
            )

        tensor = build_opa_style_input(background, foreground, x, y, input_size=self.image_size)
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor)
            score = torch.sigmoid(logit).item()
        return TrainedOPAScore(
            available=True,
            score=round(float(score), 4),
            checkpoint=str(self.checkpoint_path),
            message=self.message,
        )

    def status(self) -> dict:
        self._load()
        return {
            "available": self.model is not None,
            "checkpoint": str(self.checkpoint_path),
            "image_size": self.image_size,
            "model": "TinyOPAConvNet",
            "message": self.message,
        }


SCORER = TrainedOPAScorer()


def score_with_trained_opa(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
) -> TrainedOPAScore:
    return SCORER.score(background, foreground, x, y)


def get_trained_opa_status() -> dict:
    return SCORER.status()
