from __future__ import annotations

import torch
from torch import nn


class TinyOPAConvNet(nn.Module):
    """A small OPA-style CNN for RGB+mask placement assessment.

    The reference OPA/SimOPA model uses a 4-channel input made from the
    composite RGB image and foreground mask. This model keeps that input
    contract but uses a compact CNN so it can be trained on a GPU server and
    deployed locally on CPU.
    """

    def __init__(self, width: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, 1),
        )

    def forward(self, img_cat: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(img_cat))
        return logits.flatten()


def create_tiny_opa_model(width: int = 32, dropout: float = 0.15) -> TinyOPAConvNet:
    return TinyOPAConvNet(width=width, dropout=dropout)

