from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.opa_cnn import create_tiny_opa_model  # noqa: E402
from training.opa_dataset import OPAAssessmentDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny OPA-style RGB+mask placement model.")
    parser.add_argument("--data-root", required=True, help="Path to Object-Placement-Assessment-Dataset-OPA root.")
    parser.add_argument("--output", default="backend/checkpoints/opa_tiny.pt", help="Checkpoint output path.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    labels = labels.float()
    correct = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    recall_pos = tp / max(1, tp + fn)
    recall_neg = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall_pos / max(1e-8, precision + recall_pos)
    return {
        "accuracy": round(correct, 4),
        "balanced_accuracy": round((recall_pos + recall_neg) * 0.5, 4),
        "precision": round(precision, 4),
        "recall": round(recall_pos, 4),
        "f1": round(f1, 4),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = compute_metrics(logits_cat, labels_cat)
    metrics["loss"] = round(total_loss / max(1, len(loader.dataset)), 4)
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    train_counts: dict[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "TinyOPAConvNet",
            "model_state": model.state_dict(),
            "image_size": args.image_size,
            "width": args.width,
            "dropout": args.dropout,
            "epoch": epoch,
            "metrics": metrics,
            "train_counts": train_counts,
            "reference": "OPA/SimOPA RGB+mask placement assessment adaptation",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    train_dataset = OPAAssessmentDataset(
        args.data_root,
        split="train",
        image_size=args.image_size,
        augment=True,
        max_samples=args.max_train_samples,
    )
    val_dataset = OPAAssessmentDataset(
        args.data_root,
        split="test",
        image_size=args.image_size,
        augment=False,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_tiny_opa_model(width=args.width, dropout=args.dropout).to(device)
    counts = train_dataset.label_counts()
    pos_weight = torch.tensor([counts.get(0, 1) / max(1, counts.get(1, 1))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    best_score = -1.0
    history: list[dict] = []
    print(json.dumps({"device": str(device), "train": len(train_dataset), "val": len(val_dataset), "counts": counts}))

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        selection_score = val_metrics["balanced_accuracy"] + val_metrics["f1"]
        if selection_score > best_score:
            best_score = selection_score
            save_checkpoint(output, model, args, epoch, val_metrics, counts)

    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output), "history": str(history_path)}))


if __name__ == "__main__":
    main()

