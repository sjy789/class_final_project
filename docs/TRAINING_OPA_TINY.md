# OPA tiny model training

This folder adds a real training path for the course project.

The trained model follows the OPA/SimOPA reference idea:

```text
RGB composite image + foreground mask -> placement reasonableness score
```

It is intentionally smaller than the original OPA/TopNet-style models, so training can be done on a GPU server and deployment can still run on local CPU.

## Files

- `backend/app/opa_cnn.py`: trainable tiny CNN model.
- `training/opa_dataset.py`: OPA dataset reader.
- `training/train_opa_tiny.py`: training entrypoint.
- `backend/app/trained_scorer.py`: checkpoint loader used by the FastAPI backend.
- `backend/checkpoints/opa_tiny.pt`: expected checkpoint location after training.

## Dataset

Use the teacher reference dataset:

```text
bcmi/Object-Placement-Assessment-Dataset-OPA
```

The loader supports two forms:

1. CSV mode: `train_set.csv` / `test_set.csv`, where the last three fields are treated as:

```text
label, composite_image_path, mask_path
```

2. Folder scan mode: images under `composite/train_set` or `composite/test_set`, with matching `mask_*` files and a `0/1` label in the filename.

## Train on GPU server

Example:

```bash
python training/train_opa_tiny.py \
  --data-root /path/to/Object-Placement-Assessment-Dataset-OPA \
  --output backend/checkpoints/opa_tiny.pt \
  --epochs 12 \
  --batch-size 64 \
  --image-size 128 \
  --device cuda
```

For a quick test:

```bash
python training/train_opa_tiny.py \
  --data-root /path/to/OPA \
  --output backend/checkpoints/opa_tiny.pt \
  --epochs 1 \
  --batch-size 16 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --device cuda
```

The script saves:

- `backend/checkpoints/opa_tiny.pt`
- `backend/checkpoints/opa_tiny.history.json`

## Bring checkpoint back to local machine

Copy the trained file into:

```text
backend/checkpoints/opa_tiny.pt
```

Then start or refresh the backend:

```powershell
.\scripts\start_backend.ps1
```

The backend will load the checkpoint and fuse its score with the fallback scoring features. If no checkpoint exists, the application still runs with the fallback OPA-style feature scorer.

## Is this still local inference?

Yes.

Training can happen on a GPU server, but inference happens inside the local FastAPI backend:

```text
React frontend -> 127.0.0.1 FastAPI -> local PyTorch checkpoint -> result
```

No cloud API is needed for inference.

## Report wording

Recommended wording:

> Based on the OPA/SimOPA reference interface, the project implements a lightweight RGB+mask placement assessment model. The model changes the input form to 4 channels and outputs a 0-1 reasonableness score. Training is performed on the OPA dataset on a GPU server, while the final Web application loads the checkpoint locally and runs CPU inference through the FastAPI backend. If the checkpoint is unavailable, the system falls back to a deterministic OPA-style scoring branch so the application remains runnable.

