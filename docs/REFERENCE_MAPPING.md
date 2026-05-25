# Reference code mapping

This project now explicitly references the teacher-provided direction A resources:

- `bcmi/Object-Placement-Assessment-Dataset-OPA`
- `bcmi/TopNet-Object-Placement`
- `bcmi/libcom`

## 1. OPA / SimOPA

Reference role:

OPA is used as the main reference for the placement assessment model interface. Its SimOPA-style task evaluates whether a pasted foreground object is reasonable in a background image. The important reference idea used here is:

```text
composite RGB image + foreground mask -> placement assessment score
```

Local adaptation:

- File: `backend/app/reference_adapter.py`
- Function: `build_opa_style_input`
- Function: `opa_style_reference_forward`
- Trainable model: `backend/app/opa_cnn.py`
- Training script: `training/train_opa_tiny.py`

The project builds an OPA-style 4-channel tensor:

```text
[R, G, B, mask]
```

Then it extracts a local `opa_rgb_mask_score` from this tensor and feeds it into the scoring head in `backend/app/model.py`.

Difference from original reference:

The default runnable demo does not require the pretrained SimOPA checkpoint. The project adds a lightweight OPA-style CNN training path so the group can train a real checkpoint on the OPA dataset, then deploy that checkpoint locally in the FastAPI backend. If no checkpoint exists, the project falls back to the deterministic OPA-style scoring branch for stable classroom demonstration.

## 2. TopNet-Object-Placement

Reference role:

TopNet is used as the reference for organizing object placement as a multi-location and multi-scale scoring task. The important reference idea used here is:

```text
generate many candidate placements -> score each candidate -> select top candidates
```

Local adaptation:

- File: `backend/app/pipeline.py`
- Function: `generate_candidates`
- Function: `rank_candidates`

The current implementation generates candidates at several object scales and several background positions, scores every candidate, and returns Top-1 / Top-3 / Top-5 in the Web interface.

Difference from original reference:

TopNet predicts dense placement maps with a neural network. This project uses a smaller explicit candidate grid and a lightweight scoring model to avoid unstable dependencies and to keep local CPU inference fast.

## 3. libcom

Reference role:

libcom is used as the application-level reference toolkit. It shows that an image composition application can combine multiple tools such as object placement assessment, FOPA-style heatmap guidance, blending/harmonization and color transfer.

Local adaptation:

- Foreground processing: `backend/app/image_utils.py`
- Image composition: `backend/app/image_utils.py`
- Color harmonization: `backend/app/image_utils.py`
- Explanation heatmap: `backend/app/pipeline.py`

The current project keeps the same pipeline idea:

```text
foreground mask -> candidate generation -> placement assessment -> harmonization -> composition -> explanation heatmap
```

Difference from original reference:

The project does not call libcom as an installed Python package. Instead, it uses libcom as a reference for how to organize an integrated composition application, and implements a lightweight local version suitable for the course demo.

## Report wording

Recommended honest wording for the report:

> This project references the direction A resources from BCMI. OPA/SimOPA is used as the reference for the placement assessment input-output form: RGB composite image plus foreground mask as model input, and placement reasonableness score as output. TopNet is used as the reference for multi-scale and multi-location candidate scoring and Top-K recommendation. libcom is used as the reference for the complete image composition toolchain, including foreground processing, placement assessment, harmonization, composition and heatmap-style visualization. Due to local CPU deployment and classroom demo stability, the project does not directly load the original pretrained checkpoints, but implements a lightweight PyTorch adapter and scoring head following these interfaces.
