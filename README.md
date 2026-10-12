# VRU

PyTorch starter for current pedestrian behavior recognition using PIE annotations.
The model predicts walking, looking toward the ego vehicle, and current crossing
state from a history of pedestrian crops and image-space bounding boxes.

This version recognizes behavior at the final observed frame. Future crossing
prediction and trajectory forecasting are not implemented. The baseline uses
three independent binary outputs; `crossing-irrelevant` is ignored by the binary
crossing loss. A three-class crossing head has not been implemented.

## Setup

Run commands from the repository root. Python 3.12 was used for local validation.
Install a compatible PyTorch/torchvision pair for your hardware.

```bash
python -m pip install -r pie_behavior/requirements.txt
python -m pytest pie_behavior/tests -q
```

## Data layout

Download the original data from the [PIE project](https://github.com/aras62/PIE).
Videos and labels are external inputs and are not stored in this repository.

```text
data/PIE/
  annotations/set01/video_0001_annt.xml
  ...
  PIE_clips/set01/video_0001.mp4
  ...
  images/set01/video_0001/00000.png  # created by extract_frames
```

The predefined set split is train: set01/set02/set04; validation: set05/set06;
test: set03. Keep clips from the same recording in the same split.

## Prepare, train, evaluate

```bash
python -m pie_behavior.prepare --data-root data/PIE --out output/manifest_v1
python -m pie_behavior.extract_frames --data-root data/PIE --manifest-dir output/manifest_v1
python -m pie_behavior.train --data-root data/PIE --manifest-dir output/manifest_v1 --out output/run_v1 --epochs 20 --device auto
python -m pie_behavior.evaluate --data-root data/PIE --manifest-dir output/manifest_v1 --checkpoint output/run_v1/best.pt --out output/test_v1 --split test
```

Output directories must be new. The model uses 16 observations sampled at 10 Hz,
covering 1.5 seconds including the current frame. ResNet18 ImageNet weights are
downloaded on the first pretrained training run; evaluation reloads the saved
checkpoint without downloading them again.

For inference, provide a JSON history with `sample_id`, original image `width`
and `height`, `boxes`, `images`, and `time_offsets_s`. Targets are not needed.
The number of frames and time grid must match the checkpoint.

```bash
python -m pie_behavior.predict --checkpoint output/run_v1/best.pt --input observed_history.json --data-root data/PIE --out output/prediction.json
```

## Synthetic smoke run

Synthetic data validates the software flow, not real pedestrian accuracy.

```bash
python -m pie_behavior.make_demo --out tmp/demo_v1
python -m pie_behavior.prepare --data-root tmp/demo_v1 --out output/demo_manifest_v1
python -m pie_behavior.train --data-root tmp/demo_v1 --manifest-dir output/demo_manifest_v1 --out output/demo_run_v1 --backbone tiny --hidden-size 32 --image-size 32 --epochs 2 --device cpu
```

## Crop export and looking comparison

Export JPEG crops directly from a video and retain the original labels, bbox,
track/frame IDs and storage measurements:

```bash
python -m pie_behavior.crop_video --data-root data/PIE --set-id set01 --video-id video_0001 --out output/crops_train_128 --image-size 128 --quality 90
python -m pie_behavior.crop_video --data-root data/PIE --set-id set01 --video-id video_0001 --out output/crops_train_224 --image-size 224 --quality 95
python -m pie_behavior.crop_video --data-root data/PIE --set-id set06 --video-id video_0009 --out output/crops_val_128 --image-size 128 --quality 90
python -m pie_behavior.crop_video --data-root data/PIE --set-id set06 --video-id video_0009 --out output/crops_val_224 --image-size 224 --quality 95
python -m pie_behavior.compare_looking --data-root data/PIE --train-low output/crops_train_128 --train-high output/crops_train_224 --val-low output/crops_val_128 --val-high output/crops_val_224 --out output/looking_comparison_v1 --epochs 30
```

`compare_looking` is a dedicated CUDA pilot: frozen pretrained ResNet18, geometry
MLP and GRU, three paired seeds, fixed final epoch and threshold 0.5. It trains
only the looking output. Walking and crossing outputs in these pilot checkpoints
are not valid predictions. Results include per-seed metrics, paired deltas,
predictions, checkpoints and track-cluster bootstrap intervals.

The normal `BehaviorDataset` reads full-frame PNGs and crops them during loading.
Exported JPEG caches are consumed by the dedicated comparison script; they are
not yet supported by the general `train` CLI.

The local two-video pilot used 135 train windows and 422 validation windows.
Positive looking F1 averaged 30.77% at 128/Q90 and 33.80% at 224/Q95, but false
positives increased, AP did not improve, and the F1 difference interval included
zero. This pilot does not establish a reliable full-dataset improvement.

## Source files

| Module | Purpose |
|---|---|
| `prepare` | Parse XML, build causal windows and versioned split manifests |
| `data` | Image preprocessing, geometry features and dataset validation |
| `model` | CNN + geometry MLP + GRU with binary behavior outputs |
| `engine` | Masked losses, epoch execution, metrics and majority baseline |
| `train` | Training configuration, optimizer, validation and checkpoints |
| `evaluate`, `predict` | Held-out evaluation and one-history inference |
| `extract_frames` | Decode the full-frame PNGs required by manifests |
| `crop_video` | Export annotated JPEG crops and storage/QA reports |
| `compare_looking` | Paired crop-quality experiment for looking recognition |
| `audit_annotations` | Audit annotation ZIP without extracting video frames |
| `make_demo` | Generate a synthetic smoke-test fixture |

Local datasets, weights, generated reports and the personal Vietnamese code
reading guide are excluded from version control.
