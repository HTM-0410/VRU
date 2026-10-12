"""Causal image and geometry inputs. Behavior annotations are targets only."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import HEADS, SCHEMA_VERSION
from .prepare import SPLITS, sha256


def read_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("heads") != list(HEADS):
        raise ValueError("Incompatible manifest schema or head order")
    if manifest.get("status") != "ready":
        raise ValueError("Manifest build is incomplete")
    for split in ("train", "val", "test"):
        if sha256(directory / f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Manifest content hash mismatch: {split}")
    return manifest


def geometry(sample):
    boxes = np.asarray(sample["boxes"], dtype=np.float32)
    times = np.asarray(sample["time_offsets_s"], dtype=np.float32)
    if (len(boxes) < 2 or boxes.shape != (len(times), 4) or not np.isfinite(boxes).all()
            or not np.isfinite(times).all() or times[-1] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError(f"Invalid causal history: {sample['sample_id']}")
    w, h = sample["width"], sample["height"]
    if (w <= 0 or h <= 0 or np.any(boxes[:, 2] <= boxes[:, 0]) or
            np.any(boxes[:, 3] <= boxes[:, 1])):
        raise ValueError(f"Invalid image dimensions or box extent: {sample['sample_id']}")
    xywh = np.stack(((boxes[:, 0] + boxes[:, 2]) / (2 * w),
                     (boxes[:, 1] + boxes[:, 3]) / (2 * h),
                     (boxes[:, 2] - boxes[:, 0]) / w,
                     (boxes[:, 3] - boxes[:, 1]) / h), axis=-1)
    velocity = np.zeros_like(xywh)
    velocity[1:] = np.diff(xywh, axis=0) / np.diff(times)[:, None]
    return np.concatenate((xywh, velocity, times[:, None]), axis=-1)


def image_tensor(path, box, width, height, image_size, context):
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError(f"Image size {image.size} != annotation {(width, height)}: {path}")
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) * context, (y2 - y1) * context
        crop = (max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))),
                min(width, int(np.ceil(x2 + dx))), min(height, int(np.ceil(y2 + dy))))
        image = image.convert("RGB").crop(crop).resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def make_inputs(sample, data_root, visual=True, image_size=128, context=0.25):
    result = {"geometry": torch.from_numpy(geometry(sample))}
    if visual:
        if len(sample["images"]) != len(sample["boxes"]):
            raise ValueError("Image/box history length mismatch")
        result["images"] = torch.stack([
            image_tensor(Path(data_root) / path, box, sample["width"], sample["height"], image_size, context)
            for path, box in zip(sample["images"], sample["boxes"])
        ])
    return result


class BehaviorDataset(Dataset):
    def __init__(self, manifest_dir, split, data_root, visual=True, image_size=128, context=0.25):
        self.data_root = Path(data_root)
        self.visual, self.image_size, self.context = visual, image_size, context
        self.samples = [json.loads(line) for line in (Path(manifest_dir) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.samples:
            raise ValueError(f"Empty {split} split; provide the required PIE sets")
        if image_size < 32 or context < 0:
            raise ValueError("image_size >= 32 and context >= 0 required")
        for sample in self.samples:
            if SPLITS.get(sample["set_id"]) != split or sample["recording_group"] != sample["set_id"]:
                raise ValueError(f"Incorrect split or recording group: {sample['sample_id']}")
            if len(sample["target"]) != len(HEADS) or len(sample["valid"]) != len(HEADS):
                raise ValueError("Invalid target shape")
            if any(v and y not in (0, 1) for v, y in zip(sample["valid"], sample["target"])):
                raise ValueError("Valid targets must be binary")
        if visual:
            paths = {p for s in self.samples for p in s["images"]}
            missing = [p for p in sorted(paths) if not (self.data_root / p).is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} missing frames; first: {missing[0]}. Run pie_behavior.extract_frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        result = make_inputs(s, self.data_root, self.visual, self.image_size, self.context)
        result.update({"target": torch.tensor(s["target"], dtype=torch.float32),
                       "valid": torch.tensor(s["valid"], dtype=torch.bool),
                       "sample_id": s["sample_id"]})
        return result


def assert_disjoint(datasets):
    for i, left in enumerate(datasets):
        for right in datasets[i + 1:]:
            left_groups = {s["recording_group"] for s in left.samples}
            right_groups = {s["recording_group"] for s in right.samples}
            if left_groups & right_groups:
                raise ValueError(f"Recording leakage across splits: {left_groups & right_groups}")
