"""Evaluate a selected checkpoint without selecting thresholds on test data."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import HEADS
from .data import BehaviorDataset, assert_disjoint, read_manifest
from .engine import majority_baseline, run_epoch
from .prepare import sha256
from .train import load_model, resolve_device, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    torch.set_num_threads(4)
    read_manifest(args.manifest_dir)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = checkpoint["config"]
    if config["manifest_sha256"] != sha256(args.manifest_dir / "manifest.json"):
        raise ValueError("Checkpoint belongs to a different dataset build")
    dataset = BehaviorDataset(args.manifest_dir, args.split, args.data_root,
                              config["model"]["backbone"] != "none", config["image_size"], config["context"])
    train = BehaviorDataset(args.manifest_dir, "train", args.data_root, visual=False)
    assert_disjoint([train, dataset])
    args.out.mkdir(parents=True, exist_ok=False)
    metrics, (ids, y, probability, valid) = run_epoch(model, DataLoader(dataset, args.batch_size), device)
    write_json(args.out / "metrics.json", {"split": args.split, "checkpoint_sha256": sha256(args.checkpoint),
                                           "epoch": checkpoint["epoch"], "metrics": metrics,
                                           "evaluation_unit": "overlapping_window"})
    write_json(args.out / "majority_baseline.json", majority_baseline(train, dataset))
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for sample_id, target, pred, mask in zip(ids, y, probability, valid):
            f.write(json.dumps({"sample_id": sample_id, "probabilities": dict(zip(HEADS, pred.tolist())),
                                "targets": {h: int(t) if m else None for h, t, m in zip(HEADS, target, mask)}}) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
