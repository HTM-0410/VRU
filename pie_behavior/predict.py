"""Infer one observed pedestrian history from JSON; behavior labels unnecessary."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import HEADS
from .data import make_inputs
from .train import load_model, resolve_device, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="JSON history in the sample schema; targets not required")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = checkpoint["config"]
    sample = json.loads(args.input.read_text(encoding="utf-8"))
    if (len(sample["boxes"]) != config["history"] or
            len(sample["time_offsets_s"]) != config["history"] or
            not np.allclose(sample["time_offsets_s"], config["history_time_offsets_s"], atol=1e-5)):
        raise ValueError("Inference history must match the training time grid")
    inputs = make_inputs(sample, args.data_root, config["model"]["backbone"] != "none", config["image_size"], config["context"])
    with torch.inference_mode():
        probability = model(inputs["geometry"][None].to(device),
                            inputs["images"][None].to(device) if "images" in inputs else None).sigmoid()[0].cpu().tolist()
    if not np.isfinite(probability).all():
        raise FloatingPointError("Non-finite inference probabilities")
    result = {"sample_id": sample["sample_id"], "task": "current_behavior_recognition",
              "probabilities": dict(zip(HEADS, probability)), "probabilities_calibrated": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
