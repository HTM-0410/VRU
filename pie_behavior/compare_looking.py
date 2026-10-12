"""Paired pilot: train looking recognition with 128/Q90 versus 224/Q95 crops.

Uses one training recording and one different validation recording, fixed epochs
and three paired seeds. It is not an official full-PIE benchmark. The frozen
ResNet18 feature cache is equivalent to running that backbone in each batch.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import geometry
from .engine import seed_everything, summarize
from .model import BehaviorModel
from .prepare import SPLITS, iter_windows, parse_tracks, sha256
from collections import Counter


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def samples_for(data_root, set_id, video_id):
    width, height, tracks = parse_tracks(data_root / "annotations" / set_id / f"{video_id}_annt.xml")
    samples = []
    stats = Counter()
    for track_id, records in tracks:
        for s in iter_windows(records, 16, 3, 15, 30, stats):
            if not s["valid"][1]:
                continue
            s.update({"sample_id": f"{set_id}/{video_id}/{track_id}/{s['frames'][-1]}",
                      "track_id": track_id, "set_id": set_id, "video_id": video_id,
                      "width": width, "height": height})
            samples.append(s)
    if not samples or set(s["target"][1] for s in samples) != {0, 1}:
        raise ValueError(f"Both looking classes are required: {set_id}/{video_id}")
    return samples, dict(stats)


def features_for(crop_root, samples, cnn, device, out, batch_size=64):
    report = json.loads((crop_root / "report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (crop_root / "crops.jsonl").read_text(encoding="utf-8").splitlines()]
    lookup = {(r["track_id"], r["frame"]): r for r in rows}
    keys = sorted({(s["track_id"], frame) for s in samples for frame in s["frames"]})
    positions = {key: i for i, key in enumerate(keys)}
    vectors = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for start in range(0, len(keys), batch_size):
        inputs = []
        for key in keys[start:start + batch_size]:
            row = lookup[key]
            with Image.open(crop_root / row["image"]) as image:
                image.load()
                if image.size != (report["image_size"], report["image_size"]):
                    raise ValueError("Crop cache dimensions disagree with report")
                arr = np.array(image.convert("RGB"), dtype=np.float32) / 255
            arr = (arr - mean) / std
            inputs.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))
        with torch.inference_mode():
            output = cnn(torch.stack(inputs).to(device)).cpu()
        vectors.append(output)
        if start % (batch_size * 10) == 0:
            print(f"features {crop_root.name}: {min(start+batch_size,len(keys))}/{len(keys)}", flush=True)
    vectors = torch.cat(vectors)
    if not torch.isfinite(vectors).all():
        raise FloatingPointError("Nonfinite visual features")
    indexes = torch.tensor([[positions[(s["track_id"], f)] for f in s["frames"]] for s in samples])
    result = {
        "visual": vectors[indexes].clone(),
        "geometry": torch.from_numpy(np.stack([geometry(s) for s in samples])),
        "target": torch.tensor([s["target"][1] for s in samples], dtype=torch.float32),
        "ids": [s["sample_id"] for s in samples],
    }
    torch.save(result, out)
    return result, {"crops_used": len(keys), "image_size": report["image_size"],
                    "jpeg_quality": report["jpeg_quality"], "metadata_sha256": sha256(crop_root / "crops.jsonl"),
                    "source_video_sha256": report["video_sha256"], "source_xml_sha256": report["xml_sha256"]}


def forward_cached(model, geometry_input, visual_input):
    features = torch.cat((visual_input, model.geometry_encoder(geometry_input)), dim=-1)
    _, state = model.gru(features)
    return model.head(state[-1])[:, 1]


def looking_metrics(target, probability):
    y = np.zeros((len(target), 3), dtype=np.float32)
    p = np.zeros_like(y)
    mask = np.zeros_like(y, dtype=bool)
    y[:, 1], p[:, 1], mask[:, 1] = target, probability, True
    return summarize(y, p, mask)["looking"]


def fit(train, val, backbone_state, seed, epochs, device, out):
    seed_everything(seed)
    model = BehaviorModel(pretrained=False, freeze_backbone=True).to(device)
    model.cnn.load_state_dict(backbone_state)
    # Verify that paired variants start from the same complete state.
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    initial_hash = digest.hexdigest()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    x, v, y = (train[k].to(device) for k in ("geometry", "visual", "target"))
    vx, vv = (val[k].to(device) for k in ("geometry", "visual"))
    logs = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        total = 0.0
        for offset in range(0, len(y), 8):
            index = order[offset:offset + 8].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_cached(model, x[index], v[index])
            loss = F.binary_cross_entropy_with_logits(logits, y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite looking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(index)
        logs.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_bce": total / len(y)})
        scheduler.step()
        if epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch} train_bce={total/len(y):.4f}", flush=True)
    model.eval()
    with torch.inference_mode():
        probability = torch.cat([forward_cached(model, vx[i:i+32], vv[i:i+32]).sigmoid().cpu()
                                 for i in range(0, len(vx), 32)]).numpy()
    target = val["target"].numpy()
    metrics = looking_metrics(target, probability)
    out.mkdir()
    save_json(out / "history.json", logs)
    save_json(out / "metrics.json", metrics)
    save_json(out / "predictions.json", [{"sample_id": s, "target": int(t), "probability": float(p)}
                                           for s, t, p in zip(val["ids"], target, probability)])
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "seed": seed, "epoch": epochs,
                "task": "looking_only_current_recognition", "cnn_frozen": True,
                "warning": "Only head index 1 trained; walking/crossing logits are not valid predictions."}, out / "last.pt")
    return metrics, probability, {"initial_state_sha256": initial_hash, "train_elapsed_s": time.perf_counter() - start}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-low", type=Path, required=True)
    p.add_argument("--train-high", type=Path, required=True)
    p.add_argument("--val-low", type=Path, required=True)
    p.add_argument("--val-high", type=Path, required=True)
    p.add_argument("--train-set", default="set01")
    p.add_argument("--train-video", default="video_0001")
    p.add_argument("--val-set", default="set06")
    p.add_argument("--val-video", default="video_0009")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    # Use full FP32 convolution and matrix products. Disabling TF32 reduces
    # batch-size-dependent rounding between cached and live CNN inference.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.epochs < 1 or not torch.cuda.is_available():
        raise ValueError("Positive epochs and CUDA required for this paired GPU pilot")
    if SPLITS[args.train_set] != "train" or SPLITS[args.val_set] != "val" or args.train_set == args.val_set:
        raise ValueError("Expected independent train/validation sets")
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    train_samples, train_filter = samples_for(args.data_root, args.train_set, args.train_video)
    val_samples, val_filter = samples_for(args.data_root, args.val_set, args.val_video)
    protocol = {
        "task": "looking_only_current_recognition", "validation_only_pilot": True,
        "train_recording": f"{args.train_set}/{args.train_video}", "val_recording": f"{args.val_set}/{args.val_video}",
        "train_windows": len(train_samples), "train_positive": sum(s["target"][1] for s in train_samples),
        "val_windows": len(val_samples), "val_positive": sum(s["target"][1] for s in val_samples),
        "train_filter": train_filter, "val_filter": val_filter,
        "seeds": [42, 43, 44], "epochs": args.epochs, "checkpoint_selection": "fixed_final_epoch_no_val_selection",
        "threshold": 0.5, "threshold_tuning": False, "loss": "unweighted_looking_BCE",
        "optimizer": "AdamW lr=0.001 weight_decay=0.0001", "scheduler": "cosine", "batch_size": 8,
        "hidden_size": 128, "dropout": 0.2, "backbone": "ResNet18 ImageNet1K_V1 frozen eval",
        "feature_cache": "exact frozen eval backbone output; features and geometry causal only",
        "crop_variants": {"low": "128x128 JPEG Q90", "high": "224x224 JPEG Q95"},
        "precision": "FP32, cuDNN and matmul TF32 disabled", "torch": str(torch.__version__), "GPU": torch.cuda.get_device_name(0),
        "test_split_used": False, "probabilities_calibrated": False,
        "limitations": ["One train and one validation video", "Overlapping windows", "Only looking head is trained", "No full PIE test or deployment evaluation", "Changes resolution and JPEG quality together"],
    }
    save_json(args.out / "protocol.json", protocol)
    for name, samples in (("train", train_samples), ("val", val_samples)):
        save_json(args.out / f"{name}_samples.json", samples)
    seed_everything(42)
    base = BehaviorModel(pretrained=True, freeze_backbone=True).to(device).eval()
    backbone_state = {k: v.detach().cpu().clone() for k, v in base.cnn.state_dict().items()}
    cached, provenance = {}, {}
    for name, crop_path, samples in (("train_low", args.train_low, train_samples),
                                      ("train_high", args.train_high, train_samples),
                                      ("val_low", args.val_low, val_samples),
                                      ("val_high", args.val_high, val_samples)):
        cached[name], provenance[name] = features_for(crop_path, samples, base.cnn, device, args.out / f"{name}_features.pt")
    for split in ("train", "val"):
        low, high = cached[split + "_low"], cached[split + "_high"]
        assert low["ids"] == high["ids"]
        assert torch.equal(low["target"], high["target"])
        assert torch.equal(low["geometry"], high["geometry"])
        assert provenance[split + "_low"]["source_video_sha256"] == provenance[split + "_high"]["source_video_sha256"]
    for low_path, high_path in ((args.train_low, args.train_high), (args.val_low, args.val_high)):
        low_report = json.loads((low_path / "report.json").read_text(encoding="utf-8"))
        high_report = json.loads((high_path / "report.json").read_text(encoding="utf-8"))
        for name in ("source_xml", "video_sha256", "xml_sha256", "context_per_side", "label_counts", "exported_crops"):
            assert low_report[name] == high_report[name], f"Paired source/crop setting differs: {name}"
        assert (low_report["image_size"], low_report["jpeg_quality"]) == (128, 90)
        assert (high_report["image_size"], high_report["jpeg_quality"]) == (224, 95)
    save_json(args.out / "cache_provenance.json", provenance)
    # Verify cached forward agrees with the original model.forward, including
    # feature concatenation and final hidden-state indexing.
    with torch.inference_mode():
        test_images = torch.randn(2, 16, 3, 128, 128, device=device)
        test_geometry = torch.randn(2, 16, 9, device=device)
        features = base.cnn(test_images.reshape(32, 3, 128, 128)).reshape(2, 16, 512)
        torch.testing.assert_close(forward_cached(base, test_geometry, features), base(test_geometry, test_images)[:, 1])
    del base
    results, probabilities = {}, {}
    for seed in protocol["seeds"]:
        hashes = []
        for variant in ("low", "high"):
            metrics, prob, meta = fit(cached["train_" + variant], cached["val_" + variant], backbone_state,
                                      seed, args.epochs, device, args.out / f"{variant}_seed{seed}")
            results[f"{variant}_seed{seed}"] = {"metrics": metrics, **meta}
            probabilities[f"{variant}_seed{seed}"] = prob
            hashes.append(meta["initial_state_sha256"])
            print(variant, seed, json.dumps(metrics), flush=True)
        assert hashes[0] == hashes[1], "Paired initialization differs"
    metric_names = ("accuracy", "precision", "recall", "f1", "macro_f1", "auroc", "average_precision", "bce", "brier")
    means = {variant: {m: float(np.mean([results[f"{variant}_seed{s}"]["metrics"][m] for s in protocol["seeds"]]))
                       for m in metric_names} for variant in ("low", "high")}
    deltas = {m: means["high"][m] - means["low"][m] for m in metric_names}
    prevalence = cached["train_low"]["target"].mean().item()
    baseline = looking_metrics(cached["val_low"]["target"].numpy(), np.full(len(val_samples), prevalence))
    # Paired cluster bootstrap over validation tracks; conditional on this
    # one recording and these three seeds, not uncertainty across recordings.
    track_ids = np.array([s["track_id"] for s in val_samples])
    groups = sorted(set(track_ids))
    indexes = {t: np.where(track_ids == t)[0] for t in groups}
    rng = np.random.default_rng(20261012)
    boots = {m: [] for m in ("f1", "accuracy", "average_precision")}
    target = cached["val_low"]["target"].numpy()
    for _ in range(1000):
        ids = np.concatenate([indexes[t] for t in rng.choice(groups, len(groups), replace=True)])
        if len(np.unique(target[ids])) < 2:
            continue
        seed_deltas = {m: [] for m in boots}
        for seed in protocol["seeds"]:
            low = looking_metrics(target[ids], probabilities[f"low_seed{seed}"][ids])
            high = looking_metrics(target[ids], probabilities[f"high_seed{seed}"][ids])
            for m in boots:
                seed_deltas[m].append(high[m] - low[m])
        for m in boots:
            boots[m].append(float(np.mean(seed_deltas[m])))
    intervals = {m: np.percentile(values, [2.5, 97.5]).tolist() for m, values in boots.items()}
    result = {"status": "COMPLETED_PAIRED_LOOKING_PILOT", "protocol": protocol,
              "per_run": results, "mean_metrics": means, "high_minus_low": deltas,
              "majority_baseline": baseline, "paired_track_bootstrap_95CI": intervals,
              "bootstrap_scope": "Validation tracks within one recording, conditional on trained seeds",
              "same_samples_targets_geometry": True, "paired_initializations_equal": True,
              "cached_forward_equivalence": "PASS", "validation_tracks": len(groups),
              "primary_metric": "positive looking F1 at fixed threshold 0.5",
              "validation_positive_tracks": len({s['track_id'] for s in val_samples if s['target'][1] == 1})}
    save_json(args.out / "results.json", result)
    print(json.dumps({k: result[k] for k in ("status", "mean_metrics", "high_minus_low", "paired_track_bootstrap_95CI")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
