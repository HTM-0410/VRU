"""Train current PIE behavior. Test data is evaluated only by evaluate.py."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import HEADS
from .data import BehaviorDataset, assert_disjoint, read_manifest
from .engine import majority_baseline, run_epoch, seed_everything
from .model import BehaviorModel
from .prepare import sha256


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_model(checkpoint, device):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if checkpoint.get("heads") != list(HEADS):
        raise ValueError("Checkpoint head order mismatch")
    # Loading a checkpoint never triggers an ImageNet download.
    model = BehaviorModel(**{**config["model"], "pretrained": False})
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="New run directory")
    p.add_argument("--backbone", choices=("resnet18", "tiny", "none"), default="resnet18")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--context", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0, help="0 is the safe Windows starter setting")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.hidden_size, args.threads) < 1 or args.lr <= 0 or args.workers < 0:
        p.error("epochs/batch/patience/hidden-size/threads/lr must be positive; workers >= 0")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    manifest = read_manifest(args.manifest_dir)
    visual = args.backbone != "none"
    train = BehaviorDataset(args.manifest_dir, "train", args.data_root, visual, args.image_size, args.context)
    val = BehaviorDataset(args.manifest_dir, "val", args.data_root, visual, args.image_size, args.context)
    assert_disjoint([train, val])
    for i, head in enumerate(HEADS):
        labels = {s["target"][i] for s in train.samples if s["valid"][i]}
        if labels != {0, 1}:
            raise ValueError(f"Train head {head} needs both classes; observed {labels}")
    args.out.mkdir(parents=True, exist_ok=False)
    config = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": {"backbone": args.backbone, "pretrained": not args.no_pretrained and args.backbone == "resnet18",
                  "freeze_backbone": not args.unfreeze_backbone, "hidden_size": args.hidden_size, "dropout": 0.2},
        "image_size": args.image_size, "context": args.context, "history": manifest["history"],
        "history_time_offsets_s": [(i - manifest["history"] + 1) * manifest["frame_step"] / manifest["fps"] for i in range(manifest["history"])],
        "manifest_sha256": sha256(args.manifest_dir / "manifest.json"),
        "device": str(device), "torch_version": str(torch.__version__),
        "heads": list(HEADS), "label_semantics": manifest["task"],
        "evaluation_unit": "overlapping_window", "probabilities_calibrated": False,
    }
    write_json(args.out / "config.json", config)
    write_json(args.out / "baseline_val.json", majority_baseline(train, val))
    model = BehaviorModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val, args.batch_size, shuffle=False, num_workers=args.workers)
    best, bad_epochs, best_epoch = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, device, optimizer)
        val_metrics, _ = run_epoch(model, val_loader, device)
        score = val_metrics["mean_bce"]
        if score is None:
            raise ValueError("Validation split has no valid supervision")
        improved = score < best
        if improved:
            best, bad_epochs, best_epoch = score, 0, epoch
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        with (args.out / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        scheduler.step()
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val_bce": best,
                   "config": config, "heads": list(HEADS)}
        torch.save(payload, args.out / "last.pt")
        if improved:
            torch.save(payload, args.out / "best.pt")
            write_json(args.out / "best_val_metrics.json", val_metrics)
        print(f"epoch={epoch} train_bce={train_metrics['mean_bce']:.5f} val_bce={score:.5f} best={best:.5f}", flush=True)
        if bad_epochs >= args.patience:
            break
    write_json(args.out / "summary.json", {"status": "TRAIN_COMPLETED", "epochs_run": epoch,
                                             "best_epoch": best_epoch, "best_val_bce": best,
                                             "test_evaluated": False})


if __name__ == "__main__":
    main()
