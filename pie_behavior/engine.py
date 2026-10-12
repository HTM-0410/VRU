"""Masked training loss and held-out metrics; no invalid-label gradients."""

import random

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F

from . import HEADS


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss(logits, target, valid):
    # Select valid entries before BCE. Even an invalid target NaN cannot
    # contaminate the loss; unsupported heads have exactly zero gradient.
    terms = [F.binary_cross_entropy_with_logits(logits[:, i][valid[:, i]], target[:, i][valid[:, i]])
             for i in range(len(HEADS)) if valid[:, i].any()]
    if not terms:
        return logits.sum() * 0
    return torch.stack(terms).mean()


def summarize(target, probability, valid):
    result = {}
    losses = []
    for i, name in enumerate(HEADS):
        y, p = target[valid[:, i], i], probability[valid[:, i], i]
        if not len(y):
            result[name] = {"support": 0, "status": "NO_VALID_LABELS"}
            continue
        pred = p >= 0.5
        positive = y.astype(bool)
        tp, fp = int(np.sum(pred & positive)), int(np.sum(pred & ~positive))
        fn, tn = int(np.sum(~pred & positive)), int(np.sum(~pred & ~positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        both = len(np.unique(y)) == 2
        clip = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
        bce = float(np.mean(-y * np.log(clip) - (1 - y) * np.log1p(-clip)))
        losses.append(bce)
        result[name] = {
            "support": len(y), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
            "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
            "f1": f1, "macro_f1": (f1 + f1_negative) / 2 if both else None,
            "auroc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "bce": bce, "brier": float(np.mean((p - y) ** 2)),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "status": "OK" if both else "SINGLE_CLASS",
        }
    result["mean_bce"] = float(np.mean(losses)) if losses else None
    result["threshold"] = 0.5
    result["probabilities_calibrated"] = False
    return result


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    ys, ps, masks, sample_ids = [], [], [], []
    for batch in loader:
        x, y, valid = (batch[k].to(device) for k in ("geometry", "target", "valid"))
        images = batch.get("images")
        if images is not None:
            images = images.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x, images)
            loss = masked_loss(logits, y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; stop before saving a bad checkpoint")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        ys.append(y.detach().cpu().numpy())
        ps.append(logits.detach().sigmoid().cpu().numpy())
        masks.append(valid.cpu().numpy())
        sample_ids.extend(batch["sample_id"])
    if not ys:
        raise ValueError("Empty loader")
    y, p, valid = np.concatenate(ys), np.concatenate(ps), np.concatenate(masks)
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite probabilities")
    return summarize(y, p, valid), (sample_ids, y, p, valid)


def majority_baseline(train_dataset, eval_dataset):
    train_y = np.array([s["target"] for s in train_dataset.samples], dtype=np.float32)
    train_m = np.array([s["valid"] for s in train_dataset.samples], dtype=bool)
    prevalence = np.array([train_y[train_m[:, i], i].mean() if train_m[:, i].any() else 0.5 for i in range(len(HEADS))])
    y = np.array([s["target"] for s in eval_dataset.samples], dtype=np.float32)
    m = np.array([s["valid"] for s in eval_dataset.samples], dtype=bool)
    return {"train_prevalence": dict(zip(HEADS, prevalence.tolist())),
            "metrics": summarize(y, np.broadcast_to(prevalence, y.shape), m)}
