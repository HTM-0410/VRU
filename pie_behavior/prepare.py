"""Build causal windows from original PIE XML without modifying source files."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from . import HEADS, SCHEMA_VERSION

# Official PIE set split. Clips within a set can be consecutive recordings.
SPLITS = {"set01": "train", "set02": "train", "set04": "train",
          "set05": "val", "set06": "val", "set03": "test"}
LABELS = {
    "action": {"standing": 0, "walking": 1},
    "look": {"not-looking": 0, "looking": 1},
    "cross": {"not-crossing": 0, "crossing": 1},
}
FEATURE_NAMES = ["cx", "cy", "width", "height", "dcx_dt", "dcy_dt",
                 "dwidth_dt", "dheight_dt", "time_offset_s"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_tracks(path):
    root = ET.parse(path).getroot()
    width = int(root.findtext("./meta/task/original_size/width"))
    height = int(root.findtext("./meta/task/original_size/height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {path}")
    tracks = []
    ids = set()
    for track in root.findall("track"):
        if track.get("label") != "pedestrian":
            continue
        records = {}
        track_ids = set()
        for box in track.findall("box"):
            attrs = {a.get("name"): (a.text or "").strip() for a in box.findall("attribute")}
            if attrs.get("id"):
                track_ids.add(attrs["id"])
            if box.get("outside", "0") == "1":
                continue
            frame = int(box.get("frame"))
            if frame in records:
                raise ValueError(f"Duplicate frame in track: {path}, {frame}")
            bbox = [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            if (not all(math.isfinite(v) for v in bbox) or
                    bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or
                    min(bbox[2], width) <= max(bbox[0], 0) or
                    min(bbox[3], height) <= max(bbox[1], 0)):
                # Omitted records split windows rather than fabricating a box.
                continue
            records[frame] = {"bbox": bbox, "attrs": attrs}
        if len(track_ids) != 1:
            raise ValueError(f"Missing or inconsistent pedestrian id in {path}")
        track_id = next(iter(track_ids))
        if track_id in ids:
            raise ValueError(f"Repeated pedestrian id {track_id} in {path}")
        ids.add(track_id)
        tracks.append((track_id, records))
    return width, height, tracks


def iter_windows(records, history, frame_step, window_step, fps, stats):
    if not records:
        return
    first, last = min(records), max(records)
    span = (history - 1) * frame_step
    for t0 in range(first + span, last + 1, window_step):
        stats["candidate_windows"] += 1
        # Do not bridge gaps, outside-frame segments or invalid boxes, even
        # when the gap falls between two sampled history frames.
        if any(f not in records for f in range(t0 - span, t0 + 1)):
            stats["dropped_history_gap"] += 1
            continue
        frames = list(range(t0 - span, t0 + 1, frame_step))
        if any(records[f]["attrs"].get("occlusion") not in ("none", "part") for f in frames):
            stats["dropped_full_or_unknown_occlusion"] += 1
            continue
        attrs = records[t0]["attrs"]
        values = [LABELS[k].get(attrs.get(k), -1) for k in ("action", "look", "cross")]
        valid = [v != -1 for v in values]
        if not any(valid):
            stats["dropped_no_current_label"] += 1
            continue
        yield {
            "frames": frames,
            "boxes": [records[f]["bbox"] for f in frames],
            "time_offsets_s": [(f - t0) / fps for f in frames],
            "target": [max(0, v) for v in values],
            "valid": valid,
            "occlusion": [records[f]["attrs"]["occlusion"] for f in frames],
        }


def build(data_root, out, history=16, frame_step=3, window_step=15, fps=30.0):
    if history < 2 or min(frame_step, window_step) < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Require history >= 2, positive steps and finite positive FPS")
    data_root, out = Path(data_root).resolve(), Path(out).resolve()
    files = sorted((data_root / "annotations").glob("set*/*_annt.xml"))
    if not files:
        raise FileNotFoundError(f"No annotations/set*/*_annt.xml under {data_root}")
    # Every build is versioned by directory; no silent overwrite of old manifests.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION, "heads": list(HEADS),
        "task": "current_behavior_recognition_at_last_observed_frame",
        "source_root": str(data_root), "split_protocol": "official_PIE_sets",
        "set_to_split": SPLITS, "history": history, "frame_step": frame_step,
        "window_step": window_step, "fps": fps,
        "history_duration_s": (history - 1) * frame_step / fps,
        "feature_names": FEATURE_NAMES, "sources": [], "splits": {},
        "status": "building",
    }
    counts = {split: Counter() for split in ("train", "val", "test")}
    groups = {split: set() for split in counts}
    tracks = {split: set() for split in counts}
    with ExitStack() as stack:
        handles = {s: stack.enter_context((out / f"{s}.jsonl").open("w", encoding="utf-8")) for s in counts}
        for path in files:
            set_id, video_id = path.parent.name, path.name.removesuffix("_annt.xml")
            if set_id not in SPLITS:
                raise ValueError(f"Unknown set requires an explicit split policy: {set_id}")
            split = SPLITS[set_id]
            width, height, parsed = parse_tracks(path)
            manifest["sources"].append({"path": path.relative_to(data_root).as_posix(),
                                        "sha256": sha256(path), "split": split})
            counts[split]["source_tracks"] += len(parsed)
            for track_id, records in parsed:
                for sample in iter_windows(records, history, frame_step, window_step, fps, counts[split]):
                    sample.update({
                        "sample_id": f"{set_id}/{video_id}/{track_id}/{sample['frames'][-1]}",
                        "recording_group": set_id, "set_id": set_id, "video_id": video_id,
                        "track_id": track_id, "width": width, "height": height,
                        "images": [f"images/{set_id}/{video_id}/{f:05d}.png" for f in sample["frames"]],
                    })
                    handles[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                    counts[split]["samples"] += 1
                    for name, y, valid in zip(HEADS, sample["target"], sample["valid"]):
                        counts[split][f"{name}_{int(y)}" if valid else f"{name}_ignored"] += 1
                    groups[split].add(set_id)
                    tracks[split].add(f"{set_id}/{video_id}/{track_id}")
    for split in counts:
        manifest["splits"][split] = {**dict(counts[split]), "recording_groups": sorted(groups[split]),
                                      "unique_tracks": len(tracks[split]),
                                      "sha256": sha256(out / f"{split}.jsonl")}
    manifest["status"] = "ready"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="New directory for versioned manifests")
    p.add_argument("--history", type=int, default=16)
    p.add_argument("--frame-step", type=int, default=3)
    p.add_argument("--window-step", type=int, default=15)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()
    report = build(**vars(args))
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
