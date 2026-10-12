"""Audit original PIE annotation ZIP without extracting it or training a model."""

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

from . import HEADS
from .prepare import SPLITS, iter_windows, parse_tracks, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    counts = {split: Counter() for split in ("train", "val", "test")}
    report = {"archive_sha256": sha256(args.zip), "files": [], "training_run": False,
              "window_config": {"history": 16, "frame_step": 3, "window_step": 15, "fps": 30}}
    with zipfile.ZipFile(args.zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith("_annt.xml"):
                continue
            split = SPLITS[Path(name).parent.name]
            with archive.open(name) as f:
                width, height, tracks = parse_tracks(f)
            counts[split]["source_tracks"] += len(tracks)
            for _, records in tracks:
                for sample in iter_windows(records, 16, 3, 15, 30, counts[split]):
                    counts[split]["samples"] += 1
                    for head, y, valid in zip(HEADS, sample["target"], sample["valid"]):
                        counts[split][f"{head}_{y}" if valid else f"{head}_ignored"] += 1
            report["files"].append({"path": name, "split": split, "tracks": len(tracks),
                                    "width": width, "height": height})
    report.update({"status": "PASS_REAL_ANNOTATION_PARSER_ONLY",
                   "counts": {split: dict(c) for split, c in counts.items()}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(report["files"]), "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
