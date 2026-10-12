"""Extract only requested manifest frames from PIE MP4 (zero-based frame IDs)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .data import read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    args = p.parse_args()
    import cv2
    manifest = read_manifest(args.manifest_dir)
    needed = defaultdict(set)
    dimensions = {}
    for split in ("train", "val", "test"):
        with (args.manifest_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                key = (s["set_id"], s["video_id"])
                needed[key].update(s["frames"])
                dimensions[key] = (s["width"], s["height"])
    for (set_id, video_id), frames in sorted(needed.items()):
        out = args.data_root / "images" / set_id / video_id
        missing = {f for f in frames if not (out / f"{f:05d}.png").exists()}
        if not missing:
            continue
        video = args.data_root / "PIE_clips" / set_id / f"{video_id}.mp4"
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise FileNotFoundError(f"Cannot open {video}")
            measured_fps = capture.get(cv2.CAP_PROP_FPS)
            if abs(measured_fps - manifest["fps"]) > 0.1:
                raise ValueError(f"Video FPS {measured_fps} != manifest FPS {manifest['fps']}")
            out.mkdir(parents=True, exist_ok=True)
            remaining = set(missing)
            for frame in range(max(missing) + 1):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Video ended before requested frame {frame}: {video}")
                if frame not in missing:
                    continue
                if (image.shape[1], image.shape[0]) != dimensions[(set_id, video_id)]:
                    raise ValueError(f"Video dimensions disagree with annotation: {video}")
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise RuntimeError(f"Cannot encode {video}, frame {frame}")
                # tofile supports Unicode Windows paths; existing files are skipped.
                encoded.tofile(str(out / f"{frame:05d}.png"))
                remaining.remove(frame)
            if remaining:
                raise RuntimeError(f"Unextracted frames: {len(remaining)}")
        finally:
            capture.release()
        print(f"{set_id}/{video_id}: wrote {len(missing)} frames", flush=True)


if __name__ == "__main__":
    main()
