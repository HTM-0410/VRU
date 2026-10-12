"""Export one real PIE video's annotated pedestrian crops and measured storage.

This is a crop-cache exporter. The existing full-frame training loader does not
consume this format yet. Original XML labels and frame indices are retained.
"""

import argparse
from collections import Counter, defaultdict
import io
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .prepare import iter_windows, parse_tracks, sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def crop_bounds(box, width, height, context):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * context, (y2 - y1) * context
    return (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy)))


def font(size):
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def export(data_root, set_id, video_id, out, image_size=128, quality=90, context=0.25):
    if image_size < 32 or not 1 <= quality <= 100 or not math.isfinite(context) or context < 0:
        raise ValueError("Require size >= 32, JPEG quality 1..100 and finite nonnegative context")
    if any(Path(v).name != v or v in (".", "..") for v in (set_id, video_id)):
        raise ValueError("set_id/video_id must be plain directory names")
    data_root, out = Path(data_root).resolve(), Path(out).resolve()
    video = data_root / "PIE_clips" / set_id / f"{video_id}.mp4"
    annotation = data_root / "annotations" / set_id / f"{video_id}_annt.xml"
    width, height, tracks = parse_tracks(annotation)
    # Independent source count ensures no box disappears without being reported.
    tree = ET.parse(annotation)
    source_visible_boxes = sum(1 for t in tree.getroot().findall("track") if t.get("label") == "pedestrian"
                               for b in t.findall("box") if b.get("outside", "0") != "1")
    expected_frame_count = int(tree.findtext("./meta/task/size"))
    del tree
    by_frame = defaultdict(list)
    window_stats = Counter()
    train_crop_keys = set()
    preview_keys = []
    for track_id, records in tracks:
        if Path(track_id).name != track_id:
            raise ValueError("Unsafe track id")
        for frame, record in records.items():
            by_frame[frame].append((track_id, record))
        if records:
            frames = sorted(records)
            preview_keys.append((track_id, frames[len(frames) // 2]))
        for sample in iter_windows(records, 16, 3, 15, 30, window_stats):
            window_stats["kept_windows"] += 1
            train_crop_keys.update((track_id, frame) for frame in sample["frames"])
    if not by_frame:
        raise ValueError("No valid annotated pedestrian boxes")
    overlay_frames = set(int(v) for v in np.quantile(sorted(by_frame), np.linspace(0, 1, 6)))
    # Quantiles can fall between frame IDs; snap to the nearest observed frame.
    frame_array = np.array(sorted(by_frame))
    overlay_frames = {int(frame_array[np.argmin(abs(frame_array - value))]) for value in overlay_frames}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        video_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if video_size != (width, height) or abs(fps - 30) > 0.1:
            raise ValueError(f"Video/annotation mismatch: size={video_size}, fps={fps}")
        out.mkdir(parents=True, exist_ok=False)
        crops_dir = out / "crops" / set_id / video_id
        crops_dir.mkdir(parents=True)
        for track_id, _ in tracks:
            (crops_dir / track_id).mkdir()
        preview_dir = out / "previews"
        preview_dir.mkdir()
        source_hash = sha256(video)
        xml_hash = sha256(annotation)
        stats = {k: Counter() for k in ("action", "look", "cross", "occlusion")}
        sizes = []
        train_bytes = 0
        crop_paths = {}
        overlay_images = []
        decoded_count = 0
        start = time.perf_counter()
        last_print = start
        with (out / "crops.jsonl").open("w", encoding="utf-8") as manifest:
            for frame in range(expected_frame_count):
                if not capture.grab():
                    raise RuntimeError(f"Video ended at {frame}, annotation expects {expected_frame_count}")
                decoded_count += 1
                if frame not in by_frame:
                    continue
                ok, bgr = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot decode required frame {frame}")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                for track_id, record in by_frame[frame]:
                    bounds = crop_bounds(record["bbox"], width, height, context)
                    cropped = image.crop(bounds).resize((image_size, image_size), Image.Resampling.BILINEAR)
                    relative = Path("crops") / set_id / video_id / track_id / f"{frame:05d}.jpg"
                    buffer = io.BytesIO()
                    cropped.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=2)
                    content = buffer.getvalue()
                    (out / relative).write_bytes(content)
                    sizes.append(len(content))
                    key = (track_id, frame)
                    if key in train_crop_keys:
                        train_bytes += len(content)
                    if key in preview_keys:
                        crop_paths[key] = (relative, record["attrs"])
                    for field in stats:
                        stats[field][record["attrs"].get(field, "MISSING")] += 1
                    manifest.write(json.dumps({
                        "sample_key": f"{set_id}/{video_id}/{track_id}/{frame}",
                        "set_id": set_id, "video_id": video_id, "track_id": track_id,
                        "frame": frame, "timestamp_s": frame / fps,
                        "image": relative.as_posix(), "source_image_size": [width, height],
                        "bbox_xyxy": record["bbox"], "crop_xyxy": list(bounds),
                        "crop_size": [image_size, image_size], "labels": record["attrs"],
                        "in_current_v1_windows": key in train_crop_keys,
                    }, ensure_ascii=False) + "\n")
                if frame in overlay_frames:
                    overlay = image.copy()
                    draw = ImageDraw.Draw(overlay)
                    for track_id, record in by_frame[frame]:
                        box = record["bbox"]
                        draw.rectangle(box, outline=(0, 255, 60), width=4)
                        draw.text((box[0], max(0, box[1] - 26)), track_id, font=font(24), fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
                    overlay.thumbnail((960, 540))
                    tile = Image.new("RGB", (960, 576), "#111827")
                    tile.paste(overlay, (0, 36))
                    ImageDraw.Draw(tile).text((12, 8), f"{set_id}/{video_id}  frame={frame}  time={frame/fps:.2f}s", font=font(20), fill="white")
                    overlay_images.append(tile)
                now = time.perf_counter()
                if now - last_print >= 10:
                    print(json.dumps({"frame": frame, "crops_written": len(sizes), "elapsed_s": round(now - start, 1)}), flush=True)
                    last_print = now
            if capture.grab():
                raise RuntimeError("Video has extra frames beyond annotation task size")
        elapsed = time.perf_counter() - start
    finally:
        capture.release()
    expected_crops = sum(len(r) for _, r in tracks)
    if len(sizes) != expected_crops:
        raise AssertionError(f"Crop count mismatch {len(sizes)} != {expected_crops}")
    # Reopen every written JPEG and check the pixels decode and dimensions match.
    validation_count = 0
    for image_path in crops_dir.rglob("*.jpg"):
        with Image.open(image_path) as image:
            image.load()
            if image.size != (image_size, image_size) or image.mode != "RGB":
                raise AssertionError(f"Invalid output image: {image_path}")
        validation_count += 1
    if validation_count != len(sizes):
        raise AssertionError("Written JPEG count does not match manifest")
    sheet = Image.new("RGB", (7 * 176, 3 * 240), "#111827")
    draw = ImageDraw.Draw(sheet)
    for i, key in enumerate(preview_keys[:21]):
        path, attrs = crop_paths[key]
        x, y = (i % 7) * 176, (i // 7) * 240
        with Image.open(out / path) as image:
            sheet.paste(image.resize((128, 128)), (x + 24, y + 8))
        labels = f"{key[0]} f={key[1]}\n{attrs.get('action')}\n{attrs.get('look')}\n{attrs.get('cross')}\nocc: {attrs.get('occlusion')}"
        draw.multiline_text((x + 6, y + 143), labels, font=font(12), fill="white", spacing=2)
    sheet.save(preview_dir / "crop_contact_sheet.jpg", quality=95)
    overlays = Image.new("RGB", (1920, 576 * math.ceil(len(overlay_images) / 2)), "#111827")
    for i, tile in enumerate(overlay_images):
        overlays.paste(tile, ((i % 2) * 960, (i // 2) * 576))
    overlays.save(preview_dir / "bbox_alignment.jpg", quality=92)
    total_bytes = sum(sizes)
    metadata_bytes = (out / "crops.jsonl").stat().st_size
    source_unchanged = sha256(video) == source_hash and sha256(annotation) == xml_hash
    if not source_unchanged:
        raise AssertionError("Source changed during export")
    result = {
        "status": "PASS_EXPORT_AND_FULL_JPEG_DECODE", "source_video": str(video),
        "source_xml": str(annotation), "video_sha256": source_hash, "xml_sha256": xml_hash,
        "video_bytes": video.stat().st_size, "fps": fps, "video_dimensions": [width, height],
        "video_frames_checked": decoded_count, "video_duration_s": decoded_count / fps,
        "source_pedestrian_tracks": len(tracks), "source_nonoutside_boxes": source_visible_boxes,
        "invalid_geometry_boxes_skipped": source_visible_boxes - expected_crops,
        "exported_crops": len(sizes), "unique_frames_with_crops": len(by_frame),
        "jpeg_quality": quality, "jpeg_subsampling": "4:2:0", "jpeg_optimize": True,
        "image_size": image_size, "context_per_side": context, "crop_bytes": total_bytes,
        "metadata_bytes": metadata_bytes, "crop_plus_metadata_bytes": total_bytes + metadata_bytes,
        "mean_crop_bytes": total_bytes / len(sizes),
        "crop_bytes_percentiles": {str(p): float(np.percentile(sizes, p)) for p in [0, 50, 90, 99, 100]},
        "label_counts": {k: dict(v) for k, v in stats.items()},
        "current_v1_window_stats": dict(window_stats), "current_v1_unique_crops": len(train_crop_keys),
        "current_v1_unique_crop_bytes": train_bytes,
        "export_elapsed_s": elapsed, "verified_jpegs": validation_count, "source_unchanged": source_unchanged,
        "full_dataset_assumed_box_count": 738970,
        "projected_full_dataset_crop_GB": total_bytes / len(sizes) * 738970 / 1e9,
        "projected_full_dataset_crop_plus_metadata_GB": (total_bytes + metadata_bytes) / len(sizes) * 738970 / 1e9,
        "projection_limit": "One video only; different box sizes, scene content and label mix may change compression.",
        "current_loader_compatible": False,
        "loader_note": "Existing BehaviorDataset expects full-size PNG frames; crop-cache loader is a separate next step.",
        "training_run": False,
    }
    write_json(out / "report.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--set-id", default="set01")
    p.add_argument("--video-id", default="video_0001")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--context", type=float, default=0.25)
    export(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
