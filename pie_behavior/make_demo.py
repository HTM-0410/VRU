"""Generate tiny SYNTHETIC PIE-shaped data for software smoke tests only."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


def make_demo(out, frames=90):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for set_id in ("set01", "set03", "set05"):
        folder = out / "annotations" / set_id
        folder.mkdir(parents=True)
        image_folder = out / "images" / set_id / "video_0001"
        image_folder.mkdir(parents=True)
        root = ET.Element("annotations")
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "size").text = str(frames)
        size = ET.SubElement(task, "original_size")
        ET.SubElement(size, "width").text = "192"
        ET.SubElement(size, "height").text = "128"
        states = [(0, 0, 0), (1, 0, 1), (0, 1, -1), (1, 1, 0)]
        colors = [(60, 60, 60), (220, 40, 40), (40, 220, 40), (40, 40, 220)]
        for i, (walking, looking, crossing) in enumerate(states):
            track = ET.SubElement(root, "track", label="pedestrian")
            for frame in range(frames):
                x = 8 + i * 44 + walking * (frame % 8)
                box = ET.SubElement(track, "box", frame=str(frame), outside="0", occluded="0",
                                    xtl=str(x), ytl="30", xbr=str(x + 22), ybr="104")
                attributes = {
                    "id": f"{int(set_id[-2:])}_1_{i + 1}", "occlusion": "none",
                    "action": "walking" if walking else "standing",
                    "look": "looking" if looking else "not-looking",
                    "cross": {0: "not-crossing", 1: "crossing", -1: "crossing-irrelevant"}[crossing],
                }
                for key, value in attributes.items():
                    ET.SubElement(box, "attribute", name=key).text = value
        ET.ElementTree(root).write(folder / "video_0001_annt.xml", encoding="utf-8", xml_declaration=True)
        for frame in range(frames):
            image = Image.new("RGB", (192, 128), (110, 110, 110))
            draw = ImageDraw.Draw(image)
            for i, (walking, _, _) in enumerate(states):
                x = 8 + i * 44 + walking * (frame % 8)
                draw.rectangle((x, 30, x + 22, 104), fill=colors[i])
            image.save(image_folder / f"{frame:05d}.png")
    (out / "SYNTHETIC_ONLY.txt").write_text(
        "Synthetic schema and training smoke fixture. No real PIE accuracy may be inferred.\n", encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(make_demo(args.out))


if __name__ == "__main__":
    main()
