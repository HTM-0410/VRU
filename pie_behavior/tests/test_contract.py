from collections import Counter
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from pie_behavior.data import BehaviorDataset, geometry, read_manifest
from pie_behavior.engine import masked_loss, summarize
from pie_behavior.make_demo import make_demo
from pie_behavior.model import BehaviorModel
from pie_behavior.prepare import build, iter_windows, parse_tracks


@pytest.fixture
def demo(tmp_path):
    source = make_demo(tmp_path / "source")
    manifests = tmp_path / "manifests"
    build(source, manifests)
    return source, manifests


def test_official_split_and_history(demo):
    source, manifests = demo
    report = read_manifest(manifests)
    assert report["history_duration_s"] == 1.5
    groups = []
    for split in ("train", "val", "test"):
        dataset = BehaviorDataset(manifests, split, source, image_size=32)
        assert len(dataset) == 12
        sample = dataset[0]
        assert sample["images"].shape == (16, 3, 32, 32)
        assert sample["geometry"].shape == (16, 9)
        assert sample["geometry"][-1, -1] == 0
        groups.append({s["recording_group"] for s in dataset.samples})
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))


def test_irrelevant_is_masked_not_negative(demo):
    source, manifests = demo
    dataset = BehaviorDataset(manifests, "train", source, visual=False)
    irrelevant = [s for s in dataset.samples if s["track_id"].endswith("_3")]
    assert irrelevant
    assert all(s["valid"] == [True, True, False] for s in irrelevant)


def test_invalid_label_nan_has_zero_gradient():
    logits = torch.zeros((2, 3), requires_grad=True)
    y = torch.tensor([[1., float("nan"), 0.], [0., float("nan"), 1.]])
    mask = torch.tensor([[True, False, True], [True, False, True]])
    loss = masked_loss(logits, y, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2))
    assert torch.all(logits.grad[:, (0, 2)] != 0)


def test_all_invalid_has_no_gradient():
    logits = torch.randn(2, 3, requires_grad=True)
    loss = masked_loss(logits, torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_gap_between_sampled_frames_is_not_bridged(demo):
    source, _ = demo
    _, _, tracks = parse_tracks(source / "annotations/set01/video_0001_annt.xml")
    records = tracks[0][1]
    # Frame 1 is not a sampled frame on the [0,3,...,45] grid.
    del records[1]
    stats = Counter()
    windows = list(iter_windows(records, 16, 3, 15, 30, stats))
    assert stats["dropped_history_gap"] == 1
    assert windows[0]["frames"][-1] == 60


def test_future_annotations_cannot_change_inputs_or_current_targets(demo):
    source, _ = demo
    _, _, tracks = parse_tracks(source / "annotations/set01/video_0001_annt.xml")
    records = tracks[0][1]
    reference = next(iter_windows(records, 16, 3, 15, 30, Counter()))
    changed = copy.deepcopy(records)
    for frame, record in changed.items():
        if frame > 45:
            record["bbox"] = [0, 0, 100, 100]
            record["attrs"]["action"] = "walking"
    assert next(iter_windows(changed, 16, 3, 15, 30, Counter())) == reference


def test_full_occlusion_windows_are_reported(demo):
    source, _ = demo
    _, _, tracks = parse_tracks(source / "annotations/set01/video_0001_annt.xml")
    records = tracks[0][1]
    records[45]["attrs"]["occlusion"] = "full"
    stats = Counter()
    list(iter_windows(records, 16, 3, 15, 30, stats))
    assert stats["dropped_full_or_unknown_occlusion"] > 0


def test_hash_tampering_is_detected(demo):
    _, manifests = demo
    with (manifests / "train.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_manifest(manifests)


def test_source_and_previous_build_not_overwritten(demo):
    source, manifests = demo
    path = source / "annotations/set01/video_0001_annt.xml"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        build(source, manifests)
    assert path.read_bytes() == before


def test_metrics_ignore_invalid_and_handle_single_class():
    y = np.array([[1., 0., 1.], [0., 0., 0.]])
    p = np.array([[0.9, 0.1, 0.9], [0.1, 0.1, 0.9]])
    m = np.array([[True, True, False], [True, True, False]])
    report = summarize(y, p, m)
    assert report["walking"]["f1"] == 1
    assert report["looking"]["auroc"] is None
    assert report["crossing_now"]["support"] == 0


def test_no_behavior_label_enters_encoder(demo):
    source, manifests = demo
    dataset = BehaviorDataset(manifests, "train", source, visual=False)
    sample = dataset.samples[0]
    old = geometry(sample)
    sample["target"] = [1 - v for v in sample["target"]]
    sample["valid"] = [False] * 3
    np.testing.assert_array_equal(old, geometry(sample))


def test_frozen_resnet_batchnorm_and_no_download():
    model = BehaviorModel(backbone="resnet18", pretrained=False, freeze_backbone=True)
    model.train()
    assert not model.cnn.training
    assert model.gru.training
    assert all(not p.requires_grad for p in model.cnn.parameters())


def test_tiny_model_optimizer_updates_weights(demo):
    torch.set_num_threads(2)
    source, manifests = demo
    dataset = BehaviorDataset(manifests, "train", source, image_size=32)
    item = dataset[0]
    model = BehaviorModel(backbone="tiny", hidden_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.head[-1].weight.detach().clone()
    logits = model(item["geometry"][None], item["images"][None])
    masked_loss(logits, item["target"][None], item["valid"][None]).backward()
    optimizer.step()
    assert not torch.equal(before, model.head[-1].weight)


def test_extractor_preserves_zero_based_frame_alignment(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = make_demo(tmp_path / "source", frames=12)
    manifests = tmp_path / "manifests"
    build(source, manifests, history=2, frame_step=1, window_step=10)
    for set_id in ("set01", "set03", "set05"):
        clip_dir = source / "PIE_clips" / set_id
        clip_dir.mkdir(parents=True)
        writer = cv2.VideoWriter(str(clip_dir / "video_0001.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (192, 128))
        assert writer.isOpened()
        for frame in range(12):
            writer.write(np.full((128, 192, 3), frame * 15, dtype=np.uint8))
        writer.release()
        # Delete only named synthetic generated frames, within this test's tmp_path.
        for frame in (0, 1, 10, 11):
            (source / "images" / set_id / "video_0001" / f"{frame:05d}.png").unlink()
    result = subprocess.run([sys.executable, "-m", "pie_behavior.extract_frames", "--manifest-dir", str(manifests),
                             "--data-root", str(source)], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for frame in (0, 1, 10, 11):
        array = np.array(Image.open(source / "images/set01/video_0001" / f"{frame:05d}.png"))
        assert abs(float(array.mean()) - frame * 15) < 6


def test_invalid_geometry_time_grid_rejected(demo):
    source, manifests = demo
    dataset = BehaviorDataset(manifests, "train", source, visual=False)
    sample = copy.deepcopy(dataset.samples[0])
    sample["time_offsets_s"][-1] = 0.1
    with pytest.raises(ValueError, match="causal history"):
        geometry(sample)


def test_invalid_inference_image_dimensions_rejected(demo):
    source, manifests = demo
    sample = copy.deepcopy(BehaviorDataset(manifests, "train", source, visual=False).samples[0])
    sample["width"] = 0
    with pytest.raises(ValueError, match="image dimensions"):
        geometry(sample)
