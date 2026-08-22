"""Tests for the dataset standardizer's pure building blocks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

try:  # pragma: no cover - optional test dependency
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None

from dataset_builder import (
    MIN_FRAMES,
    discover_frame_dirs,
    frames_to_video,
    natural_key,
    probe_video,
)


def test_natural_key_orders_frames_numerically():
    names = ["frame_10.png", "frame_2.png", "frame_1.png"]
    assert sorted(names, key=natural_key) == ["frame_1.png", "frame_2.png", "frame_10.png"]


@pytest.mark.skipif(cv2 is None or np is None, reason="requires cv2 and numpy")
def test_discover_and_assemble_frame_sequence(tmp_path):
    frame_dir = tmp_path / "cam1"
    frame_dir.mkdir()
    for i in range(MIN_FRAMES + 5):
        image = np.full((48, 64, 3), i * 3 % 255, dtype=np.uint8)
        cv2.imwrite(str(frame_dir / f"frame_{i}.png"), image)
    # A folder with too few images must not be picked up.
    sparse = tmp_path / "calib"
    sparse.mkdir()
    cv2.imwrite(str(sparse / "board.png"), np.zeros((10, 10, 3), dtype=np.uint8))

    found = discover_frame_dirs(tmp_path)
    assert found == [frame_dir]

    out = tmp_path / "out" / "video.mp4"
    written = frames_to_video(frame_dir, out, fps=25.0)
    assert written == MIN_FRAMES + 5
    info = probe_video(out)
    assert info["frames"] == MIN_FRAMES + 5
    assert info["resolution"] == [64, 48]
    assert info["fps"] == pytest.approx(25.0, abs=0.1)


@pytest.mark.skipif(cv2 is None or np is None, reason="requires cv2 and numpy")
def test_frames_to_video_handles_mixed_sizes(tmp_path):
    frame_dir = tmp_path / "cam"
    frame_dir.mkdir()
    for i in range(MIN_FRAMES):
        h = 48 if i % 2 == 0 else 32  # inconsistent sizes get normalized
        cv2.imwrite(str(frame_dir / f"f_{i}.png"),
                    np.zeros((h, 64, 3), dtype=np.uint8))
    out = tmp_path / "v.mp4"
    assert frames_to_video(frame_dir, out) == MIN_FRAMES
    assert probe_video(out)["resolution"] == [64, 48]


def test_frames_to_video_empty_dir(tmp_path):
    if cv2 is None:
        pytest.skip("requires cv2")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert frames_to_video(empty, tmp_path / "x.mp4") == 0
