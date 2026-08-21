"""Tests for the demo launcher dispatch and the video download helpers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import demo
import video_downloader
from video_downloader import TrafficVideoSetup, ensure_video, is_plausible_video


# ---------------------------------------------------------------------------
# is_plausible_video / ensure_video
# ---------------------------------------------------------------------------


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_is_plausible_video_rejects_stubs_and_pointers(tmp_path):
    assert not is_plausible_video(tmp_path / "missing.mp4")
    assert not is_plausible_video(_write_bytes(tmp_path / "empty.mp4", b""))
    assert not is_plausible_video(
        _write_bytes(
            tmp_path / "pointer.mp4",
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n",
        )
    )
    # An HTML error page served as a 200 is not a video, no matter its size.
    assert not is_plausible_video(
        _write_bytes(tmp_path / "error.mp4", b"<html>" + b"x" * 200 * 1024)
    )
    assert is_plausible_video(
        _write_bytes(tmp_path / "real.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200 * 1024)
    )


def test_ensure_video_returns_existing_valid_file(tmp_path):
    target = _write_bytes(tmp_path / "clip.mp4", b"RIFF" + b"\x00" * 200 * 1024)
    assert ensure_video("clip.mp4", output_dir=str(tmp_path)) == str(target)


def test_ensure_video_redownloads_invalid_stub(tmp_path, monkeypatch):
    stub = _write_bytes(tmp_path / "clip.mp4", b"version https://git-lfs stub")

    def fake_download(self, url, filename):
        _write_bytes(self.output_dir / filename, b"\x00" * 200 * 1024)
        return True

    monkeypatch.setattr(TrafficVideoSetup, "download_video", fake_download)
    monkeypatch.setitem(TrafficVideoSetup.VIDEO_URLS, "clip.mp4", ["https://example.invalid/x"])

    result = ensure_video("clip.mp4", output_dir=str(tmp_path))
    assert result == str(stub)
    assert is_plausible_video(stub)


def test_ensure_video_returns_none_without_known_source(tmp_path):
    assert ensure_video("unknown-video.mp4", output_dir=str(tmp_path)) is None


def test_ensure_video_sanitizes_traversal_names(tmp_path, monkeypatch):
    # A hostile name must not escape the output directory.
    calls = {}

    def fake_download(self, url, filename):
        calls["filename"] = filename
        return False

    monkeypatch.setattr(TrafficVideoSetup, "download_video", fake_download)
    monkeypatch.setitem(
        TrafficVideoSetup.VIDEO_URLS, "evil.mp4", ["https://example.invalid/x"]
    )
    ensure_video("..\\..\\evil.mp4", output_dir=str(tmp_path))
    assert calls["filename"] == "evil.mp4"


def test_download_video_rejects_checksum_mismatch(tmp_path, monkeypatch):
    import io

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    payload = b"\x00" * 4096
    monkeypatch.setattr(
        video_downloader.urllib.request,
        "urlopen",
        lambda request, timeout=0: FakeResponse(payload),
    )
    monkeypatch.setitem(
        TrafficVideoSetup.VIDEO_SHA256, "pinned.mp4", "0" * 64  # wrong on purpose
    )
    monkeypatch.setitem(
        TrafficVideoSetup.VIDEO_URLS, "pinned.mp4", ["https://example.invalid/x"]
    )

    setup = TrafficVideoSetup(str(tmp_path))
    assert setup.download_video("https://example.invalid/x", "pinned.mp4") is False
    assert not (tmp_path / "pinned.mp4").exists()
    assert not (tmp_path / "pinned.mp4.part").exists()


# ---------------------------------------------------------------------------
# demo.py dispatch
# ---------------------------------------------------------------------------


def test_tracker_nms_conf_feeds_bytetracks_low_score_tier():
    """The NMS conf passed to model.track must not exceed the tracker's
    track_low_thresh, or ByteTrack's occlusion-recovery tier receives
    nothing and track IDs fragment (audit finding: one parked car split
    into 5+ IDs)."""

    from smart_traffic_system import VehicleDetector

    config_text = Path(VehicleDetector.TRACKER_CONFIG).read_text()
    values = {}
    for line in config_text.splitlines():
        if ":" in line and not line.strip().startswith("#"):
            key, _, value = line.partition(":")
            try:
                values[key.strip()] = float(value.strip())
            except ValueError:
                pass

    assert VehicleDetector.TRACKER_NMS_CONF <= values["track_low_thresh"]
    assert values["track_low_thresh"] < values["track_high_thresh"]


def test_parser_covers_all_cases_and_modes():
    parser = demo.build_parser()
    args = parser.parse_args(["--case", "3", "--mode", "real", "--max-frames", "10"])
    assert args.case == 3
    assert args.mode == "real"
    assert args.max_frames == 10
    assert set(demo.CASES) == {1, 2, 3}


def test_main_dispatches_to_selected_case(monkeypatch):
    called = {}
    monkeypatch.setattr(demo, "run_case1", lambda args: called.setdefault("case", 1))
    monkeypatch.setattr(demo, "run_case2", lambda args: called.setdefault("case", 2))
    monkeypatch.setattr(demo, "run_case3", lambda args: called.setdefault("case", 3))

    demo.main(["--case", "2", "--mode", "simulation"])
    assert called["case"] == 2


def test_main_rejects_out_of_range_values(monkeypatch):
    monkeypatch.setattr(demo, "run_case3", lambda args: None)
    with pytest.raises(SystemExit):
        demo.main(["--case", "3", "--fps", "0"])
    with pytest.raises(SystemExit):
        demo.main(["--case", "3", "--size", "10"])
    with pytest.raises(SystemExit):
        demo.main(["--case", "3", "--max-frames", "-5"])
    # Valid values pass through.
    demo.main(["--case", "3", "--fps", "30", "--size", "640", "--max-frames", "0"])


def test_case2_real_rejects_partial_video_override():
    parser = demo.build_parser()
    args = parser.parse_args(["--case", "2", "--mode", "real", "--video-road1", "x.mp4"])
    with pytest.raises(SystemExit):
        demo.run_case2(args)


def test_ensure_case_video_errors_on_missing_override(tmp_path):
    parser = demo.build_parser()
    args = parser.parse_args(
        ["--case", "1", "--mode", "real", "--video", str(tmp_path / "nope.mp4")]
    )
    with pytest.raises(FileNotFoundError):
        demo._ensure_case_video(1, args.video)
