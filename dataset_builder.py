"""Standardize heterogeneous traffic datasets into one demo-ready layout.

Every public dataset arrives shaped differently — Ko-PER ships zips of
per-camera PNG frame folders, AAU RainSnow ships RGB+thermal video pairs,
DAWN ships bare weather-binned images, Urban Tracker ships AVIs. The demos
want one thing: a video file path. This tool converts whatever is present
under ``datasets/`` into:

    datasets/standardized/<dataset>/<sequence>/
        video.mp4 | images/          (payload)
        meta.json                    (source, fps, resolution, frames, notes)

Usage:
    python dataset_builder.py --build                # everything present
    python dataset_builder.py --build --dataset koper
    python dataset_builder.py --rainsnow <path>      # point at the Kaggle download
    python dataset_builder.py --index                # list demo-ready sequences

Then:  python demo.py --case 3 --mode real --video datasets/standardized/<...>/video.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _CV2_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
STANDARD_DIR = DATASETS_DIR / "standardized"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}
#: A directory must hold at least this many images to count as a frame
#: sequence (filters calibration shots and thumbnails).
MIN_FRAMES = 25


def natural_key(name: str) -> List[object]:
    """Sort helper: frame_2 before frame_10."""

    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def discover_frame_dirs(root: Path) -> List[Path]:
    """Directories under ``root`` containing an image sequence."""

    found = []
    for directory in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        images = [p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if len(images) >= MIN_FRAMES:
            found.append(directory)
    return found


def frames_to_video(frame_dir: Path, out_path: Path, fps: float = 25.0) -> int:
    """Assemble a sorted image sequence into an mp4; returns frame count."""

    if cv2 is None:  # pragma: no cover - requires optional dependency
        raise ImportError("opencv-python is required") from _CV2_IMPORT_ERROR

    frames = sorted(
        (p for p in frame_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: natural_key(p.name),
    )
    if not frames:
        return 0
    first = cv2.imread(str(frames[0]))
    if first is None:
        return 0
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    written = 0
    for frame_path in frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        writer.write(image)
        written += 1
    writer.release()
    return written


def probe_video(path: Path) -> Dict[str, object]:
    if cv2 is None:  # pragma: no cover
        return {}
    capture = cv2.VideoCapture(str(path))
    info = {
        "fps": round(capture.get(cv2.CAP_PROP_FPS) or 0.0, 2),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "resolution": [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))],
    }
    capture.release()
    return info


def write_meta(seq_dir: Path, meta: Dict[str, object]) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def register_video(dataset: str, sequence: str, source: Path,
                   notes: str = "", copy: bool = True) -> Path:
    """Place one playable video into the standardized layout."""

    seq_dir = STANDARD_DIR / dataset / sequence
    target = seq_dir / "video.mp4"
    if not target.exists():
        seq_dir.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copyfile(source, target)
        else:
            source.replace(target)
    meta = {"dataset": dataset, "sequence": sequence, "source": str(source),
            "kind": "video", "notes": notes, **probe_video(target)}
    write_meta(seq_dir, meta)
    return target


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------


def build_urbantracker() -> int:
    """The bundled Urban Tracker demo videos (already video files)."""

    built = 0
    for name, notes in [
        ("rouen_crosswalk.avi", "elevated crosswalk view; pedestrians + vehicles"),
        ("sherbrooke_intersection.avi", "fixed camera over a four-way intersection"),
    ]:
        source = ROOT / "videos" / name
        if source.exists():
            register_video("urbantracker", source.stem, source, notes=notes)
            built += 1
    return built


def build_koper() -> int:
    """Ko-PER: unzip sequences, assemble each camera's frame folder to mp4."""

    koper_dir = DATASETS_DIR / "koper"
    if not koper_dir.exists():
        return 0
    built = 0

    example = koper_dir / "ExampleOneCamSeq1a.avi"
    if example.exists():
        register_video("koper", "example_onecam_seq1a", example,
                       notes="single-camera example clip")
        built += 1

    for archive in sorted(koper_dir.glob("Sequence*.zip")):
        extracted = koper_dir / "extracted" / archive.stem
        if not extracted.exists():
            print(f"  extracting {archive.name} ...")
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extracted)
        for frame_dir in discover_frame_dirs(extracted):
            sequence = f"{archive.stem}_{frame_dir.name}".lower()
            seq_dir = STANDARD_DIR / "koper" / sequence
            target = seq_dir / "video.mp4"
            if target.exists():
                built += 1
                continue
            print(f"  assembling {sequence} from {frame_dir} ...")
            frames = frames_to_video(frame_dir, target)
            if frames == 0:
                continue
            write_meta(seq_dir, {
                "dataset": "koper", "sequence": sequence,
                "source": str(frame_dir), "kind": "video",
                "notes": "assembled from PNG frames at nominal 25 fps",
                **probe_video(target),
            })
            built += 1
    return built


def build_rainsnow(rainsnow_root: Optional[Path]) -> int:
    """AAU RainSnow (user-downloaded from Kaggle): copy every video found."""

    roots: List[Path] = []
    if rainsnow_root:
        roots.append(rainsnow_root)
    roots.extend(p for p in DATASETS_DIR.glob("*rainsnow*") if p.is_dir())
    built = 0
    for root in roots:
        for video in sorted(root.rglob("*")):
            if video.suffix.lower() not in VIDEO_EXTS or not video.is_file():
                continue
            sequence = "_".join(video.relative_to(root).with_suffix("").parts).lower()
            register_video("rainsnow", sequence, video,
                           notes="AAU RainSnow (rain/snow/night intersections)")
            built += 1
    return built


def build_dawn() -> int:
    """DAWN: adverse-weather image set (kept as images; detector benchmarking).

    Arrives as one zip per weather condition (Fog/Rain/Sand/Snow).
    """

    archives = sorted((DATASETS_DIR / "dawn").glob("*.zip"))
    if not archives:
        return 0
    extracted = DATASETS_DIR / "dawn" / "extracted"
    for archive in archives:
        marker = extracted / f".{archive.stem}.done"
        if marker.exists():
            continue
        print(f"  extracting {archive.name} ...")
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        marker.touch()
    built = 0
    for frame_dir in discover_frame_dirs(extracted):
        sequence = frame_dir.name.lower()
        seq_dir = STANDARD_DIR / "dawn" / sequence
        images_dir = seq_dir / "images"
        if not images_dir.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for image in frame_dir.iterdir():
                if image.suffix.lower() in IMAGE_EXTS:
                    shutil.copyfile(image, images_dir / image.name)
                    count += 1
        else:
            count = len(list(images_dir.iterdir()))
        write_meta(seq_dir, {
            "dataset": "dawn", "sequence": sequence, "source": str(frame_dir),
            "kind": "images", "frames": count,
            "notes": "adverse-weather stills (fog/rain/snow/sandstorm); "
                     "detector benchmarking, not signal-loop testing",
        })
        built += 1
    return built


BUILDERS = {
    "urbantracker": lambda args: build_urbantracker(),
    "koper": lambda args: build_koper(),
    "rainsnow": lambda args: build_rainsnow(args.rainsnow),
    "dawn": lambda args: build_dawn(),
}


def print_index() -> None:
    if not STANDARD_DIR.exists():
        print("nothing standardized yet — run with --build")
        return
    rows = []
    for meta_path in sorted(STANDARD_DIR.rglob("meta.json")):
        meta = json.loads(meta_path.read_text())
        payload = meta_path.parent / ("video.mp4" if meta.get("kind") == "video" else "images")
        size = (payload.stat().st_size >> 20) if payload.is_file() else "-"
        rows.append((meta["dataset"], meta["sequence"], meta.get("kind", "?"),
                     meta.get("frames", "?"), f"{size}", str(payload.relative_to(ROOT))))
    if not rows:
        print("nothing standardized yet — run with --build")
        return
    widths = [max(len(str(r[i])) for r in rows + [("dataset", "sequence", "kind", "frames", "MB", "path")])
              for i in range(6)]
    header = ("dataset", "sequence", "kind", "frames", "MB", "path")
    for row in [header, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    print("\nrun a sequence:  python demo.py --case 3 --mode real --video <path>")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="standardize everything present")
    parser.add_argument("--dataset", choices=sorted(BUILDERS), help="only this dataset")
    parser.add_argument("--rainsnow", type=Path,
                        help="path to the user-downloaded AAU RainSnow folder")
    parser.add_argument("--index", action="store_true", help="list standardized sequences")
    args = parser.parse_args(argv)

    if not args.build and not args.index:
        parser.error("nothing to do: pass --build and/or --index")

    if args.build:
        names = [args.dataset] if args.dataset else sorted(BUILDERS)
        for name in names:
            print(f"[{name}]")
            built = BUILDERS[name](args)
            print(f"  {built} sequence(s) ready")
    if args.index:
        print_index()
    return 0


if __name__ == "__main__":
    sys.exit(main())
