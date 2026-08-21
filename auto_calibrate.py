"""Trajectory-based auto-calibration: learn detection zones from traffic.

Instead of hand-drawing per-camera detection zones, watch the intersection
for a while and let the traffic explain the scene:

* where vehicles drive IS the road (no segmentation model needed),
* each track's direction of travel identifies its approach — a car moving
  down the frame *came from the North*, and so on,
* where tracks repeatedly stop and dwell marks the stop line,
* regions with detections but no through-flow are parking, discovered
  rather than declared.

The output is the same fractional-zone format the real-mode demos consume
(`DEFAULT_ZONES` in four_way_intersection.py), plus a visual overlay for
review. This is a prototype of the roadmap's Phase 2 "semi-automatic zone
calibration" — self-supervised, using only the detector and tracker the
system already runs.

Usage:
    python auto_calibrate.py videos/sherbrooke_intersection.avi \
        --loops 1 --out docs/auto_calibration.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _CV2_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

APPROACHES = ("N", "S", "E", "W")

#: A track must move at least this fraction of the frame diagonal to count
#: as through-traffic (filters parked cars and detection jitter).
MIN_DISPLACEMENT_FRAC = 0.08
#: Minimum number of samples in a usable track.
MIN_TRACK_POINTS = 8
#: A point is "dwelling" when the local speed drops below this fraction of
#: the frame diagonal per second (queued at a stop line).
DWELL_SPEED_FRAC = 0.005


@dataclass(slots=True)
class Track:
    """One vehicle's observed path: (time, x, y) samples in pixels."""

    points: List[Tuple[float, float, float]] = field(default_factory=list)

    def displacement(self) -> float:
        if len(self.points) < 2:
            return 0.0
        _, x0, y0 = self.points[0]
        _, x1, y1 = self.points[-1]
        return math.dist((x0, y0), (x1, y1))

    def direction(self, first_frac: float = 1.0) -> Tuple[float, float]:
        """Unit travel vector (dx, dy) over the first ``first_frac`` of the path.

        ``first_frac=1.0`` gives the net direction; a smaller fraction gives
        the ENTRY direction, which is what identifies the approach — a
        vehicle that arrives from the top and then turns right still *came
        from the North*, even though its net displacement points sideways.
        """

        cut = max(2, int(len(self.points) * first_frac))
        _, x0, y0 = self.points[0]
        _, x1, y1 = self.points[cut - 1]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        return dx / norm, dy / norm


#: Fraction of a track used to measure its ENTRY direction. Long enough to
#: average out detection jitter, short enough to precede any turn.
ENTRY_FRAC = 0.35


def classify_approach(track: Track, frame_diag: float) -> Optional[str]:
    """Name the approach a track CAME FROM, or None for non-through traffic.

    Image convention (matches the demo zones): a vehicle moving down the
    frame entered from the top — the North approach — and so on. The entry
    segment decides, so turning vehicles keep their true approach; when the
    entry segment is too short/noisy to be meaningful, the net direction is
    the fallback.
    """

    if len(track.points) < MIN_TRACK_POINTS:
        return None
    if track.displacement() < MIN_DISPLACEMENT_FRAC * frame_diag:
        return None

    _, ex0, ey0 = track.points[0]
    cut = max(2, int(len(track.points) * ENTRY_FRAC))
    _, ex1, ey1 = track.points[cut - 1]
    entry_disp = math.dist((ex0, ey0), (ex1, ey1))
    frac = ENTRY_FRAC if entry_disp >= 0.03 * frame_diag else 1.0

    dx, dy = track.direction(first_frac=frac)
    if abs(dy) >= abs(dx):
        return "N" if dy > 0 else "S"
    return "W" if dx > 0 else "E"


def derive_zone(
    tracks: List[Track],
    frame_size: Tuple[int, int],
    upstream_frac: float = 0.45,
    lo_pct: float = 8.0,
    hi_pct: float = 92.0,
) -> Optional[Tuple[float, float, float, float]]:
    """Fractional (x, y, w, h) covering the upstream portion of the tracks.

    The first ``upstream_frac`` of each track's points is where vehicles
    approach and queue — exactly what a detection zone should cover.
    Percentile bounds reject stragglers and tracker glitches.
    """

    height, width = frame_size
    xs: List[float] = []
    ys: List[float] = []
    for track in tracks:
        cut = max(2, int(len(track.points) * upstream_frac))
        for _, x, y in track.points[:cut]:
            xs.append(x)
            ys.append(y)
    if len(xs) < MIN_TRACK_POINTS:
        return None

    xs.sort()
    ys.sort()

    def pct(values: List[float], q: float) -> float:
        index = min(len(values) - 1, max(0, int(round(q / 100.0 * (len(values) - 1)))))
        return values[index]

    x0, x1 = pct(xs, lo_pct), pct(xs, hi_pct)
    y0, y1 = pct(ys, lo_pct), pct(ys, hi_pct)
    if x1 <= x0 or y1 <= y0:
        return None
    return (
        max(0.0, x0 / width),
        max(0.0, y0 / height),
        min(1.0, (x1 - x0) / width),
        min(1.0, (y1 - y0) / height),
    )


def dwell_points(tracks: List[Track], frame_diag: float) -> List[Tuple[float, float]]:
    """Locations where through-traffic stood still (queue heads: stop lines)."""

    threshold = DWELL_SPEED_FRAC * frame_diag  # px per second
    points: List[Tuple[float, float]] = []
    for track in tracks:
        for (t0, x0, y0), (t1, x1, y1) in zip(track.points, track.points[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            if math.dist((x0, y0), (x1, y1)) / dt < threshold:
                points.append((x1, y1))
    return points


def collect_trajectories(
    video_path: str | Path,
    loops: int = 1,
    frame_stride: int = 1,
) -> Tuple[List[Track], Tuple[int, int], "object"]:
    """Run the standard detector+tracker over the video, gathering tracks."""

    if cv2 is None:  # pragma: no cover - requires optional dependency
        raise ImportError("opencv-python is required for auto-calibration.") from _CV2_IMPORT_ERROR

    from smart_traffic_system import DetectorConfig, VehicleDetector

    detector = VehicleDetector(DetectorConfig())
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    finished: List[Track] = []
    active: Dict[int, Track] = defaultdict(Track)
    frame_index = 0
    loop = 0
    sample_frame = None

    while loop < loops:
        ok, frame = capture.read()
        if not ok:
            loop += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            detector.reset_tracker()
            finished.extend(active.values())
            active = defaultdict(Track)
            continue
        if sample_frame is None:
            sample_frame = frame.copy()
        if frame_index % max(1, frame_stride) == 0:
            now = frame_index / fps
            for det in detector.track_vehicles(frame):
                if det.track_id is None:
                    continue
                cx, cy = det.center
                active[det.track_id].points.append((now, cx, cy))
        frame_index += 1
        if frame_index % 500 == 0:
            logger.info("calibration: %d frames, %d active / %d finished tracks",
                        frame_index, len(active), len(finished))

    finished.extend(active.values())
    capture.release()
    height, width = sample_frame.shape[:2] if sample_frame is not None else (0, 0)
    return finished, (height, width), sample_frame


def save_tracks(path: str | Path, tracks: List[Track], frame_size: Tuple[int, int]) -> None:
    payload = {
        "frame_size": list(frame_size),
        "tracks": [[list(pt) for pt in t.points] for t in tracks],
    }
    Path(path).write_text(json.dumps(payload))


def load_tracks(path: str | Path) -> Tuple[List[Track], Tuple[int, int]]:
    payload = json.loads(Path(path).read_text())
    tracks = [Track(points=[tuple(pt) for pt in pts]) for pts in payload["tracks"]]
    height, width = payload["frame_size"]
    return tracks, (height, width)


def auto_calibrate(
    video_path: str | Path,
    loops: int = 1,
    tracks_cache: Optional[str] = None,
) -> Dict[str, object]:
    """Learn approach zones and dwell locations from observed traffic.

    ``tracks_cache``: optional JSON path — reused when it exists (instant
    analysis iteration), written after collection when it does not.
    """

    sample_frame = None
    if tracks_cache and Path(tracks_cache).exists():
        logger.info("loading cached tracks from %s", tracks_cache)
        tracks, frame_size = load_tracks(tracks_cache)
        capture = cv2.VideoCapture(str(video_path))
        ok, sample_frame = capture.read()
        capture.release()
    else:
        tracks, frame_size, sample_frame = collect_trajectories(video_path, loops=loops)
        if tracks_cache:
            save_tracks(tracks_cache, tracks, frame_size)
            logger.info("tracks cached to %s", tracks_cache)
    height, width = frame_size
    diag = math.hypot(width, height)

    grouped: Dict[str, List[Track]] = {name: [] for name in APPROACHES}
    stationary = 0
    for track in tracks:
        name = classify_approach(track, diag)
        if name is None:
            stationary += 1
            continue
        grouped[name].append(track)

    zones: Dict[str, Tuple[float, float, float, float]] = {}
    for name, group in grouped.items():
        zone = derive_zone(group, frame_size)
        if zone is not None:
            zones[name] = zone

    through = [t for group in grouped.values() for t in group]
    return {
        "video": str(video_path),
        "frame_size": frame_size,
        "tracks_total": len(tracks),
        "tracks_through": len(through),
        "tracks_stationary_or_short": stationary,
        "flows": {name: len(group) for name, group in grouped.items()},
        "zones": zones,
        "dwell_points": dwell_points(through, diag),
        "sample_frame": sample_frame,
    }


def render_overlay(
    result: Dict[str, object],
    hand_zones: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> "object":
    """Draw learned zones (solid), dwell heat (dots), hand zones (dashed)."""

    frame = result["sample_frame"].copy()
    height, width = result["frame_size"]  # type: ignore[misc]

    colors = {"N": (70, 180, 255), "S": (255, 180, 70), "E": (120, 255, 120), "W": (200, 120, 255)}

    for x, y in result["dwell_points"]:  # type: ignore[union-attr]
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 230), -1)

    def rect_px(zone):
        fx, fy, fw, fh = zone
        return int(fx * width), int(fy * height), int(fw * width), int(fh * height)

    for name, zone in result["zones"].items():  # type: ignore[union-attr]
        zx, zy, zw, zh = rect_px(zone)
        cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), colors[name], 3)
        flow = result["flows"][name]  # type: ignore[index]
        cv2.putText(frame, f"{name} (learned, {flow} tracks)", (zx + 4, zy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)

    if hand_zones:
        for name, zone in hand_zones.items():
            zx, zy, zw, zh = rect_px(zone)
            for x in range(zx, zx + zw, 14):
                cv2.line(frame, (x, zy), (min(x + 7, zx + zw), zy), (255, 255, 255), 1)
                cv2.line(frame, (x, zy + zh), (min(x + 7, zx + zw), zy + zh), (255, 255, 255), 1)
            for y in range(zy, zy + zh, 14):
                cv2.line(frame, (zx, y), (zx, min(y + 7, zy + zh)), (255, 255, 255), 1)
                cv2.line(frame, (zx + zw, y), (zx + zw, min(y + 7, zy + zh)), (255, 255, 255), 1)

    cv2.rectangle(frame, (10, 10), (470, 58), (30, 30, 30), -1)
    cv2.putText(frame, "AUTO-CALIBRATION: learned zones (solid) vs hand zones (dashed)",
                (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, "red dots = learned dwell (stop-line) locations",
                (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 230), 1)
    return frame


def zone_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Learn detection zones from observed traffic")
    parser.add_argument("video", help="Fixed-camera intersection video")
    parser.add_argument("--loops", type=int, default=1, help="Playback passes to accumulate")
    parser.add_argument("--out", default="auto_calibration.jpg", help="Overlay image output path")
    parser.add_argument("--json-out", help="Write learned zones as JSON")
    parser.add_argument("--tracks-cache",
                        help="JSON path: reuse collected tracks if present, else write them")
    parser.add_argument("--compare-defaults", action="store_true",
                        help="Overlay and score against four_way_intersection.DEFAULT_ZONES")
    args = parser.parse_args(argv)

    result = auto_calibrate(args.video, loops=args.loops, tracks_cache=args.tracks_cache)

    logger.info("tracks: %d total, %d through-traffic, %d stationary/short",
                result["tracks_total"], result["tracks_through"],
                result["tracks_stationary_or_short"])
    logger.info("flows per approach: %s", result["flows"])
    for name, zone in result["zones"].items():
        logger.info("learned zone %s: (%.3f, %.3f, %.3f, %.3f)", name, *zone)

    hand = None
    if args.compare_defaults:
        from four_way_intersection import DEFAULT_ZONES
        hand = DEFAULT_ZONES
        for name, zone in result["zones"].items():
            if name in hand:
                logger.info("IoU vs hand-drawn %s zone: %.2f", name, zone_iou(zone, hand[name]))

    overlay = render_overlay(result, hand_zones=hand)
    cv2.imwrite(args.out, overlay)
    logger.info("overlay written to %s", args.out)

    if args.json_out:
        payload = {k: v for k, v in result.items() if k != "sample_frame" and k != "dwell_points"}
        payload["dwell_point_count"] = len(result["dwell_points"])  # type: ignore[arg-type]
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        logger.info("zones written to %s", args.json_out)


if __name__ == "__main__":
    main()
