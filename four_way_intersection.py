"""Case 3 — Adaptive four-way intersection (crossroads).

Two crossing roads with four approaches (North, South, East, West). The two
approaches of one road share a phase (NS axis vs. EW axis), like a standard
two-phase signal plan. The adaptive controller:

* keeps an axis green while it has demand (up to ``MAX_GREEN``),
* switches early when the green axis is empty and the cross axis is waiting
  ("no car coming from one way -> the other can proceed"),
* never leaves an axis red for longer than ``MAX_RED`` (fairness cap),
* always sequences green -> yellow -> all-red -> cross green, so the
  signalization stays safe no matter how the demand fluctuates.

Simulation mode renders a top-down intersection with queueing vehicles.
Real mode runs in *shadow mode* on prerecorded overhead footage: YOLOv8
counts vehicles inside four configurable detection zones and the overlay
shows the phase decisions the controller would issue for that traffic.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _NUMPY_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

APPROACHES = ("N", "S", "E", "W")
AXIS_OF = {"N": "NS", "S": "NS", "E": "EW", "W": "EW"}


class FourWayController:
    """Two-phase adaptive controller for a four-way intersection."""

    MIN_GREEN = 5.0
    MAX_GREEN = 30.0
    YELLOW_TIME = 3.0
    ALL_RED_TIME = 1.5
    #: Detector-failure recall: an axis is served after this much red even if
    #: no demand was *measured* there, so a dead camera or mis-calibrated
    #: zone can never starve an approach. Axes with detected demand are
    #: served much sooner (worst case MAX_GREEN + YELLOW + ALL_RED ≈ 35 s).
    MAX_RED = 300.0

    def __init__(self, time_func: Callable[[], float] | None = None) -> None:
        self._time_func = time_func or time.monotonic
        now = self._time_func()
        self.active_axis = "NS"
        self.phase = "GREEN"  # GREEN | YELLOW | ALL_RED
        self.phase_start = now
        self._red_start = {"NS": now, "EW": now}
        self.early_switches = 0
        self.total_switches = 0
        self._pending_early = False

    def _elapsed(self) -> float:
        return self._time_func() - self.phase_start

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self.phase_start = self._time_func()

    @staticmethod
    def _other(axis: str) -> str:
        return "EW" if axis == "NS" else "NS"

    def update(self, counts: Dict[str, int]) -> Dict[str, object]:
        """Advance the controller with per-approach vehicle counts."""

        now = self._time_func()
        active = self.active_axis
        cross = self._other(active)
        active_demand = sum(counts.get(a, 0) for a in APPROACHES if AXIS_OF[a] == active)
        cross_demand = sum(counts.get(a, 0) for a in APPROACHES if AXIS_OF[a] == cross)

        if self.phase == "GREEN":
            elapsed = self._elapsed()
            cross_red_for = now - self._red_start[cross]
            switch = False
            if elapsed >= self.MIN_GREEN:
                if cross_demand > 0:
                    if active_demand == 0:
                        switch = True
                        self._pending_early = True
                    elif elapsed >= self.MAX_GREEN:
                        switch = True
                # Detector-failure failsafe: serve the cross axis after
                # MAX_RED regardless of measured demand, so a dead camera,
                # occlusion, or a mis-calibrated zone can never starve an
                # approach (fixed-time recall, as real controllers do).
                if not switch and cross_red_for >= self.MAX_RED:
                    switch = True
            if switch:
                # The active axis effectively goes red at yellow onset.
                self._red_start[active] = now
                self._enter("YELLOW")
        elif self.phase == "YELLOW":
            if self._elapsed() >= self.YELLOW_TIME:
                self._enter("ALL_RED")
        elif self.phase == "ALL_RED":
            if self._elapsed() >= self.ALL_RED_TIME:
                self.active_axis = cross
                self.total_switches += 1
                # Both counters advance at changeover completion, so the
                # displayed "demand-driven X of Y" ratio is always coherent.
                if self._pending_early:
                    self.early_switches += 1
                    self._pending_early = False
                self._enter("GREEN")

        return self.status(counts)

    def status(self, counts: Optional[Dict[str, int]] = None) -> Dict[str, object]:
        signals: Dict[str, str] = {}
        for approach in APPROACHES:
            if AXIS_OF[approach] == self.active_axis:
                signals[approach] = {"GREEN": "GREEN", "YELLOW": "YELLOW"}.get(
                    self.phase, "RED"
                )
            else:
                signals[approach] = "RED"
        durations = {"YELLOW": self.YELLOW_TIME, "ALL_RED": self.ALL_RED_TIME}
        duration = durations.get(self.phase)
        remaining = None if duration is None else max(0.0, duration - self._elapsed())
        return {
            "signals": signals,
            "active_axis": self.active_axis,
            "phase": self.phase,
            "time_remaining": remaining,
            "counts": dict(counts or {}),
            "early_switches": self.early_switches,
            "total_switches": self.total_switches,
        }


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

#: All world/geometry constants are expressed in this logical coordinate
#: space; rendering scales them to the actual window size. World behavior
#: (spawns, queues, controller decisions) is therefore identical on every
#: monitor resolution.
LOGICAL_SIZE = 640.0


def detect_display_size(default: int = 720) -> int:
    """Pick a square render size that fits the current monitor.

    Falls back to ``default`` when the screen size cannot be determined
    (headless runs, unusual platforms).
    """

    width = height = 0
    try:  # Windows
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        try:  # cross-platform fallback
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            width, height = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
        except Exception:
            width = height = 0

    if width > 0 and height > 0:
        return max(480, min(1400, int(min(width, height) * 0.85)))
    return default


@dataclass(slots=True)
class ApproachVehicle:
    """A vehicle on one approach; ``distance`` is measured to the stop line."""

    distance: float  # > 0: approaching, <= 0: inside / past the intersection
    speed: float
    length: float = 34.0
    wait_time: float = 0.0


@dataclass(slots=True)
class Approach:
    """A single approach lane feeding the intersection."""

    name: str
    spawn_rate: float  # vehicles per second (Poisson)
    entry_distance: float = 300.0
    detection_length: float = 110.0
    clear_distance: float = 160.0
    min_gap: float = 12.0
    vehicles: List[ApproachVehicle] = field(default_factory=list)

    def maybe_spawn(self, rng: random.Random, dt: float) -> None:
        if rng.random() >= self.spawn_rate * dt:
            return
        vehicle = ApproachVehicle(
            distance=self.entry_distance + rng.uniform(0, 40),
            speed=rng.uniform(60.0, 95.0),
        )
        if self.vehicles:
            last = max(v.distance + v.length for v in self.vehicles)
            if vehicle.distance < last + self.min_gap:
                return  # entry occupied; skip this spawn
        self.vehicles.append(vehicle)

    def step(self, signal: str, dt: float) -> None:
        # Nearest to the stop line first, so followers queue behind leaders.
        self.vehicles.sort(key=lambda v: v.distance)
        leader_tail = -float("inf")
        for vehicle in self.vehicles:
            target = vehicle.distance - vehicle.speed * dt
            if vehicle.distance >= 0 and signal != "GREEN":
                # Hold at the line (including exactly on it) unless the
                # vehicle has already entered the intersection.
                target = max(target, 0.0)
            if leader_tail != -float("inf"):
                target = max(target, leader_tail + self.min_gap)
            moved = target < vehicle.distance - 1e-9
            if not moved and vehicle.distance > 0:
                vehicle.wait_time += dt
            vehicle.distance = target
            leader_tail = vehicle.distance + vehicle.length
        self.vehicles = [v for v in self.vehicles if v.distance > -self.clear_distance]

    def demand_count(self) -> int:
        return sum(1 for v in self.vehicles if 0.0 <= v.distance <= self.detection_length)

    def average_wait(self) -> float:
        waiting = [v.wait_time for v in self.vehicles if v.distance >= 0]
        return sum(waiting) / len(waiting) if waiting else 0.0


class FourWaySimulation:
    """Top-down synthetic crossroads with two-phase adaptive control."""

    APPROACH_COLORS = {
        "N": (70, 180, 255),
        "S": (255, 180, 70),
        "E": (120, 255, 120),
        "W": (200, 120, 255),
    }

    def __init__(
        self,
        fps: int = 30,
        size: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        spawn_rates: Optional[Dict[str, float]] = None,
    ) -> None:
        if cv2 is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "opencv-python is required for the four-way simulation."
            ) from _CV2_IMPORT_ERROR
        if np is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "numpy is required for the four-way simulation."
            ) from _NUMPY_IMPORT_ERROR

        self.fps = max(1, fps)
        # Auto-fit the window to the monitor unless a size is forced. The
        # scale factor only affects rendering; the world itself always runs
        # in LOGICAL_SIZE coordinates, so behavior is resolution-independent.
        self.size = size if size is not None else detect_display_size()
        self.scale = self.size / LOGICAL_SIZE
        self.rng = random.Random(seed)

        # Asymmetric defaults make the demo point obvious: the main NS road
        # is busy while EW only sees the occasional car, so EW red time is
        # skipped whenever nothing is waiting there.
        rates = spawn_rates or {"N": 0.16, "S": 0.14, "E": 0.05, "W": 0.04}
        self.approaches: Dict[str, Approach] = {
            name: Approach(name=name, spawn_rate=rates.get(name, 0.1))
            for name in APPROACHES
        }

        self._sim_time = 0.0
        self.controller = FourWayController(time_func=lambda: self._sim_time)

        # Geometry, in LOGICAL coordinates (scaled only when drawing).
        self.center = LOGICAL_SIZE / 2
        self.road_half_width = 54.0
        self.lane_offset = 27.0
        self.stop_offset = self.road_half_width + 10.0
        self._background = self._create_background()

    # -- scaling helpers -----------------------------------------------------
    def px(self, value: float) -> int:
        """Convert a logical coordinate/length to window pixels."""

        return int(round(value * self.scale))

    def _font_scale(self, mult: float = 1.0) -> float:
        return max(0.45, 0.55 * self.scale) * mult

    def _thickness(self, mult: float = 1.0) -> int:
        return max(1, int(round(2 * self.scale * mult)))

    def _text(
        self,
        frame: "np.ndarray",
        text: str,
        logical_org: Tuple[float, float],
        color: Tuple[int, int, int] = (255, 255, 255),
        mult: float = 1.0,
    ) -> None:
        """Draw outlined text at a logical position (readable on any ground)."""

        org = (self.px(logical_org[0]), self.px(logical_org[1]))
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
            self._font_scale(mult), (0, 0, 0), self._thickness(mult) + 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
            self._font_scale(mult), color, self._thickness(mult), cv2.LINE_AA,
        )

    # -- world -------------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        return {name: app.demand_count() for name, app in self.approaches.items()}

    def step(self, dt: float) -> Dict[str, object]:
        self._sim_time += dt
        for approach in self.approaches.values():
            approach.maybe_spawn(self.rng, dt)
        status = self.controller.update(self.counts())
        signals: Dict[str, str] = status["signals"]  # type: ignore[assignment]
        for name, approach in self.approaches.items():
            approach.step(signals[name], dt)
        return status

    # -- rendering -----------------------------------------------------------
    def _create_background(self) -> "np.ndarray":
        s = self.size
        c, rw, so = self.center, self.road_half_width, self.stop_offset
        px = self.px
        frame = np.full((s, s, 3), 24, dtype=np.uint8)
        road = (66, 66, 66)
        cv2.rectangle(frame, (px(c - rw), 0), (px(c + rw), s), road, -1)
        cv2.rectangle(frame, (0, px(c - rw)), (s, px(c + rw)), road, -1)
        # Centre lines (dashed), in logical steps so density is identical at
        # every resolution.
        dash, gap = 18.0, 16.0
        pos = 0.0
        while pos < LOGICAL_SIZE:
            if abs(pos - c) > rw + 10:
                cv2.line(
                    frame,
                    (px(c), px(pos)),
                    (px(c), min(px(pos + dash), s)),
                    (170, 170, 170),
                    self._thickness(),
                )
                cv2.line(
                    frame,
                    (px(pos), px(c)),
                    (min(px(pos + dash), s), px(c)),
                    (170, 170, 170),
                    self._thickness(),
                )
            pos += dash + gap
        # Stop lines (one per approach, on the incoming half of each road).
        stop_th = self._thickness(1.5)
        cv2.line(frame, (px(c - rw), px(c - so)), (px(c), px(c - so)), (230, 230, 230), stop_th)  # N
        cv2.line(frame, (px(c), px(c + so)), (px(c + rw), px(c + so)), (230, 230, 230), stop_th)  # S
        cv2.line(frame, (px(c + so), px(c - rw)), (px(c + so), px(c)), (230, 230, 230), stop_th)  # E
        cv2.line(frame, (px(c - so), px(c)), (px(c - so), px(c + rw)), (230, 230, 230), stop_th)  # W
        return frame

    def _vehicle_rect(self, name: str, vehicle: ApproachVehicle) -> Tuple[int, int, int, int]:
        c, lo, so = self.center, self.lane_offset, self.stop_offset
        px = self.px
        width = 20.0
        length = float(vehicle.length)
        d = vehicle.distance
        if name == "N":  # southbound, drives on the west half, moving down
            front = c - so - d
            return (px(c - lo - width / 2), px(front - length), px(width), px(length))
        if name == "S":  # northbound, east half, moving up
            front = c + so + d
            return (px(c + lo - width / 2), px(front), px(width), px(length))
        if name == "E":  # westbound, north half, moving left
            front = c + so + d
            return (px(front), px(c - lo - width / 2), px(length), px(width))
        # W: eastbound, south half, moving right
        front = c - so - d
        return (px(front - length), px(c + lo - width / 2), px(length), px(width))

    #: Logical top-left corners of the four signal housings. Each sits in
    #: the corner quadrant beside its approach's stop line, so the four
    #: housings can never overlap one another or the roads.
    _HOUSING_W, _HOUSING_H = 24.0, 62.0
    _SIGNAL_POSITIONS = {
        "N": (320.0 - 54.0 - 16.0 - 24.0, 320.0 - 64.0 - 62.0),  # NW corner
        "E": (320.0 + 64.0 + 4.0, 320.0 - 54.0 - 16.0 - 62.0),   # NE corner
        "S": (320.0 + 54.0 + 16.0, 320.0 + 64.0),                # SE corner
        "W": (320.0 - 64.0 - 4.0 - 24.0, 320.0 + 54.0 + 16.0),   # SW corner
    }

    def _draw_signal_housing(self, frame: "np.ndarray", name: str, signal: str) -> None:
        px = self.px
        lx, ly = self._SIGNAL_POSITIONS[name]
        w, h = self._HOUSING_W, self._HOUSING_H
        cv2.rectangle(
            frame, (px(lx), px(ly)), (px(lx + w), px(ly + h)), (38, 38, 42), -1
        )
        cv2.rectangle(
            frame, (px(lx), px(ly)), (px(lx + w), px(ly + h)), (200, 200, 200), self._thickness(0.5)
        )
        lamp_radius = max(3, px(7.0))
        lamps = [
            ("RED", (0, 0, 220), ly + h * 0.20),
            ("YELLOW", (0, 210, 230), ly + h * 0.50),
            ("GREEN", (0, 200, 0), ly + h * 0.80),
        ]
        for lamp_name, color, lamp_y in lamps:
            lit = signal == lamp_name
            cv2.circle(
                frame,
                (px(lx + w / 2), px(lamp_y)),
                lamp_radius,
                color if lit else (70, 70, 70),
                -1,
            )
        # Approach letter beneath the housing, outlined for contrast.
        self._text(frame, name, (lx + w / 2 - 6.0, ly + h + 16.0), mult=0.95)

    def _draw_hud(self, frame: "np.ndarray", status: Dict[str, object]) -> None:
        counts = self.counts()
        info = [
            f"Axis: {status['active_axis']} {status['phase']}",
            "Waiting  N:%d S:%d E:%d W:%d" % (counts["N"], counts["S"], counts["E"], counts["W"]),
            "Switches: %d (demand-driven: %d)"
            % (status["total_switches"], status["early_switches"]),
            "Avg wait  NS: %.1fs  EW: %.1fs"
            % (
                (self.approaches["N"].average_wait() + self.approaches["S"].average_wait()) / 2,
                (self.approaches["E"].average_wait() + self.approaches["W"].average_wait()) / 2,
            ),
        ]
        remaining = status["time_remaining"]
        if remaining is not None:
            info.insert(1, f"Phase ends in: {float(remaining):.1f}s")

        # Translucent panel keeps the text readable over any scene content.
        line_h = 24.0
        panel_w, panel_h = 264.0, 12.0 + line_h * len(info)
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (self.px(8.0), self.px(8.0)),
            (self.px(8.0 + panel_w), self.px(8.0 + panel_h)), (20, 20, 20), -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        for idx, text in enumerate(info):
            self._text(frame, text, (16.0, 30.0 + idx * line_h))

    def render(self, status: Dict[str, object]) -> "np.ndarray":
        frame = self._background.copy()
        signals: Dict[str, str] = status["signals"]  # type: ignore[assignment]

        for name, approach in self.approaches.items():
            color = self.APPROACH_COLORS[name]
            for vehicle in approach.vehicles:
                x, y, w, h = self._vehicle_rect(name, vehicle)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (25, 25, 25), 1)

        for name in APPROACHES:
            self._draw_signal_housing(frame, name, signals[name])

        self._draw_hud(frame, status)
        return frame

    # -- loop ------------------------------------------------------------------
    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        display_window: bool = True,
        window_name: str = "Case 3 - Four-Way Intersection",
        fullscreen: bool = False,
    ) -> None:
        logger.info("Four-way intersection simulation started. Press 'q' to quit.")
        dt = 1.0 / float(self.fps)
        frame_count = 0
        fullscreen_active = fullscreen and display_window

        if display_window:
            # KEEPRATIO letterboxes instead of stretching when the user
            # resizes or maximizes the window on any monitor shape.
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window_name, self.size, self.size)
            state = cv2.WINDOW_FULLSCREEN if fullscreen_active else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, state)

        try:
            while max_frames is None or frame_count < max_frames:
                status = self.step(dt)
                if display_window:
                    cv2.imshow(window_name, self.render(status))
                    key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
                    if key == ord("q"):
                        break
                    if key in (ord("f"), ord("F")):
                        fullscreen_active = not fullscreen_active
                        state = (
                            cv2.WINDOW_FULLSCREEN if fullscreen_active else cv2.WINDOW_NORMAL
                        )
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, state)
                frame_count += 1
        finally:
            if display_window:
                cv2.destroyAllWindows()
            logger.info(
                "Four-way simulation done. %d switches (%d early / demand-driven).",
                self.controller.total_switches,
                self.controller.early_switches,
            )


# ---------------------------------------------------------------------------
# Real (prerecorded video) mode — shadow-mode demonstration
# ---------------------------------------------------------------------------

#: Default fractional zones (x, y, w, h), calibrated for
#: ``videos/sherbrooke_intersection.avi`` (Urban Tracker "Sherbrooke"): the
#: camera looks along the main road (N = the visible queue approaching the
#: stop line), with the cross street entering from the left (W) and the
#: foreground (S/E). Calibrate per camera for other footage; see README.
DEFAULT_ZONES: Dict[str, Tuple[float, float, float, float]] = {
    "N": (0.30, 0.22, 0.28, 0.30),
    "S": (0.25, 0.72, 0.55, 0.26),
    "E": (0.72, 0.35, 0.28, 0.28),
    "W": (0.00, 0.45, 0.22, 0.30),
}

ZONE_COLORS = {
    "N": (70, 180, 255),
    "S": (255, 180, 70),
    "E": (120, 255, 120),
    "W": (200, 120, 255),
}


class RealFourWayIntersection:
    """Shadow-mode adaptive control over prerecorded overhead footage."""

    def __init__(
        self,
        video_path: str | Path,
        *,
        zones: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        detector_config=None,
        smoothing_window: int = 10,
    ) -> None:
        if cv2 is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "opencv-python is required for the real four-way demo."
            ) from _CV2_IMPORT_ERROR

        from smart_traffic_system import DetectorConfig, VehicleDetector

        self.video_path = Path(video_path)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0

        from pedestrian_crossing import validate_fractional_rect

        self.detector = VehicleDetector(detector_config or DetectorConfig())
        self.zones = dict(zones or DEFAULT_ZONES)
        missing = set(APPROACHES) - set(self.zones)
        if missing:
            raise ValueError(f"zones must cover all approaches; missing {sorted(missing)}")
        for name, rect in self.zones.items():
            validate_fractional_rect(f"zone {name}", rect)

        self._frame_index = 0
        self.controller = FourWayController(
            time_func=lambda: self._frame_index / self.fps
        )
        self._count_history: Dict[str, Deque[int]] = {
            name: deque(maxlen=max(1, smoothing_window)) for name in APPROACHES
        }

    def _zone_pixels(self, frame_shape: Tuple[int, ...], name: str) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        fx, fy, fw, fh = self.zones[name]
        return (int(fx * w), int(fy * h), int(fw * w), int(fh * h))

    def _smoothed_counts(self, raw: Dict[str, int]) -> Dict[str, int]:
        smoothed = {}
        for name in APPROACHES:
            history = self._count_history[name]
            history.append(raw.get(name, 0))
            smoothed[name] = int(round(sum(history) / len(history)))
        return smoothed

    def process_frame(self, frame: "np.ndarray") -> Tuple["np.ndarray", Dict[str, object]]:
        detections = self.detector.detect_vehicles(frame)

        raw_counts = {name: 0 for name in APPROACHES}
        zone_px = {name: self._zone_pixels(frame.shape, name) for name in APPROACHES}
        for det in detections:
            x, y, w, h = det.bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            for name, (zx, zy, zw, zh) in zone_px.items():
                if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
                    raw_counts[name] += 1
                    break

        counts = self._smoothed_counts(raw_counts)
        self._frame_index += 1
        status = self.controller.update(counts)
        annotated = self._annotate(frame, status, detections, zone_px, counts)
        return annotated, status

    def _annotate(
        self,
        frame: "np.ndarray",
        status: Dict[str, object],
        detections,
        zone_px: Dict[str, Tuple[int, int, int, int]],
        counts: Dict[str, int],
    ) -> "np.ndarray":
        out = frame.copy()
        signals: Dict[str, str] = status["signals"]  # type: ignore[assignment]

        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (80, 255, 120), 2)

        for name, (zx, zy, zw, zh) in zone_px.items():
            signal = signals[name]
            color = {"GREEN": (0, 210, 0), "YELLOW": (0, 210, 230)}.get(signal, (0, 0, 220))
            cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), ZONE_COLORS[name], 2)
            cv2.putText(
                out,
                f"{name}: {counts[name]} [{signal}]",
                (zx + 4, zy + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.rectangle(out, (14, 14), (330, 96), (40, 40, 40), -1)
        lines = [
            "SHADOW MODE - adaptive plan",
            f"Axis {status['active_axis']} {status['phase']}",
            f"Early switches: {status['early_switches']} / {status['total_switches']}",
        ]
        for idx, text in enumerate(lines):
            cv2.putText(out, text, (24, 40 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        display_window: bool = True,
        window_name: str = "Case 3 - Four-Way Intersection (Real)",
        fullscreen: bool = False,
    ) -> None:
        logger.info("Real four-way demo on %s. Press 'q' to quit.", self.video_path)
        frame_count = 0
        fullscreen_active = fullscreen and display_window

        if display_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            state = cv2.WINDOW_FULLSCREEN if fullscreen_active else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, state)

        read_failures = 0
        try:
            while max_frames is None or frame_count < max_frames:
                ok, frame = self.capture.read()
                if not ok:
                    read_failures += 1
                    if read_failures > 1:
                        raise RuntimeError(
                            f"Unable to read frames from {self.video_path} even after "
                            "rewinding — the file may be corrupt or truncated."
                        )
                    logger.info("End of video reached; looping playback.")
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                read_failures = 0
                annotated, _ = self.process_frame(frame)
                if display_window:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key in (ord("f"), ord("F")):
                        fullscreen_active = not fullscreen_active
                        state = (
                            cv2.WINDOW_FULLSCREEN if fullscreen_active else cv2.WINDOW_NORMAL
                        )
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, state)
                frame_count += 1
        finally:
            self.capture.release()
            if display_window:
                cv2.destroyAllWindows()
