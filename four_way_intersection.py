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

#: Human-readable names: an approach is the traffic arriving FROM that
#: compass direction (the "North" approach drives southward, and so on).
APPROACH_NAMES = {"N": "North", "S": "South", "E": "East", "W": "West"}
AXIS_NAMES = {"NS": "North-South", "EW": "East-West"}


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


class FixedCycleController:
    """The dumb baseline: 25 s per axis + yellow + all-red, demand-blind."""

    GREEN = 25.0
    YELLOW = 3.0
    ALL_RED = 1.5

    def signals(self, now: float) -> Dict[str, str]:
        cycle = self.GREEN + self.YELLOW + self.ALL_RED
        phase = now % (2 * cycle)
        axis, offset = ("NS", phase) if phase < cycle else ("EW", phase - cycle)
        if offset < self.GREEN:
            state = "GREEN"
        elif offset < self.GREEN + self.YELLOW:
            state = "YELLOW"
        else:
            state = "RED"
        return {name: (state if AXIS_OF[name] == axis else "RED")
                for name in APPROACHES}


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
    #: Cumulative vehicle-seconds spent waiting (for baseline comparison).
    total_wait: float = 0.0

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
                self.total_wait += dt
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

        # Asymmetric defaults make the demo point obvious: the main NS road
        # is busy while EW only sees the occasional car, so EW red time is
        # skipped whenever nothing is waiting there.
        rates = spawn_rates or {"N": 0.16, "S": 0.14, "E": 0.05, "W": 0.04}
        resolved_seed = seed if seed is not None else random.randrange(1 << 30)
        self.rng = random.Random(resolved_seed)
        self.approaches: Dict[str, Approach] = {
            name: Approach(name=name, spawn_rate=rates.get(name, 0.1))
            for name in APPROACHES
        }
        # Invisible twin: identical traffic (mirrored random stream) under a
        # fixed 25 s cycle, so the HUD can show the measured saving live.
        self._baseline_rng = random.Random(resolved_seed)
        self.baseline_approaches: Dict[str, Approach] = {
            name: Approach(name=name, spawn_rate=rates.get(name, 0.1))
            for name in APPROACHES
        }
        self.baseline_controller = FixedCycleController()

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

    def _text_centered(
        self,
        frame: "np.ndarray",
        text: str,
        logical_center_x: float,
        logical_baseline_y: float,
        color: Tuple[int, int, int] = (255, 255, 255),
        mult: float = 1.0,
    ) -> None:
        """Outlined text horizontally centred on a logical x position."""

        (text_w, _), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale(mult), self._thickness(mult)
        )
        org = (self.px(logical_center_x) - text_w // 2, self.px(logical_baseline_y))
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

    def adaptive_wait(self) -> float:
        return sum(app.total_wait for app in self.approaches.values())

    def baseline_wait(self) -> float:
        return sum(app.total_wait for app in self.baseline_approaches.values())

    def step(self, dt: float) -> Dict[str, object]:
        self._sim_time += dt
        for approach in self.approaches.values():
            approach.maybe_spawn(self.rng, dt)
        status = self.controller.update(self.counts())
        signals: Dict[str, str] = status["signals"]  # type: ignore[assignment]
        for name, approach in self.approaches.items():
            approach.step(signals[name], dt)

        # Twin world under the fixed-time plan (same spawn stream).
        baseline_signals = self.baseline_controller.signals(self._sim_time)
        for approach in self.baseline_approaches.values():
            approach.maybe_spawn(self._baseline_rng, dt)
        for name, approach in self.baseline_approaches.items():
            approach.step(baseline_signals[name], dt)
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
        # Detection zones: the regions the controller actually watches
        # (stop line back to detection_length), shaded per approach.
        det = 110.0  # matches Approach.detection_length
        overlay = frame.copy()
        zones = [
            (px(c - rw), px(c - so - det), px(c), px(c - so)),          # N (west half)
            (px(c), px(c + so), px(c + rw), px(c + so + det)),          # S (east half)
            (px(c + so), px(c - rw), px(c + so + det), px(c)),          # E (north half)
            (px(c - so - det), px(c), px(c - so), px(c + rw)),          # W (south half)
        ]
        for x0, y0, x1, y1 in zones:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (120, 120, 60), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        for x0, y0, x1, y1 in zones:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (160, 160, 90), 1)
        self._text(frame, "detection zone", (c - rw - 4.0, c - so - det - 8.0),
                   color=(190, 190, 130), mult=0.75)

        # Stop lines (one per approach, on the incoming half of each road).
        stop_th = self._thickness(1.5)
        cv2.line(frame, (px(c - rw), px(c - so)), (px(c), px(c - so)), (230, 230, 230), stop_th)  # N
        cv2.line(frame, (px(c), px(c + so)), (px(c + rw), px(c + so)), (230, 230, 230), stop_th)  # S
        cv2.line(frame, (px(c + so), px(c - rw)), (px(c + so), px(c)), (230, 230, 230), stop_th)  # E
        cv2.line(frame, (px(c - so), px(c)), (px(c - so), px(c + rw)), (230, 230, 230), stop_th)  # W

        # Direction-of-travel arrows at the entry of each incoming lane, so
        # it is obvious which way every approach drives.
        lo = self.lane_offset
        arrow_color = (205, 205, 205)
        arrows = [
            ((c - lo, 26.0), (c - lo, 68.0)),                          # North approach: downward
            ((c + lo, LOGICAL_SIZE - 26.0), (c + lo, LOGICAL_SIZE - 68.0)),  # South: upward
            ((LOGICAL_SIZE - 26.0, c - lo), (LOGICAL_SIZE - 68.0, c - lo)),  # East: leftward
            ((26.0, c + lo), (68.0, c + lo)),                          # West: rightward
        ]
        for (x1, y1), (x2, y2) in arrows:
            cv2.arrowedLine(
                frame, (px(x1), px(y1)), (px(x2), px(y2)),
                arrow_color, self._thickness(), tipLength=0.35,
            )
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
        # Full approach name beneath the housing: traffic arriving FROM
        # that compass direction.
        self._text_centered(frame, APPROACH_NAMES[name], lx + w / 2, ly + h + 18.0, mult=0.9)

    def _draw_hud(self, frame: "np.ndarray", status: Dict[str, object]) -> None:
        counts = self.counts()
        axis_name = AXIS_NAMES[str(status["active_axis"])]
        phase = str(status["phase"])
        phase_text = {
            "GREEN": f"{axis_name} road has GREEN",
            "YELLOW": f"{axis_name} road: YELLOW (changing)",
            "ALL_RED": "All red (safety clearance)",
        }[phase]
        info = [
            phase_text,
            "Cars waiting  North:%d  South:%d  East:%d  West:%d"
            % (counts["N"], counts["S"], counts["E"], counts["W"]),
            "Switches: %d (demand-driven: %d)"
            % (status["total_switches"], status["early_switches"]),
            "Avg wait  North-South: %.1fs  East-West: %.1fs"
            % (
                (self.approaches["N"].average_wait() + self.approaches["S"].average_wait()) / 2,
                (self.approaches["E"].average_wait() + self.approaches["W"].average_wait()) / 2,
            ),
        ]
        remaining = status["time_remaining"]
        if remaining is not None:
            info.insert(1, f"Phase ends in: {float(remaining):.1f}s")
        baseline = self.baseline_wait()
        if baseline > 1.0:
            adaptive = self.adaptive_wait()
            pct = 100.0 * (baseline - adaptive) / baseline
            info.append(
                f"Waiting vs fixed 25s timer: {adaptive:.0f}s vs {baseline:.0f}s "
                f"({pct:+.0f}% saved)"
            )

        # Translucent panel sized to the widest line, so no resolution or
        # wording change can push text outside the background.
        line_h = 24.0
        max_text_px = max(
            cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale(), self._thickness()
            )[0][0]
            for text in info
        )
        panel_w = max_text_px / self.scale + 24.0
        panel_h = 12.0 + line_h * len(info)
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

        from demo_ui import draw_controls_hint, handle_display_keys, show_end_card

        action = None
        try:
            while max_frames is None or frame_count < max_frames:
                status = self.step(dt)
                if display_window:
                    frame = self.render(status)
                    draw_controls_hint(frame)
                    cv2.imshow(window_name, frame)
                    action, fullscreen_active = handle_display_keys(
                        window_name, int(1000 / self.fps), fullscreen_active
                    )
                    if action:
                        break
                frame_count += 1
        finally:
            controller = self.controller
            if display_window and action != "exit":
                waits = (
                    (self.approaches["N"].average_wait() + self.approaches["S"].average_wait()) / 2,
                    (self.approaches["E"].average_wait() + self.approaches["W"].average_wait()) / 2,
                )
                baseline = self.baseline_wait()
                adaptive = self.adaptive_wait()
                saved_line = "Twin fixed-timer world: warming up"
                if baseline > 1.0:
                    pct = 100.0 * (baseline - adaptive) / baseline
                    saved_line = (
                        f"Waiting vs fixed 25 s timer: {adaptive:.0f} s vs "
                        f"{baseline:.0f} s  ({pct:+.0f}%)"
                    )
                card_action = show_end_card(
                    window_name,
                    "Case 3 - Four-Way Intersection",
                    [
                        f"Simulated time: {self._sim_time:.0f} s",
                        f"Signal switches: {controller.total_switches} "
                        f"(demand-driven: {controller.early_switches})",
                        saved_line,
                        "Identical cars ran in an invisible twin world under",
                        "a dumb fixed timer - that is the saving.",
                    ],
                )
                action = card_action or action
            if display_window:
                cv2.destroyAllWindows()
            logger.info(
                "Four-way simulation done. %d switches (%d early / demand-driven).",
                controller.total_switches,
                controller.early_switches,
            )
        return action


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

        from motion_filter import MotionFilter
        from pedestrian_crossing import validate_fractional_rect

        self.detector = VehicleDetector(detector_config or DetectorConfig())
        # Parked-car immunity: vehicles are tracked across frames, and a
        # vehicle stationary far longer than any signal cycle (or never seen
        # moving at all) stops counting as demand until it moves again.
        self.motion_filter = MotionFilter()
        # A partial zone set is valid: an approach that is not visible from
        # this camera (auto-calibration may prove there is none) simply never
        # reports demand, and the controller's MAX_RED recall still
        # guarantees it would be served if it existed.
        self.zones = dict(zones or DEFAULT_ZONES)
        unknown = set(self.zones) - set(APPROACHES)
        if unknown:
            raise ValueError(f"unknown approach names in zones: {sorted(unknown)}")
        if not self.zones:
            raise ValueError("zones must cover at least one approach")
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

    def _reset_playback_state(self) -> None:
        """Reset per-video temporal state when the looping video rewinds.

        See RealPedestrianCrossing._reset_playback_state: tracker and count
        smoothing restart, the motion filter keeps only position-stable
        parked-candidate state, and the controller clock stays monotonic.
        """

        now = self._frame_index / self.fps
        self.detector.reset_tracker()
        self.motion_filter.handle_discontinuity(now)
        for history in self._count_history.values():
            history.clear()

    def process_frame(self, frame: "np.ndarray") -> Tuple["np.ndarray", Dict[str, object]]:
        detections = self.detector.track_vehicles(frame)
        now = self._frame_index / self.fps

        # Classify each tracked vehicle as active demand or parked. Untracked
        # detections (no ID yet) count as demand — fail toward serving them.
        self.motion_filter.prune(now)
        active, parked = [], []
        for det in detections:
            if det.track_id is None:
                active.append(det)
                continue
            self.motion_filter.observe(det.track_id, det.center, now)
            if self.motion_filter.is_parked(det.track_id, now):
                parked.append(det)
            else:
                active.append(det)

        raw_counts = {name: 0 for name in APPROACHES}
        zone_px = {name: self._zone_pixels(frame.shape, name) for name in self.zones}
        for det in active:
            cx, cy = det.center
            for name, (zx, zy, zw, zh) in zone_px.items():
                if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
                    raw_counts[name] += 1
                    break

        counts = self._smoothed_counts(raw_counts)
        self._frame_index += 1
        status = self.controller.update(counts)
        annotated = self._annotate(frame, status, active, parked, zone_px, counts)
        status = dict(status)
        status["parked_ignored"] = len(parked)
        return annotated, status

    def _annotate(
        self,
        frame: "np.ndarray",
        status: Dict[str, object],
        active,
        parked,
        zone_px: Dict[str, Tuple[int, int, int, int]],
        counts: Dict[str, int],
    ) -> "np.ndarray":
        out = frame.copy()
        signals: Dict[str, str] = status["signals"]  # type: ignore[assignment]

        def in_any_zone(det) -> bool:
            cx, cy = det.center
            return any(zx <= cx <= zx + zw and zy <= cy <= zy + zh
                       for zx, zy, zw, zh in zone_px.values())

        # Only detections the controller actually counts get bold boxes;
        # out-of-zone hits (road markings, signs, far background) stay a
        # thin unobtrusive gray so they cannot be mistaken for demand.
        for det in active:
            x, y, w, h = det.bbox
            if in_any_zone(det):
                cv2.rectangle(out, (x, y), (x + w, y + h), (80, 255, 120), 2)
            else:
                cv2.rectangle(out, (x, y), (x + w, y + h), (150, 150, 150), 1)
        for det in parked:
            x, y, w, h = det.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (140, 140, 140), 1)
            cv2.putText(
                out, "PARKED", (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA,
            )

        for name, (zx, zy, zw, zh) in zone_px.items():
            signal = signals[name]
            color = {"GREEN": (0, 210, 0), "YELLOW": (0, 210, 230)}.get(signal, (0, 0, 220))
            cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), ZONE_COLORS[name], 2)
            cv2.putText(
                out,
                f"{APPROACH_NAMES[name]}: {counts[name]} [{signal}]",
                (zx + 4, zy + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        # Slim translucent top strip instead of a video-blocking box.
        strip = out.copy()
        cv2.rectangle(strip, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
        cv2.addWeighted(strip, 0.6, out, 0.4, 0, out)
        summary = (
            f"SHADOW MODE   {AXIS_NAMES[str(status['active_axis'])]} {status['phase']}   "
            f"switches {status['total_switches']} (demand {status['early_switches']})   "
            f"parked ignored {len(parked)}"
        )
        cv2.putText(out, summary, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
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

        from demo_ui import draw_controls_hint, handle_display_keys, show_end_card

        read_failures = 0
        action = None
        max_parked = 0
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
                    self._reset_playback_state()
                    continue
                read_failures = 0
                annotated, status = self.process_frame(frame)
                max_parked = max(max_parked, int(status.get("parked_ignored", 0)))
                if display_window:
                    draw_controls_hint(annotated)
                    cv2.imshow(window_name, annotated)
                    action, fullscreen_active = handle_display_keys(
                        window_name, 1, fullscreen_active
                    )
                    if action:
                        break
                frame_count += 1
        finally:
            self.capture.release()
            if display_window and action != "exit":
                controller = self.controller
                card_action = show_end_card(
                    window_name,
                    "Case 3 - Four-Way Intersection (Real, shadow mode)",
                    [
                        f"Frames analysed: {frame_count}",
                        f"Signal switches: {controller.total_switches} "
                        f"(demand-driven: {controller.early_switches})",
                        f"Parked cars excluded from demand (max): {max_parked}",
                        "Real detections, adaptive plan overlaid - no",
                        "signal hardware touched.",
                    ],
                )
                action = card_action or action
            if display_window:
                cv2.destroyAllWindows()
        return action
