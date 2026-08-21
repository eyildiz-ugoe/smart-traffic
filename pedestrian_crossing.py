"""Case 1 — Adaptive pedestrian crossing (single road + crosswalk).

Demonstrates demand-driven signalization at its simplest:

* No pedestrian waiting  -> the car light stays GREEN indefinitely (no
  pointless red phases, no idling, no wasted fuel).
* A pedestrian arrives   -> after a minimum car-green time the controller
  switches to a protected WALK phase, but only when it is safe to do so.

Safety design ("cars go so fast" concern):
The controller never starts a yellow phase while a vehicle is inside the
*dilemma zone* — the stretch just before the stop line where a fast car can
neither stop comfortably nor clear before the red. The switch is deferred to
the next safe gap. A waiting-time cap guarantees the pedestrian is eventually
served even under constant traffic.

Both a synthetic simulation and a real prerecorded-video mode are provided.
The real mode runs in *shadow mode*: it detects pedestrians and vehicles with
YOLOv8 and displays the signal decisions the controller WOULD make for that
scene, which is how adaptive-signal pilots are validated in practice before
touching live signal hardware.
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

PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = (2, 3, 5, 7)


class PedestrianSignalController:
    """State machine for a mid-block pedestrian crossing.

    States and transitions::

        CAR_GREEN --(pedestrian waiting, min green served, safe gap)--> CAR_YELLOW
        CAR_YELLOW --(yellow time)--> ALL_RED
        ALL_RED   --(clearance)-----> WALK
        WALK      --(walk time)-----> PED_CLEARANCE
        PED_CLEARANCE --(clearance)-> CAR_GREEN
    """

    MIN_CAR_GREEN = 6.0
    YELLOW_TIME = 3.0
    ALL_RED_TIME = 2.0
    WALK_TIME = 8.0
    PED_CLEARANCE_TIME = 4.0
    #: A pedestrian is never left waiting longer than this, even under
    #: continuous traffic; the controller then waits only for the dilemma
    #: zone to clear before starting the change.
    MAX_PED_WAIT = 45.0
    #: A waiting pedestrian's clock survives detection dropouts shorter than
    #: this (occlusion by passing vehicles must not reset the fairness cap).
    PED_ABSENCE_RESET = 3.0

    def __init__(self, time_func: Callable[[], float] | None = None) -> None:
        self._time_func = time_func or time.monotonic
        now = self._time_func()
        self.state = "CAR_GREEN"
        self.state_start = now
        self._ped_wait_start: Optional[float] = None
        self._ped_last_seen: Optional[float] = None
        self.pedestrians_served = 0
        self.total_ped_wait = 0.0

    # -- helpers ---------------------------------------------------------
    def _elapsed(self) -> float:
        return self._time_func() - self.state_start

    def _enter(self, state: str) -> None:
        self.state = state
        self.state_start = self._time_func()

    def pedestrian_wait_time(self) -> float:
        if self._ped_wait_start is None:
            return 0.0
        return self._time_func() - self._ped_wait_start

    # -- main update -----------------------------------------------------
    def update(
        self,
        pedestrian_waiting: bool,
        vehicle_in_dilemma_zone: bool = False,
        vehicle_count: int = 0,
    ) -> Dict[str, object]:
        """Advance the state machine with the latest detections."""

        now = self._time_func()

        if self.state == "CAR_GREEN":
            if pedestrian_waiting:
                if self._ped_wait_start is None:
                    self._ped_wait_start = now
                self._ped_last_seen = now
            elif self._ped_wait_start is not None:
                # Only forget the waiting pedestrian after a sustained
                # absence — a brief occlusion by passing traffic must not
                # reset the MAX_PED_WAIT fairness clock.
                last_seen = self._ped_last_seen if self._ped_last_seen is not None else now
                if now - last_seen >= self.PED_ABSENCE_RESET:
                    self._ped_wait_start = None
                    self._ped_last_seen = None

            if self._ped_wait_start is not None and self._elapsed() >= self.MIN_CAR_GREEN:
                waited = self.pedestrian_wait_time()
                # Serve at the first gap with no fast vehicle trapped in the
                # dilemma zone (callers must pass a debounced flag — a raw
                # single-frame dropout must never override it). After
                # MAX_PED_WAIT the pedestrian wins regardless: the yellow +
                # all-red clearance still protects any committed vehicle.
                if not vehicle_in_dilemma_zone or waited >= self.MAX_PED_WAIT:
                    self._enter("CAR_YELLOW")
        elif self.state == "CAR_YELLOW":
            if self._elapsed() >= self.YELLOW_TIME:
                self._enter("ALL_RED")
        elif self.state == "ALL_RED":
            if self._elapsed() >= self.ALL_RED_TIME:
                if self._ped_wait_start is not None:
                    self.total_ped_wait += now - self._ped_wait_start
                self.pedestrians_served += 1
                self._ped_wait_start = None
                self._ped_last_seen = None
                self._enter("WALK")
        elif self.state == "WALK":
            if self._elapsed() >= self.WALK_TIME:
                self._enter("PED_CLEARANCE")
        elif self.state == "PED_CLEARANCE":
            if self._elapsed() >= self.PED_CLEARANCE_TIME:
                self._enter("CAR_GREEN")

        return self.status()

    def status(self) -> Dict[str, object]:
        car_signal = {
            "CAR_GREEN": "GREEN",
            "CAR_YELLOW": "YELLOW",
        }.get(self.state, "RED")
        ped_signal = {
            "WALK": "WALK",
            "PED_CLEARANCE": "CLEAR",
        }.get(self.state, "DONT_WALK")
        durations = {
            "CAR_GREEN": None,
            "CAR_YELLOW": self.YELLOW_TIME,
            "ALL_RED": self.ALL_RED_TIME,
            "WALK": self.WALK_TIME,
            "PED_CLEARANCE": self.PED_CLEARANCE_TIME,
        }
        duration = durations[self.state]
        remaining = None if duration is None else max(0.0, duration - self._elapsed())
        return {
            "state": self.state,
            "car_signal": car_signal,
            "ped_signal": ped_signal,
            "time_remaining": remaining,
            "ped_wait_time": self.pedestrian_wait_time(),
            "pedestrians_served": self.pedestrians_served,
        }


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SimulatedPedestrian:
    """A pedestrian waiting at, or walking across, the crosswalk."""

    x: float
    y: float
    direction: int  # +1 walks left->right, -1 walks right->left
    speed: float
    walking: bool = False


class PedestrianCrossingSimulation:
    """Synthetic single-road crossing with demand-actuated pedestrian phase."""

    #: Vehicles closer to the stop line than this cannot stop safely; the
    #: controller must not begin a yellow while one is inside the zone.
    DILEMMA_ZONE_DEPTH = 90.0

    def __init__(
        self,
        fps: int = 30,
        frame_size: Tuple[int, int] = (480, 640),
        *,
        seed: Optional[int] = None,
        car_spawn_rate: float = 0.35,
        pedestrian_rate: float = 0.08,
    ) -> None:
        if cv2 is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "opencv-python is required for the pedestrian crossing simulation."
            ) from _CV2_IMPORT_ERROR
        if np is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "numpy is required for the pedestrian crossing simulation."
            ) from _NUMPY_IMPORT_ERROR

        from smart_traffic_system import SimulatedRoad  # local import avoids cycle

        self.fps = max(1, fps)
        self.frame_height, self.frame_width = frame_size
        self.rng = random.Random(seed)
        self.road = SimulatedRoad(
            "vertical",
            frame_size,
            self.rng,
            spawn_rate=car_spawn_rate,
            max_vehicles=8,
        )
        self.pedestrian_rate = max(0.0, pedestrian_rate)
        self.pedestrians: List[SimulatedPedestrian] = []

        self._sim_time = 0.0
        self.controller = PedestrianSignalController(time_func=lambda: self._sim_time)

        # Crosswalk geometry: a band just downstream of the stop line.
        self.crosswalk_top = self.road.stop_line + 12
        self.crosswalk_bottom = self.crosswalk_top + 46
        self._walk_y = (self.crosswalk_top + self.crosswalk_bottom) // 2
        self._left_wait_x = self.road._lane_left - 26
        self._right_wait_x = self.road._lane_right + 26

        self._background = self._create_background()

    # -- world ------------------------------------------------------------
    def _create_background(self) -> "np.ndarray":
        frame = self.road._create_background()
        stripe_width = 12
        for x in range(self.road._lane_left + 4, self.road._lane_right - 4, stripe_width * 2):
            cv2.rectangle(
                frame,
                (x, self.crosswalk_top),
                (x + stripe_width, self.crosswalk_bottom),
                (210, 210, 210),
                -1,
            )
        return frame

    def _maybe_spawn_pedestrian(self, dt: float) -> None:
        if self.rng.random() < self.pedestrian_rate * dt:
            direction = self.rng.choice((1, -1))
            x = self._left_wait_x if direction == 1 else self._right_wait_x
            self.pedestrians.append(
                SimulatedPedestrian(
                    x=float(x),
                    y=float(self._walk_y + self.rng.uniform(-10, 10)),
                    direction=direction,
                    speed=self.rng.uniform(45.0, 70.0),
                )
            )

    def _update_pedestrians(self, ped_signal: str, dt: float) -> None:
        walk_allowed = ped_signal == "WALK"
        for ped in self.pedestrians:
            if not ped.walking and walk_allowed:
                ped.walking = True
            if ped.walking:
                # Pedestrians already on the road finish crossing during
                # the clearance phase as well.
                ped.x += ped.direction * ped.speed * dt

        self.pedestrians = [
            ped
            for ped in self.pedestrians
            if (self._left_wait_x - 12) <= ped.x <= (self._right_wait_x + 12)
        ]

    def pedestrian_waiting(self) -> bool:
        return any(not ped.walking for ped in self.pedestrians)

    def vehicle_in_dilemma_zone(self) -> bool:
        stop_line = self.road.stop_line
        for vehicle in self.road.vehicles:
            front = vehicle.position + vehicle.length
            if front <= stop_line and (stop_line - front) <= self.DILEMMA_ZONE_DEPTH:
                return True
        return False

    def vehicle_count(self) -> int:
        stop_line = self.road.stop_line
        return sum(
            1
            for vehicle in self.road.vehicles
            if vehicle.position + vehicle.length <= stop_line
        )

    def step(self, dt: float) -> Dict[str, object]:
        """Advance the world and controller by ``dt`` simulated seconds."""

        self._sim_time += dt
        self._maybe_spawn_pedestrian(dt)

        status = self.controller.update(
            pedestrian_waiting=self.pedestrian_waiting(),
            vehicle_in_dilemma_zone=self.vehicle_in_dilemma_zone(),
            vehicle_count=self.vehicle_count(),
        )
        self.road.step(str(status["car_signal"]), dt)
        self._update_pedestrians(str(status["ped_signal"]), dt)
        return status

    # -- rendering ---------------------------------------------------------
    def _draw_ped_signal(self, frame: "np.ndarray", ped_signal: str) -> None:
        x, y = self.road._lane_right + 14, self.crosswalk_top - 66
        cv2.rectangle(frame, (x, y), (x + 92, y + 52), (45, 45, 45), -1)
        color = {
            "WALK": (0, 220, 0),
            "CLEAR": (0, 200, 255),
        }.get(ped_signal, (0, 0, 230))
        label = {"WALK": "WALK", "CLEAR": "CLEAR"}.get(ped_signal, "WAIT")
        cv2.circle(frame, (x + 22, y + 26), 14, color, -1)
        cv2.putText(
            frame, label, (x + 40, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

    def render(self, status: Dict[str, object]) -> "np.ndarray":
        from smart_traffic_system import draw_traffic_light

        frame = self._background.copy()
        self.road.draw_vehicles(frame)

        for ped in self.pedestrians:
            color = (60, 220, 255) if not ped.walking else (80, 255, 120)
            center = (int(ped.x), int(ped.y))
            cv2.circle(frame, center, 8, color, -1)
            cv2.circle(frame, center, 8, (30, 30, 30), 1)

        frame = draw_traffic_light(frame, str(status["car_signal"]), "top-right")
        self._draw_ped_signal(frame, str(status["ped_signal"]))

        wait = float(status["ped_wait_time"])
        remaining = status["time_remaining"]
        info = [
            f"State: {status['state']}",
            f"Cars approaching: {self.vehicle_count()}",
            f"Peds waiting: {sum(1 for p in self.pedestrians if not p.walking)}",
            f"Ped wait: {wait:.1f}s",
            f"Served: {status['pedestrians_served']}",
        ]
        if remaining is not None:
            info.append(f"Phase ends in: {float(remaining):.1f}s")
        if self.vehicle_in_dilemma_zone():
            info.append("Dilemma zone occupied - holding")
        for idx, text in enumerate(info):
            cv2.putText(
                frame, text, (16, 26 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
        return frame

    # -- loop ---------------------------------------------------------------
    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        display_window: bool = True,
        window_name: str = "Case 1 - Pedestrian Crossing",
        fullscreen: bool = False,
    ) -> None:
        logger.info("Pedestrian crossing simulation started. Press 'q' to quit.")
        dt = 1.0 / float(self.fps)
        frame_count = 0
        fullscreen_active = fullscreen and display_window

        if display_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
            served = self.controller.pedestrians_served
            avg_wait = self.controller.total_ped_wait / served if served else 0.0
            logger.info(
                "Pedestrian crossing simulation done. Served %d walk phases, avg wait %.1fs",
                served,
                avg_wait,
            )


# ---------------------------------------------------------------------------
# Real (prerecorded video) mode — shadow-mode demonstration
# ---------------------------------------------------------------------------


def validate_fractional_rect(name: str, rect: Tuple[float, float, float, float]) -> None:
    """Reject degenerate or out-of-frame detection zones at configuration time.

    A silently empty zone would read as "no demand" forever — a starved
    pedestrian or approach with no error message. Fail loudly instead.
    """

    x, y, w, h = rect
    if w <= 0 or h <= 0:
        raise ValueError(f"{name}: zone width/height must be positive, got {rect}")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9):
        raise ValueError(f"{name}: zone must lie within the frame (fractions 0-1), got {rect}")


@dataclass(slots=True)
class ZoneConfig:
    """Fractional rectangles describing where to look in the camera frame.

    The defaults are calibrated for ``videos/rouen_crosswalk.avi`` (Urban
    Tracker "Rouen" sequence): the roadway occupies the left/top of the
    frame while the monitored zebra crossing and its waiting area are in
    the upper-right. Zones may overlap freely — vehicles are only counted
    inside the vehicle zone and persons only inside the pedestrian zone.
    Calibrate per camera for other footage.
    """

    # (x, y, w, h) as fractions of the frame. The vehicle zone covers the
    # corridor where moving vehicles approach the crossing — parked cars at
    # the frame edges are deliberately excluded so they cannot hold the
    # pedestrian phase hostage.
    vehicle_zone: Tuple[float, float, float, float] = (0.10, 0.0, 0.40, 0.55)
    pedestrian_zone: Tuple[float, float, float, float] = (0.52, 0.0, 0.48, 0.50)

    def __post_init__(self) -> None:
        validate_fractional_rect("vehicle_zone", self.vehicle_zone)
        validate_fractional_rect("pedestrian_zone", self.pedestrian_zone)

    def to_pixels(
        self, frame_shape: Tuple[int, ...], which: str
    ) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        fx, fy, fw, fh = getattr(self, which)
        return (int(fx * w), int(fy * h), int(fw * w), int(fh * h))


@dataclass(slots=True)
class _Presence:
    """Debounced boolean presence over a short window of frames."""

    window: int = 8
    threshold: int = 3
    history: Deque[bool] = field(default_factory=deque)

    def update(self, present: bool) -> bool:
        self.history.append(present)
        while len(self.history) > self.window:
            self.history.popleft()
        return sum(self.history) >= self.threshold


class RealPedestrianCrossing:
    """Run the pedestrian-crossing controller on prerecorded footage.

    Shadow mode: detections are real (YOLOv8), while the rendered signals show
    the decisions the adaptive controller would issue for that traffic. This
    is the standard first stage of a signal-control pilot.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        zones: ZoneConfig | None = None,
        detector_config=None,
    ) -> None:
        if cv2 is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "opencv-python is required for the real pedestrian crossing demo."
            ) from _CV2_IMPORT_ERROR

        from smart_traffic_system import DetectorConfig, VehicleDetector

        self.video_path = Path(video_path)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0

        config = detector_config or DetectorConfig(
            classes=[PERSON_CLASS_ID, *VEHICLE_CLASS_IDS]
        )
        self.detector = VehicleDetector(config)
        self.zones = zones or ZoneConfig()

        self._frame_index = 0
        self.controller = PedestrianSignalController(
            time_func=lambda: self._frame_index / self.fps
        )
        self._ped_presence = _Presence()
        self._vehicle_presence = _Presence(window=6, threshold=2)

    @staticmethod
    def _center_in_zone(detection, zone: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = detection.bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        zx, zy, zw, zh = zone
        return zx <= cx <= zx + zw and zy <= cy <= zy + zh

    def process_frame(self, frame: "np.ndarray") -> Tuple["np.ndarray", Dict[str, object]]:
        detections = self.detector.detect_vehicles(frame)
        persons = [d for d in detections if d.class_id == PERSON_CLASS_ID]
        vehicles = [d for d in detections if d.class_id != PERSON_CLASS_ID]

        veh_zone = self.zones.to_pixels(frame.shape, "vehicle_zone")
        ped_zone = self.zones.to_pixels(frame.shape, "pedestrian_zone")

        vehicles_in_zone = [d for d in vehicles if self._center_in_zone(d, veh_zone)]
        persons_in_zone = [d for d in persons if self._center_in_zone(d, ped_zone)]

        ped_waiting = self._ped_presence.update(bool(persons_in_zone))
        vehicle_near = self._vehicle_presence.update(bool(vehicles_in_zone))

        self._frame_index += 1
        status = self.controller.update(
            pedestrian_waiting=ped_waiting,
            vehicle_in_dilemma_zone=vehicle_near,
            vehicle_count=len(vehicles_in_zone),
        )

        annotated = self._annotate(
            frame, status, vehicles, persons, veh_zone, ped_zone,
            len(vehicles_in_zone), len(persons_in_zone),
        )
        status = dict(status)
        status["vehicles_in_zone"] = len(vehicles_in_zone)
        status["persons_in_zone"] = len(persons_in_zone)
        return annotated, status

    def _annotate(
        self,
        frame: "np.ndarray",
        status: Dict[str, object],
        vehicles,
        persons,
        veh_zone: Tuple[int, int, int, int],
        ped_zone: Tuple[int, int, int, int],
        vehicles_in_zone: int,
        persons_in_zone: int,
    ) -> "np.ndarray":
        from smart_traffic_system import draw_traffic_light

        out = frame.copy()
        zx, zy, zw, zh = veh_zone
        cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), (255, 160, 0), 2)
        cv2.putText(out, "VEHICLE ZONE", (zx + 4, zy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)
        zx, zy, zw, zh = ped_zone
        cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), (0, 220, 255), 2)
        cv2.putText(out, "PED ZONE", (zx + 4, zy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        for det in vehicles:
            x, y, w, h = det.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (80, 255, 120), 2)
        for det in persons:
            x, y, w, h = det.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)

        out = draw_traffic_light(out, str(status["car_signal"]), "top-right")
        ped_signal = str(status["ped_signal"])
        color = {"WALK": (0, 220, 0), "CLEAR": (0, 200, 255)}.get(ped_signal, (0, 0, 230))
        cv2.rectangle(out, (20, 20), (250, 118), (40, 40, 40), -1)
        lines = [
            ("SHADOW MODE", (255, 255, 255)),
            (f"Ped signal: {ped_signal}", color),
            (f"Cars in zone: {vehicles_in_zone}", (255, 255, 255)),
            (f"Peds in zone: {persons_in_zone}", (255, 255, 255)),
        ]
        for idx, (text, col) in enumerate(lines):
            cv2.putText(out, text, (30, 44 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        return out

    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        display_window: bool = True,
        window_name: str = "Case 1 - Pedestrian Crossing (Real)",
        fullscreen: bool = False,
    ) -> None:
        logger.info("Real pedestrian-crossing demo on %s. Press 'q' to quit.", self.video_path)
        frame_count = 0
        fullscreen_active = fullscreen and display_window

        if display_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
