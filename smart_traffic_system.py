"""Smart traffic system supporting real and simulated modes."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - import success depends on environment
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:  # pragma: no cover - import success depends on environment
    _NUMPY_IMPORT_ERROR = None

from counter import VehicleCounter
from sorter import VehicleSorter
from traffic_core import TrafficLightController, TrafficStats
from video_downloader import TrafficVideoSetup


logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised at runtime when YOLO is used
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - handled lazily in VehicleDetector
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = exc
else:  # pragma: no cover - import success path depends on runtime environment
    _YOLO_IMPORT_ERROR = None


@dataclass(slots=True)
class DetectorConfig:
    """Configuration options for the YOLO vehicle detector."""

    model_path: str | Path = "weights/yolov8n.pt"
    confidence: float = 0.25
    iou: float = 0.5
    classes: Iterable[int] | None = None
    device: str | None = "cuda"  # Use GPU by default (falls back to CPU if unavailable)
    max_detections: Optional[int] = 100


@dataclass(slots=True)
class VehicleDetection:
    """Container for a single vehicle detection result."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int

    @property
    def bottom_edge(self) -> int:
        x, y, w, h = self.bbox
        return y + h

    @property
    def right_edge(self) -> int:
        x, y, w, h = self.bbox
        return x + w


@dataclass(slots=True)
class QueueMetrics:
    """Aggregate queue information for a single road."""

    count: int
    sorted_detections: List[VehicleDetection]
    pressure: float
    class_breakdown: Dict[int, int]
    approach_line: int
    exit_line: int
    stopline_occupied: bool
    exit_zone_active: bool
    leading_edge: Optional[int]


def draw_traffic_light(frame: np.ndarray, signal: str, position: str) -> np.ndarray:
    """Draw a simple traffic light indicator on ``frame``."""

    h, w = frame.shape[:2]

    if position == "top-left":
        x_offset, y_offset = 20, 20
    elif position == "top-right":
        x_offset, y_offset = w - 120, 20
    elif position == "bottom-left":
        x_offset, y_offset = 20, h - 240
    elif position == "bottom-right":
        x_offset, y_offset = w - 120, h - 240
    else:  # pragma: no cover - defensive fallback for unexpected value
        raise ValueError(f"Unsupported light position: {position}")

    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + 80, y_offset + 200), (50, 50, 50), -1)

    colors = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 255, 0),
    }
    light_positions = {"RED": 50, "YELLOW": 110, "GREEN": 170}

    for light, y_pos in light_positions.items():
        color = colors[light] if light == signal else (80, 80, 80)
        center = (x_offset + 40, y_offset + y_pos)
        cv2.circle(frame, center, 25, color, -1)
        cv2.circle(frame, center, 25, (255, 255, 255), 2)

    return frame


def draw_vehicle_annotations(frame: np.ndarray, metrics: QueueMetrics) -> np.ndarray:
    """Annotate vehicles ordered by queue priority."""

    for index, detection in enumerate(metrics.sorted_detections, start=1):
        x, y, w, h = detection.bbox
        color = (0, 255, 0) if index == 1 else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"#{index} {detection.confidence:.2f}",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame


def draw_threshold_lines(
    frame: np.ndarray, metrics: QueueMetrics, analyzer: VehicleQueueAnalyzer
) -> np.ndarray:
    """Visualise threshold lines used by the queue analyser."""

    h, w = frame.shape[:2]

    if analyzer.orientation == "vertical":
        approach_start = (0, metrics.approach_line)
        approach_end = (w - 1, metrics.approach_line)
        exit_start = (0, metrics.exit_line)
        exit_end = (w - 1, metrics.exit_line)
    else:
        approach_start = (metrics.approach_line, 0)
        approach_end = (metrics.approach_line, h - 1)
        exit_start = (metrics.exit_line, 0)
        exit_end = (metrics.exit_line, h - 1)

    approach_color = (0, 215, 255) if metrics.stopline_occupied else (128, 128, 128)
    exit_color = (0, 255, 0) if metrics.exit_zone_active else (80, 80, 80)

    cv2.line(frame, approach_start, approach_end, approach_color, 2)
    cv2.line(frame, exit_start, exit_end, exit_color, 2)

    return frame


def draw_queue_summary(
    frame: np.ndarray, metrics: QueueMetrics, signal: str, anchor: Tuple[int, int]
) -> np.ndarray:
    """Overlay queue summary information on ``frame``."""

    x, y = anchor
    if metrics.class_breakdown:
        dominant_class = max(metrics.class_breakdown, key=metrics.class_breakdown.get)
        dominant_text = f"Top class: {dominant_class}"
    else:
        dominant_text = "Top class: --"

    info_lines = [
        f"Signal: {signal}",
        f"Vehicles: {metrics.count}",
        f"Queue pressure: {metrics.pressure:.2f}",
        dominant_text,
        f"Stopline: {'occupied' if metrics.stopline_occupied else 'clear'}",
    ]

    for idx, text in enumerate(info_lines):
        cv2.putText(
            frame,
            text,
            (x, y + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return frame


class VehicleDetector:
    """Vehicle detector backed by the Ultralytics YOLOv8 model."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        if YOLO is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "ultralytics is required for YOLO vehicle detection. "
                "Install it with `pip install ultralytics`."
            ) from _YOLO_IMPORT_ERROR

        self.config = config or DetectorConfig()
        classes = list(self.config.classes) if self.config.classes is not None else [2, 3, 5, 7]
        self._target_classes = set(int(cls) for cls in classes)

        model_path = Path(self.config.model_path)
        # YOLO accepts either a local path or a model name; don't resolve unless it exists locally
        self.model = YOLO(str(model_path))
        self.model.fuse()  # type: ignore[no-untyped-call]

    def detect_vehicles(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Detect vehicles on a frame using YOLOv8."""

        results = self.model(
            frame,
            verbose=False,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.config.device,
        )[0]

        detections: List[VehicleDetection] = []

        if not hasattr(results, "boxes") or results.boxes is None:  # pragma: no cover - defensive
            return detections

        for cls, conf, xyxy in zip(results.boxes.cls, results.boxes.conf, results.boxes.xyxy):
            if int(cls) not in self._target_classes:
                continue
            if float(conf) < self.config.confidence:
                continue

            x1, y1, x2, y2 = map(int, xyxy.tolist())
            detections.append(
                VehicleDetection(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=float(conf),
                    class_id=int(cls),
                )
            )

        detections.sort(key=lambda det: det.confidence, reverse=True)

        if self.config.max_detections is not None:
            detections = detections[: self.config.max_detections]

        return detections


class VehicleQueueAnalyzer:
    """Derive queue metrics from raw vehicle detections."""

    def __init__(
        self,
        orientation: str = "vertical",
        *,
        approach_threshold_ratio: float = 0.65,
        exit_margin: int = 5,
    ) -> None:
        if not 0.0 < approach_threshold_ratio < 1.0:
            raise ValueError("approach_threshold_ratio must be between 0 and 1")
        if exit_margin < 0:
            raise ValueError("exit_margin must be non-negative")

        self.sorter = VehicleSorter(orientation=orientation)
        self.counter = VehicleCounter()
        self._approach_threshold_ratio = approach_threshold_ratio
        self._exit_margin = exit_margin

    @property
    def orientation(self) -> str:
        return self.sorter.orientation

    def calculate_metrics(
        self, frame_shape: Tuple[int, int, int], detections: Sequence[VehicleDetection]
    ) -> QueueMetrics:
        sorted_detections = self.sorter.sort(detections)
        pressure = self._calculate_pressure(frame_shape, sorted_detections)

        approach_line, exit_line, stopline_occupied, exit_zone_active, leading_edge = (
            self._calculate_thresholds(frame_shape, sorted_detections)
        )
        count, class_breakdown = self.counter.summarize(sorted_detections)
        return QueueMetrics(
            count=count,
            sorted_detections=list(sorted_detections),
            pressure=pressure,
            class_breakdown=class_breakdown,
            approach_line=approach_line,
            exit_line=exit_line,
            stopline_occupied=stopline_occupied,
            exit_zone_active=exit_zone_active,
            leading_edge=leading_edge,
        )

    def _calculate_pressure(
        self, frame_shape: Tuple[int, int, int], detections: Sequence[VehicleDetection]
    ) -> float:
        """Compute a queue pressure score that weights vehicles by distance.

        The score combines the raw vehicle count with a normalized distance term so
        that vehicles farther from the stop line contribute more pressure. This
        reflects that longer queues should influence the controller to extend the
        green phase.
        """
        if not detections:
            return 0.0

        dimension = frame_shape[0] if self.orientation == "vertical" else frame_shape[1]
        if dimension <= 0:  # pragma: no cover - defensive fallback
            dimension = 1

        normalized_sum = 0.0
        for detection in detections:
            edge = detection.bottom_edge if self.orientation == "vertical" else detection.right_edge
            normalized_distance = 1.0 - max(0.0, min(edge / float(dimension), 1.0))
            normalized_sum += normalized_distance

        return len(detections) + normalized_sum

    def _calculate_thresholds(
        self, frame_shape: Tuple[int, int, int], detections: Sequence[VehicleDetection]
    ) -> Tuple[int, int, bool, bool, Optional[int]]:
        """Compute threshold positions and occupancy flags for the lane."""

        dimension = frame_shape[0] if self.orientation == "vertical" else frame_shape[1]
        if dimension <= 0:
            dimension = 1

        approach_line = int(round(dimension * self._approach_threshold_ratio))
        approach_line = max(0, min(dimension - 1, approach_line))

        exit_line = max(0, dimension - 1 - self._exit_margin)

        if self.orientation == "vertical":
            edges = [det.bottom_edge for det in detections]
        else:
            edges = [det.right_edge for det in detections]

        leading_edge = edges[0] if edges else None
        stopline_occupied = any(edge >= approach_line for edge in edges)
        exit_zone_active = any(edge >= exit_line for edge in edges)

        return approach_line, exit_line, stopline_occupied, exit_zone_active, leading_edge


class SmartTrafficSystem:
    """Main system that integrates YOLO detection with traffic light control."""

    def __init__(
        self,
        video_road1: str,
        video_road2: str,
        detector_config: DetectorConfig | None = None,
        orientation_road1: str = "vertical",
        orientation_road2: str = "vertical",
    ) -> None:
        """
        Initialize the smart traffic system.

        Args:
            video_road1: Path to video file for road 1
            video_road2: Path to video file for road 2
            detector_config: Optional configuration for the YOLO detector
            orientation_road1: Orientation of traffic flow in road 1 feed ('vertical' or 'horizontal')
            orientation_road2: Orientation of traffic flow in road 2 feed ('vertical' or 'horizontal')
        """
        if cv2 is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "opencv-python is required to run the smart traffic system. "
                "Install it with `pip install opencv-python`."
            ) from _CV2_IMPORT_ERROR

        if np is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "numpy is required to run the smart traffic system. "
                "Install it with `pip install numpy`."
            ) from _NUMPY_IMPORT_ERROR

        # Initialize video captures
        self.cap_road1 = cv2.VideoCapture(video_road1)
        self.cap_road2 = cv2.VideoCapture(video_road2)

        # Check if videos opened successfully
        if not self.cap_road1.isOpened():
            raise ValueError(f"Unable to open video: {video_road1}")
        if not self.cap_road2.isOpened():
            raise ValueError(f"Unable to open video: {video_road2}")
        
        # Initialize detector shared between both roads
        self.detector = VehicleDetector(detector_config)

        # Queuing heuristics per road (support cameras pointing in different directions)
        self.queue_analyzer_road1 = VehicleQueueAnalyzer(
            orientation=orientation_road1,
            approach_threshold_ratio=0.55,
        )
        self.queue_analyzer_road2 = VehicleQueueAnalyzer(orientation=orientation_road2)

        # Initialize traffic light controller
        self.controller = TrafficLightController()

        # Initialize statistics
        self.stats_road1 = TrafficStats()
        self.stats_road2 = TrafficStats()

        self.last_metrics_road1: QueueMetrics | None = None
        self.last_metrics_road2: QueueMetrics | None = None
        
        # Get video properties
        self.fps = int(self.cap_road1.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            self.fps = 30  # Default FPS
        
    def _process_road(
        self, frame: np.ndarray, analyzer: VehicleQueueAnalyzer
    ) -> Tuple[QueueMetrics, np.ndarray]:
        detections = self.detector.detect_vehicles(frame)
        metrics = analyzer.calculate_metrics(frame.shape, detections)
        annotated_frame = draw_vehicle_annotations(frame, metrics)
        annotated_frame = draw_threshold_lines(annotated_frame, metrics, analyzer)
        return metrics, annotated_frame

    def run(self):
        """
        Main loop to run the smart traffic system.
        """
        logger.info("Smart Traffic Light Automation System initialised. Press 'q' to quit.")

        frame_count = 0
        
        try:
            while True:
                # Read frames from both videos
                ret1, frame1 = self.cap_road1.read()
                ret2, frame2 = self.cap_road2.read()
                
                # Check if we've reached the end of either video
                if not ret1 or not ret2:
                    logger.info("End of video reached. Restarting playback from the beginning.")
                    self.cap_road1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap_road2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.queue_analyzer_road1.counter.reset()
                    self.queue_analyzer_road2.counter.reset()
                    continue
                
                # Resize frames for better visualization
                frame1 = cv2.resize(frame1, (640, 480))
                frame2 = cv2.resize(frame2, (640, 480))
                
                # Detect vehicles and build queue metrics on both roads
                metrics1, frame1 = self._process_road(frame1, self.queue_analyzer_road1)
                metrics2, frame2 = self._process_road(frame2, self.queue_analyzer_road2)

                self.last_metrics_road1 = metrics1
                self.last_metrics_road2 = metrics2

                # Update statistics
                self.stats_road1.update(metrics1.count)
                self.stats_road2.update(metrics2.count)

                # Update traffic signal timing with queue pressure heuristics
                signal_status = self.controller.update_signal_timing(
                    metrics1.count,
                    metrics2.count,
                    road1_queue_pressure=metrics1.pressure,
                    road2_queue_pressure=metrics2.pressure,
                    road1_stopline_occupied=metrics1.stopline_occupied,
                    road2_stopline_occupied=metrics2.stopline_occupied,
                    road1_exit_ready=metrics1.exit_zone_active or metrics1.count == 0,
                    road2_exit_ready=metrics2.exit_zone_active or metrics2.count == 0,
                    road1_leading_edge=metrics1.leading_edge,
                    road2_leading_edge=metrics2.leading_edge,
                    road1_approach_line=metrics1.approach_line,
                    road2_approach_line=metrics2.approach_line,
                )

                # Draw traffic lights and queue summaries
                frame1 = draw_traffic_light(frame1, signal_status['road1'], 'top-right')
                frame2 = draw_traffic_light(frame2, signal_status['road2'], 'top-right')

                frame1 = draw_queue_summary(
                    frame1,
                    metrics1,
                    signal_status['road1'],
                    (20, frame1.shape[0] - 60),
                )
                frame2 = draw_queue_summary(
                    frame2,
                    metrics2,
                    signal_status['road2'],
                    (20, frame2.shape[0] - 60),
                )

                # Combine frames side by side
                combined_frame = np.hstack([frame1, frame2])

                # Display the result
                cv2.imshow('Smart Traffic Light System', combined_frame)

                # Print statistics every 30 frames
                if frame_count % 30 == 0:
                    logger.debug(
                        "Frame %d | Road1: vehicles=%d pressure=%.2f congestion=%s signal=%s | Road2: vehicles=%d pressure=%.2f congestion=%s signal=%s",
                        frame_count,
                        metrics1.count,
                        metrics1.pressure,
                        self.stats_road1.congestion_level,
                        signal_status["road1"],
                        metrics2.count,
                        metrics2.pressure,
                        self.stats_road2.congestion_level,
                        signal_status["road2"],
                    )
                
                frame_count += 1
                
                # Check for quit command
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                    
        except Exception:
            logger.exception("Error occurred during smart traffic system execution.")
            raise
        finally:
            # Cleanup
            self.cap_road1.release()
            self.cap_road2.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            logger.info(
                "Run complete. Avg vehicles per frame -> Road1: %.2f, Road2: %.2f",
                self.stats_road1.avg_vehicles_per_frame,
                self.stats_road2.avg_vehicles_per_frame,
            )
            if self.last_metrics_road1 is not None:
                logger.debug(
                    "Final Road1 metrics: pressure=%.2f vehicles=%d",
                    self.last_metrics_road1.pressure,
                    self.last_metrics_road1.count,
                )
            if self.last_metrics_road2 is not None:
                logger.debug(
                    "Final Road2 metrics: pressure=%.2f vehicles=%d",
                    self.last_metrics_road2.pressure,
                    self.last_metrics_road2.count,
                )


@dataclass(slots=True)
class SimulatedVehicle:
    """Lightweight vehicle representation used by the simulation mode."""

    position: float
    speed: float
    length: int
    width: int
    color: Tuple[int, int, int]


class SimulatedRoad:
    """Maintain vehicles and drawings for a synthetic traffic lane."""

    def __init__(
        self,
        orientation: str,
        frame_size: Tuple[int, int],
        rng: random.Random,
        *,
        spawn_rate: float = 3.5,
        max_vehicles: int = 10,
    ) -> None:
        if orientation not in {"vertical", "horizontal"}:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        self.orientation = orientation
        self.frame_height, self.frame_width = frame_size
        self.rng = rng
        self.spawn_rate = max(0.0, spawn_rate)
        self.max_vehicles = max_vehicles

        self.vehicles: List[SimulatedVehicle] = []
        self.min_gap = 12

        if orientation == "vertical":
            self.vehicle_length, self.vehicle_width = 70, 40
            self.stop_line = self.frame_height // 2 - 30
            self._despawn_limit = self.frame_height
            self._lane_left = self.frame_width // 2 - 60
            self._lane_right = self.frame_width // 2 + 60
        else:
            self.vehicle_length, self.vehicle_width = 40, 70
            merge_entry = self.frame_width // 2 - 70
            merge_exit = self.frame_width // 2 + 70
            self.stop_line = max(0, merge_entry - 20)
            self._despawn_limit = min(self.frame_width, merge_exit + 160)
            self._lane_left = 0
            self._lane_right = merge_exit

        self.background = self._create_background()

    def _create_background(self) -> np.ndarray:
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        road_color = (65, 65, 65)
        line_color = (200, 200, 200)

        if self.orientation == "vertical":
            lane_left = self._lane_left
            lane_right = self._lane_right
            cv2.rectangle(frame, (lane_left, 0), (lane_right, self.frame_height), road_color, -1)
            cv2.line(frame, (self.frame_width // 2, 0), (self.frame_width // 2, self.frame_height), line_color, 2)
            cv2.line(frame, (lane_left, self.stop_line), (lane_right, self.stop_line), (0, 0, 0), 2)
        else:
            lane_top = self.frame_height // 2 - 60
            lane_bottom = self.frame_height // 2 + 60
            cv2.rectangle(frame, (self._lane_left, lane_top), (self._lane_right, lane_bottom), road_color, -1)
            cv2.line(frame, (self._lane_left, self.frame_height // 2), (self._lane_right, self.frame_height // 2), line_color, 2)
            cv2.line(frame, (self.stop_line, lane_top), (self.stop_line, lane_bottom), (0, 0, 0), 2)
            merge_mark_right = min(self.frame_width - 1, self._lane_right + 20)
            cv2.line(
                frame,
                (merge_mark_right, lane_top),
                (merge_mark_right, lane_bottom),
                (30, 30, 30),
                2,
            )

        return frame

    def _new_vehicle(self) -> SimulatedVehicle:
        base_speed = 180 if self.orientation == "vertical" else 170
        speed_variation = self.rng.uniform(-40, 30)
        color_options = [(66, 245, 189), (66, 134, 244), (244, 199, 66), (240, 96, 96)]
        color = self.rng.choice(color_options)

        if self.orientation == "vertical":
            start_position = -self.vehicle_length - self.rng.uniform(10, 80)
        else:
            start_position = -self.vehicle_length - self.rng.uniform(10, 80)

        return SimulatedVehicle(
            position=start_position,
            speed=max(40.0, base_speed + speed_variation),
            length=self.vehicle_length,
            width=self.vehicle_width,
            color=color,
        )

    def _maybe_spawn(self, dt: float) -> None:
        if len(self.vehicles) >= self.max_vehicles:
            return
        probability = self.spawn_rate * dt
        if self.rng.random() < probability:
            self.vehicles.append(self._new_vehicle())

    def _update_vehicle_positions(self, signal: str, dt: float) -> None:
        if self.orientation == "vertical":
            self._update_vertical(signal, dt)
        else:
            self._update_horizontal(signal, dt)

    def _update_vertical(self, signal: str, dt: float) -> None:
        if not self.vehicles:
            return

        self.vehicles.sort(key=lambda vehicle: vehicle.position + vehicle.length, reverse=True)
        next_front_limit: float = float("inf")

        for vehicle in self.vehicles:
            current_front = vehicle.position + vehicle.length
            target_front = current_front + vehicle.speed * dt

            if signal == "RED" and current_front <= self.stop_line:
                target_front = min(target_front, self.stop_line)
            elif signal == "YELLOW" and current_front < self.stop_line:
                target_front = min(target_front, self.stop_line)

            if next_front_limit != float("inf"):
                target_front = min(target_front, next_front_limit - self.min_gap)

            target_front = max(target_front, current_front)
            vehicle.position = target_front - vehicle.length
            next_front_limit = vehicle.position

        self.vehicles = [v for v in self.vehicles if v.position < self.frame_height]

    def _update_horizontal(self, signal: str, dt: float) -> None:
        if not self.vehicles:
            return

        self.vehicles.sort(key=lambda vehicle: vehicle.position + vehicle.length, reverse=True)
        next_front_limit: float = float("inf")

        for vehicle in self.vehicles:
            current_front = vehicle.position + vehicle.length
            target_front = current_front + vehicle.speed * dt

            if signal == "RED" and current_front <= self.stop_line:
                target_front = min(target_front, self.stop_line)
            elif signal == "YELLOW" and current_front < self.stop_line:
                target_front = min(target_front, self.stop_line)

            if next_front_limit != float("inf"):
                target_front = min(target_front, next_front_limit - self.min_gap)

            target_front = max(target_front, current_front)
            vehicle.position = target_front - vehicle.length
            next_front_limit = vehicle.position

        self.vehicles = [v for v in self.vehicles if v.position < self._despawn_limit]

    def step(self, signal: str, dt: float) -> None:
        self._maybe_spawn(dt)
        self._update_vehicle_positions(signal, dt)

    def _draw_vehicle(self, frame: np.ndarray, vehicle: SimulatedVehicle) -> None:
        if self.orientation == "vertical":
            x = self.frame_width // 2 - self.vehicle_width // 2
            top = int(vehicle.position)
            cv2.rectangle(
                frame,
                (x, top),
                (x + self.vehicle_width, top + self.vehicle_length),
                vehicle.color,
                -1,
            )
        else:
            y = self.frame_height // 2 - self.vehicle_width // 2
            left = int(vehicle.position)
            cv2.rectangle(
                frame,
                (left, y),
                (left + self.vehicle_length, y + self.vehicle_width),
                vehicle.color,
                -1,
            )

    def draw_vehicles(self, frame: np.ndarray) -> None:
        """Render the road's vehicles onto ``frame`` in-place."""

        if frame.shape[:2] != (self.frame_height, self.frame_width):  # pragma: no cover - defensive
            raise ValueError("Frame size mismatch when drawing simulated vehicles")

        for vehicle in self.vehicles:
            self._draw_vehicle(frame, vehicle)

    def render_frame(self) -> np.ndarray:
        frame = self.background.copy()
        self.draw_vehicles(frame)
        return frame

    def detections(self) -> List[VehicleDetection]:
        detections: List[VehicleDetection] = []

        if self.orientation == "vertical":
            x = self.frame_width // 2 - self.vehicle_width // 2
            for vehicle in self.vehicles:
                detections.append(
                    VehicleDetection(
                        bbox=(x, int(vehicle.position), self.vehicle_width, self.vehicle_length),
                        confidence=1.0,
                        class_id=2,
                    )
                )
        else:
            y = self.frame_height // 2 - self.vehicle_width // 2
            for vehicle in self.vehicles:
                detections.append(
                    VehicleDetection(
                        bbox=(int(vehicle.position), y, self.vehicle_length, self.vehicle_width),
                        confidence=1.0,
                        class_id=2,
                    )
                )

        return detections


class SimulationTrafficSystem:
    """Generate synthetic frames and queue data without using a camera feed."""

    def __init__(
        self,
        fps: int = 30,
        frame_size: Tuple[int, int] = (480, 640),
        *,
        seed: Optional[int] = None,
        spawn_rate: float = 3.5,
    ) -> None:
        if cv2 is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "opencv-python is required to run the simulation mode. "
                "Install it with `pip install opencv-python`."
            ) from _CV2_IMPORT_ERROR

        if np is None:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "numpy is required to run the simulation mode. "
                "Install it with `pip install numpy`."
            ) from _NUMPY_IMPORT_ERROR

        self.fps = max(1, fps)
        self.frame_shape = (frame_size[0], frame_size[1], 3)
        rng = random.Random(seed)

        self.road1 = SimulatedRoad("vertical", frame_size, rng, spawn_rate=spawn_rate)
        self.road2 = SimulatedRoad("horizontal", frame_size, rng, spawn_rate=spawn_rate)
        self._scene_background = self._create_scene_background()

        self.queue_analyzer_road1 = VehicleQueueAnalyzer(orientation="vertical")
        self.queue_analyzer_road2 = VehicleQueueAnalyzer(orientation="horizontal")

        self.controller = TrafficLightController()
        self.stats_road1 = TrafficStats()
        self.stats_road2 = TrafficStats()

        self.last_metrics_road1: QueueMetrics | None = None
        self.last_metrics_road2: QueueMetrics | None = None

        self._current_signal = self.controller.update_signal_timing(
            0,
            0,
            road1_queue_pressure=0.0,
            road2_queue_pressure=0.0,
            road1_stopline_occupied=False,
            road2_stopline_occupied=False,
            road1_exit_ready=True,
            road2_exit_ready=True,
            road1_leading_edge=None,
            road2_leading_edge=None,
            road1_approach_line=frame_size[0] // 2,
            road2_approach_line=frame_size[1] // 2,
        )

    def _create_scene_background(self) -> np.ndarray:
        """Combine road backgrounds into a single intersection view."""

        base = np.full(self.frame_shape, 20, dtype=np.uint8)
        road_overlay = np.maximum(self.road1.background, self.road2.background)
        mask = road_overlay > 0
        base[mask] = road_overlay[mask]
        return base

    def _process_simulated_road(
        self, road: SimulatedRoad, analyzer: VehicleQueueAnalyzer
    ) -> QueueMetrics:
        detections = road.detections()
        metrics = analyzer.calculate_metrics(self.frame_shape, detections)
        return metrics

    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        display_window: bool = True,
        window_name: str = "Smart Traffic Simulation",
    ) -> None:
        logger.info("Simulation mode initialised. Press 'q' to quit.")

        dt = 1.0 / float(self.fps)
        frame_count = 0

        try:
            while max_frames is None or frame_count < max_frames:
                self.road1.step(self._current_signal["road1"], dt)
                self.road2.step(self._current_signal["road2"], dt)

                metrics1 = self._process_simulated_road(
                    self.road1, self.queue_analyzer_road1
                )
                metrics2 = self._process_simulated_road(
                    self.road2, self.queue_analyzer_road2
                )

                self.last_metrics_road1 = metrics1
                self.last_metrics_road2 = metrics2

                self.stats_road1.update(metrics1.count)
                self.stats_road2.update(metrics2.count)

                # Use time-based switching for simulation (simpler and more predictable)
                self._current_signal = self.controller.update_signal_timing(
                    metrics1.count,
                    metrics2.count,
                    road1_queue_pressure=metrics1.pressure,
                    road2_queue_pressure=metrics2.pressure,
                )

                frame = self._scene_background.copy()
                self.road1.draw_vehicles(frame)
                self.road2.draw_vehicles(frame)

                frame = draw_vehicle_annotations(frame, metrics1)
                frame = draw_vehicle_annotations(frame, metrics2)
                frame = draw_threshold_lines(frame, metrics1, self.queue_analyzer_road1)
                frame = draw_threshold_lines(frame, metrics2, self.queue_analyzer_road2)

                frame = draw_traffic_light(frame, self._current_signal["road1"], "top-right")
                frame = draw_traffic_light(frame, self._current_signal["road2"], "bottom-left")

                frame = draw_queue_summary(
                    frame,
                    metrics1,
                    self._current_signal["road1"],
                    (frame.shape[1] - 240, 40),
                )
                frame = draw_queue_summary(
                    frame,
                    metrics2,
                    self._current_signal["road2"],
                    (20, frame.shape[0] - 120),
                )

                if display_window:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord("q"):
                        break

                frame_count += 1

        finally:
            if display_window:
                cv2.destroyAllWindows()

            logger.info(
                "Simulation complete. Avg vehicles per frame -> Road1: %.2f, Road2: %.2f",
                self.stats_road1.avg_vehicles_per_frame,
                self.stats_road2.avg_vehicles_per_frame,
            )

SetupFactory = Callable[[Path], TrafficVideoSetup]


def resolve_video_sources(
    candidate_pairs: Sequence[Tuple[Path, Path]] | None = None,
    setup_factory: SetupFactory | None = None,
) -> Tuple[str, str]:
    """Locate or synthesise the preferred pair of input videos for the system."""

    pairs: List[Tuple[Path, Path]] = [
        (Path("videos") / "road1.mp4", Path("videos") / "road2.mp4"),
        (Path("road1.mp4"), Path("road2.mp4")),
    ]

    if candidate_pairs is not None:
        pairs = [
            (Path(path1), Path(path2))
            for path1, path2 in candidate_pairs
        ]

    def _existing_pair() -> Tuple[str, str] | None:
        for first, second in pairs:
            if first.exists() and second.exists():
                return str(first), str(second)
        return None

    existing = _existing_pair()
    if existing is not None:
        return existing

    fallback_first, fallback_second = pairs[-1]
    fallback_dir = fallback_first.parent

    if setup_factory is None:
        def setup_factory(output_dir: Path) -> TrafficVideoSetup:
            return TrafficVideoSetup(str(output_dir))

    try:
        setup_helper = setup_factory(fallback_dir)
    except (OSError, RuntimeError, ValueError, CalledProcessError) as exc:
        logger.warning(
            "Failed to initialize TrafficVideoSetup in %s: %s", fallback_dir, exc
        )
        setup_helper = None

    if setup_helper is not None:
        try:
            if setup_helper.verify_setup():
                existing = _existing_pair()
                if existing is not None:
                    return existing
        except (OSError, RuntimeError, ValueError, CalledProcessError) as exc:
            logger.warning("Error verifying traffic video setup: %s", exc)

        try:
            if setup_helper.create_test_videos():
                existing = _existing_pair()
                if existing is not None:
                    return existing
        except (OSError, RuntimeError, ValueError, CalledProcessError) as exc:
            logger.warning("Error creating fallback traffic videos: %s", exc)

    return str(fallback_first), str(fallback_second)


def main() -> None:
    """Entry-point for both the real and simulation modes."""

    parser = argparse.ArgumentParser(description="Smart traffic management system")
    parser.add_argument(
        "--mode",
        choices=["real", "simulation"],
        default="real",
        help="Select between real camera processing and the synthetic simulation",
    )
    parser.add_argument("--video-road1", type=str, help="Video source for road 1 (real mode)")
    parser.add_argument("--video-road2", type=str, help="Video source for road 2 (real mode)")
    parser.add_argument(
        "--orientation-road1",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Orientation of the first video feed",
    )
    parser.add_argument(
        "--orientation-road2",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Orientation of the second video feed",
    )
    parser.add_argument("--fps", type=int, default=30, help="Simulation frame rate")
    parser.add_argument("--spawn-rate", type=float, default=3.5, help="Average vehicles spawned per second")
    parser.add_argument("--max-frames", type=int, help="Limit frames processed in simulation mode")
    parser.add_argument("--seed", type=int, help="Random seed for simulation reproducibility")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run simulation without rendering a GUI window",
    )

    args = parser.parse_args()

    if args.mode == "simulation":
        try:
            simulation = SimulationTrafficSystem(
                fps=args.fps,
                seed=args.seed,
                spawn_rate=args.spawn_rate,
            )
            simulation.run(
                max_frames=args.max_frames,
                display_window=not args.no_display,
            )
        except ImportError as exc:
            logger.error("Missing dependency for simulation: %s", exc)
            raise
        return

    video_road1 = args.video_road1
    video_road2 = args.video_road2
    if not video_road1 or not video_road2:
        video_road1, video_road2 = resolve_video_sources()

    try:
        system = SmartTrafficSystem(
            video_road1,
            video_road2,
            orientation_road1=args.orientation_road1,
            orientation_road2=args.orientation_road2,
        )
        system.run()

    except FileNotFoundError as e:
        logger.error("Video file not found: %s", e)
        logger.error(
            "Please ensure both road1.mp4 and road2.mp4 are available either in the 'videos/' directory or project root."
        )
        raise
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        logger.error("Install the Ultralytics package with: pip install ultralytics")
        logger.error("Ensure YOLOv8 weights (e.g., yolov8n.pt) are available locally.")
        raise
    except Exception as e:
        logger.exception("Unhandled error in smart traffic system: %s", e)
        raise


if __name__ == "__main__":
    main()
