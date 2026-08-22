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


def resolve_detection_device(requested: str | None) -> str | None:
    """Return a usable inference device, falling back to CPU when CUDA is absent."""

    if requested is None or not str(requested).startswith("cuda"):
        return requested

    try:
        import torch
    except ImportError:
        logger.info("PyTorch unavailable; using CPU for detection")
        return "cpu"

    if not torch.cuda.is_available():
        logger.info("CUDA not available; falling back to CPU for detection")
        return "cpu"

    return requested


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
    #: Persistent tracker ID (from VehicleDetector.track_vehicles); None when
    #: plain per-frame detection was used or the tracker had no match.
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def bottom_edge(self) -> int:
        x, y, w, h = self.bbox
        return y + h

    @property
    def right_edge(self) -> int:
        x, y, w, h = self.bbox
        return x + w

    @property
    def top_edge(self) -> int:
        x, y, w, h = self.bbox
        return y

    @property
    def left_edge(self) -> int:
        x, y, w, h = self.bbox
        return x


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
        self._device = resolve_detection_device(self.config.device)

        model_path = Path(self.config.model_path)
        # YOLO accepts either a local path or a model name; don't resolve unless it exists locally
        self.model = YOLO(str(model_path))
        self.model.fuse()  # type: ignore[no-untyped-call]

    #: Tracker configuration tuned for distant traffic-camera detections
    #: (resolved relative to this file so the working directory is irrelevant).
    TRACKER_CONFIG = str(Path(__file__).resolve().parent / "trackers" / "bytetrack_traffic.yaml")

    #: NMS confidence floor for the *tracker* call. This must sit at the
    #: tracker config's ``track_low_thresh`` — passing the display threshold
    #: here would starve ByteTrack's low-score association pass (its
    #: occlusion-recovery mechanism) and fragment track IDs whenever a
    #: vehicle's confidence dips. Detections below ``config.confidence``
    #: are still filtered out of the returned list afterwards.
    TRACKER_NMS_CONF = 0.1

    def _results_to_detections(self, results) -> List[VehicleDetection]:
        detections: List[VehicleDetection] = []

        if not hasattr(results, "boxes") or results.boxes is None:  # pragma: no cover - defensive
            return detections

        ids = getattr(results.boxes, "id", None)
        id_list = ids.int().tolist() if ids is not None else [None] * len(results.boxes)

        for cls, conf, xyxy, track_id in zip(
            results.boxes.cls, results.boxes.conf, results.boxes.xyxy, id_list
        ):
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
                    track_id=int(track_id) if track_id is not None else None,
                )
            )

        detections.sort(key=lambda det: det.confidence, reverse=True)

        if self.config.max_detections is not None:
            detections = detections[: self.config.max_detections]

        return detections

    def detect_vehicles(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Detect vehicles on a frame using YOLOv8."""

        results = self.model(
            frame,
            verbose=False,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self._device,
        )[0]
        return self._results_to_detections(results)

    def reset_tracker(self) -> None:
        """Clear ByteTrack state (e.g. when a looping video rewinds).

        Best-effort: the tracker objects only exist after the first
        ``track_vehicles`` call, and the internal API may vary between
        ultralytics releases.

        Caution: ultralytics' ``reset()`` zeroes a *process-global* track-ID
        counter (``BaseTrack._count``), so if multiple VehicleDetector
        instances ever track concurrently in one process, resetting one
        recycles IDs for all of them. Each real-mode demo currently owns
        exactly one detector, and MotionFilter state is always cleared via
        ``handle_discontinuity`` alongside this call, so recycled IDs can
        never collide with stale dwell state.
        """

        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        for tracker in trackers:
            try:
                tracker.reset()
            except Exception:  # pragma: no cover - version-dependent internals
                logger.debug("Tracker reset unsupported; continuing without it")

    def track_vehicles(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Detect vehicles with persistent tracker IDs (ByteTrack).

        IDs are stable across frames for the lifetime of this detector
        instance, enabling motion-history logic such as the parked-car
        dwell-time filter.
        """

        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=min(self.TRACKER_NMS_CONF, self.config.confidence),
            iou=self.config.iou,
            device=self._device,
            tracker=self.TRACKER_CONFIG,
        )[0]
        return self._results_to_detections(results)


class VehicleQueueAnalyzer:
    """Derive queue metrics from raw vehicle detections."""

    def __init__(
        self,
        orientation: str = "vertical",
        *,
        approach_threshold_ratio: float = 0.65,
        exit_margin: int = 5,
        line_contact_margin: int = 2,
    ) -> None:
        if not 0.0 < approach_threshold_ratio < 1.0:
            raise ValueError("approach_threshold_ratio must be between 0 and 1")
        if exit_margin < 0:
            raise ValueError("exit_margin must be non-negative")

        self.sorter = VehicleSorter(orientation=orientation)
        self.counter = VehicleCounter()
        self._approach_threshold_ratio = approach_threshold_ratio
        self._exit_margin = exit_margin
        self._line_contact_margin = max(0, line_contact_margin)

    @property
    def orientation(self) -> str:
        return self.sorter.orientation

    def calculate_metrics(
        self, frame_shape: Tuple[int, int, int], detections: Sequence[VehicleDetection]
    ) -> QueueMetrics:
        sorted_detections = self.sorter.sort(detections)

        # Threshold flags consider every detection (a vehicle past the line
        # still matters for exit-zone occupancy).
        approach_line, exit_line, stopline_occupied, exit_zone_active, leading_edge = (
            self._calculate_thresholds(frame_shape, sorted_detections)
        )

        # Filter detections to only include vehicles that are approaching or at the stop line.
        # Vehicles that have fully passed the approach line are considered "clearing" and
        # should not contribute to the demand count for keeping the light green.
        relevant_detections = []
        for det in sorted_detections:
            is_passed = False
            if self.orientation == "vertical":
                # Moving Top to Bottom. Position increases.
                # If top_edge > approach_line, the vehicle has fully crossed the line.
                if det.top_edge > approach_line:
                    is_passed = True
            else:
                # Moving Left to Right. Position increases.
                # If left_edge > approach_line, the vehicle has fully crossed the line.
                if det.left_edge > approach_line:
                    is_passed = True

            if not is_passed:
                relevant_detections.append(det)

        # count, pressure, and sorted_detections all describe the same set:
        # the queued vehicles, in priority order. Passed/clearing vehicles
        # are excluded so len(sorted_detections) == count always holds.
        pressure = self._calculate_pressure(frame_shape, relevant_detections)
        count, class_breakdown = self.counter.summarize(relevant_detections)
        return QueueMetrics(
            count=count,
            sorted_detections=list(relevant_detections),
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

        stopline_occupied = any(
            self._line_intersection(det, approach_line) for det in detections
        )
        exit_zone_active = any(
            self._line_intersection(det, exit_line) for det in detections
        )

        return approach_line, exit_line, stopline_occupied, exit_zone_active, leading_edge

    def _line_intersection(self, detection: VehicleDetection, line_position: int) -> bool:
        margin = self._line_contact_margin
        if self.orientation == "vertical":
            top = detection.top_edge - margin
            bottom = detection.bottom_edge + margin
            return top <= line_position <= bottom
        else:
            left = detection.left_edge - margin
            right = detection.right_edge + margin
            return left <= line_position <= right


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

    def run(
        self,
        max_frames: Optional[int] = None,
        *,
        display_window: bool = True,
        window_name: str = "Smart Traffic Light System",
        fullscreen: bool = False,
    ):
        """
        Main loop to run the smart traffic system.

        Args:
            max_frames: Optional frame budget; ``None`` runs until 'q'.
            display_window: Render a GUI window; disable for headless runs.
            window_name: Title of the display window.
            fullscreen: Start fullscreen (press 'f' to toggle).
        """
        logger.info("Smart Traffic Light Automation System initialised. Press 'q' to quit.")

        from demo_ui import draw_controls_hint, handle_display_keys, show_end_card

        frame_count = 0
        fullscreen_active = fullscreen and display_window
        action = None
        switches = 0
        previous_active = None

        if display_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            target_state = cv2.WINDOW_FULLSCREEN if fullscreen_active else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, target_state)

        read_failures = 0
        try:
            while max_frames is None or frame_count < max_frames:
                # Read frames from both videos
                ret1, frame1 = self.cap_road1.read()
                ret2, frame2 = self.cap_road2.read()

                # Check if we've reached the end of either video
                if not ret1 or not ret2:
                    read_failures += 1
                    if read_failures > 1:
                        raise RuntimeError(
                            "Unable to read frames from the video sources even after "
                            "rewinding — a file may be corrupt or truncated."
                        )
                    logger.info("End of video reached. Restarting playback from the beginning.")
                    self.cap_road1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap_road2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.queue_analyzer_road1.counter.reset()
                    self.queue_analyzer_road2.counter.reset()
                    continue
                read_failures = 0
                
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

                if previous_active is None:
                    previous_active = signal_status["active_road"]
                elif signal_status["active_road"] != previous_active:
                    switches += 1
                    previous_active = signal_status["active_road"]

                # Combine frames side by side
                combined_frame = np.hstack([frame1, frame2])

                # Display the result
                if display_window:
                    draw_controls_hint(combined_frame)
                    cv2.imshow(window_name, combined_frame)

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

                # Check for quit / pause / fullscreen commands
                if display_window:
                    action, fullscreen_active = handle_display_keys(
                        window_name, 30, fullscreen_active
                    )
                    if action:
                        break

        except Exception:
            logger.exception("Error occurred during smart traffic system execution.")
            raise
        finally:
            # Cleanup
            self.cap_road1.release()
            self.cap_road2.release()
            if display_window and action != "exit":
                card_action = show_end_card(
                    window_name,
                    "Case 2 - Two-Road Intersection (Real, shadow mode)",
                    [
                        f"Frames analysed: {frame_count}",
                        f"Signal switches: {switches}",
                        f"Avg vehicles/frame  Road 1: "
                        f"{self.stats_road1.avg_vehicles_per_frame:.2f}   Road 2: "
                        f"{self.stats_road2.avg_vehicles_per_frame:.2f}",
                        "Live YOLOv8 detection driving the adaptive plan.",
                    ],
                )
                action = card_action or action
            if display_window:
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
        return action


@dataclass(slots=True)
class SimulatedVehicle:
    """Lightweight vehicle representation used by the simulation mode."""

    position: float
    speed: float
    length: int
    width: int
    color: Tuple[int, int, int]
    waited: float = 0.0


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
        spawn_pause_chance: float = 0.0,
        spawn_pause_duration: Tuple[float, float] = (2.0, 4.0),
        spawn_pause_cooldown: Tuple[float, float] = (5.0, 9.0),
        enforce_clearance: bool = False,
        post_clear_pause: Tuple[float, float] | None = None,
        palette: Sequence[Tuple[int, int, int]] | None = None,
    ) -> None:
        if orientation not in {"vertical", "horizontal"}:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        self.orientation = orientation
        self.frame_height, self.frame_width = frame_size
        self.rng = rng
        self.spawn_rate = max(0.0, spawn_rate)
        self.max_vehicles = max_vehicles

        self.vehicles: List[SimulatedVehicle] = []
        self.min_gap = 30
        #: Cumulative vehicle-seconds spent waiting (for baseline comparisons).
        self.total_wait = 0.0
        #: Per-road vehicle colour family; overridden for on-screen road identity.
        self.palette: List[Tuple[int, int, int]] = list(
            palette or [(66, 245, 189), (66, 134, 244), (244, 199, 66), (240, 96, 96)]
        )

        self._spawn_pause_chance = max(0.0, spawn_pause_chance)
        self._spawn_pause_duration_range = spawn_pause_duration
        self._spawn_pause_cooldown_range = spawn_pause_cooldown
        self._spawn_pause_remaining = 0.0
        self._spawn_pause_cooldown = (
            self.rng.uniform(*spawn_pause_cooldown) if self._spawn_pause_chance > 0.0 else 0.0
        )
        self._awaiting_clearance = False
        self._enforce_clearance = enforce_clearance
        self._post_clear_pause_range = post_clear_pause or (0.0, 0.0)
        self._post_clear_pause_remaining = 0.0

        if orientation == "vertical":
            self.vehicle_length, self.vehicle_width = 64, 32
            self.stop_line = self.frame_height // 2 - 30
            self._despawn_limit = self.frame_height
            self._lane_left = self.frame_width // 2 - 60
            self._lane_right = self.frame_width // 2 + 60
        else:
            self.vehicle_length, self.vehicle_width = 64, 32
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
        color = self.rng.choice(self.palette)

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

    def _spawn_paused(self, dt: float) -> bool:
        if self._spawn_pause_chance <= 0.0:
            return False

        if self._awaiting_clearance:
            if not self._enforce_clearance or not self.vehicles:
                self._awaiting_clearance = False
                self._spawn_pause_cooldown = self.rng.uniform(*self._spawn_pause_cooldown_range)
                if self._post_clear_pause_range[1] > 0:
                    self._post_clear_pause_remaining = self.rng.uniform(*self._post_clear_pause_range)
                return True
            return True

        if self._post_clear_pause_remaining > 0.0:
            self._post_clear_pause_remaining = max(0.0, self._post_clear_pause_remaining - dt)
            return True

        if self._spawn_pause_remaining > 0.0:
            self._spawn_pause_remaining = max(0.0, self._spawn_pause_remaining - dt)
            if self._spawn_pause_remaining == 0.0:
                self._awaiting_clearance = self._enforce_clearance
            return True

        self._spawn_pause_cooldown = max(0.0, self._spawn_pause_cooldown - dt)
        if self._spawn_pause_cooldown == 0.0:
            if self.rng.random() < self._spawn_pause_chance:
                self._spawn_pause_remaining = self.rng.uniform(*self._spawn_pause_duration_range)
                return True
            self._spawn_pause_cooldown = self.rng.uniform(*self._spawn_pause_cooldown_range)

        return False

    def _maybe_spawn(self, dt: float) -> None:
        if len(self.vehicles) >= self.max_vehicles:
            return

        if self._spawn_paused(dt):
            return

        probability = self.spawn_rate * dt
        if self.rng.random() < probability:
            # Check if spawn area is clear
            new_vehicle = self._new_vehicle()
            
            # Check for overlap with existing vehicles
            is_clear = True
            for v in self.vehicles:
                # Simple 1D overlap check
                # New vehicle is at new_vehicle.position (top/left) to new_vehicle.position + length (bottom/right)
                # Existing vehicle is at v.position to v.position + length
                
                new_start = new_vehicle.position
                new_end = new_vehicle.position + new_vehicle.length
                v_start = v.position
                v_end = v.position + v.length
                
                # Check if intervals overlap
                if max(new_start, v_start) < min(new_end, v_end) + self.min_gap:
                    is_clear = False
                    break
            
            if is_clear:
                self.vehicles.append(new_vehicle)

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
            if target_front - current_front < 1e-6 and current_front <= self.stop_line:
                vehicle.waited += dt
                self.total_wait += dt
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
            if target_front - current_front < 1e-6 and current_front <= self.stop_line:
                vehicle.waited += dt
                self.total_wait += dt
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
            cv2.rectangle(
                frame,
                (x, top),
                (x + self.vehicle_width, top + self.vehicle_length),
                (25, 25, 25),
                1,
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
            cv2.rectangle(
                frame,
                (left, y),
                (left + self.vehicle_length, y + self.vehicle_width),
                (25, 25, 25),
                1,
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


def _outlined_text(frame, text, org, scale=0.55, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_compact_signal(frame, signal: str, top_left: Tuple[int, int], label: str):
    """Mini three-lamp housing (matches the Case 3 look)."""

    x, y = top_left
    w, h = 26, 66
    cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 38, 42), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    lamps = [("RED", (0, 0, 220), y + 13), ("YELLOW", (0, 210, 230), y + 33),
             ("GREEN", (0, 200, 0), y + 53)]
    for name, color, cy in lamps:
        lit = signal == name
        cv2.circle(frame, (x + w // 2, cy), 8, color if lit else (70, 70, 70), -1)
    _outlined_text(frame, label, (x - 8, y + h + 18), scale=0.5)


def draw_case2_hud(
    frame,
    signal_status: Dict[str, object],
    metrics1: "QueueMetrics",
    metrics2: "QueueMetrics",
    switches: int,
    sim_time: float,
    adaptive_wait: Optional[float] = None,
    baseline_wait: Optional[float] = None,
):
    """Case 3-style HUD for the two-road simulation: translucent panel with
    outlined text, plus compact per-road signal housings — no overlapping
    widgets."""

    active = "Road 1" if signal_status["active_road"] == "road1" else "Road 2"
    remaining = signal_status.get("time_remaining")
    lines = [
        (f"{active} has GREEN"
         if signal_status[signal_status["active_road"]] == "GREEN"
         else f"Changing over ({signal_status['road1']} / {signal_status['road2']})",
         (255, 255, 255)),
        (f"Road 1: {metrics1.count} cars   pressure {metrics1.pressure:.1f}", ROAD1_COLOR),
        (f"Road 2: {metrics2.count} cars   pressure {metrics2.pressure:.1f}", ROAD2_COLOR),
        (f"Switches: {switches}   elapsed: {sim_time:.0f} s", (255, 255, 255)),
    ]
    if isinstance(remaining, (int, float)):
        lines.insert(1, (f"Phase ends in: {float(remaining):.1f} s", (255, 255, 255)))
    if baseline_wait is not None and adaptive_wait is not None and baseline_wait > 1.0:
        saved = baseline_wait - adaptive_wait
        pct = 100.0 * saved / baseline_wait
        lines.append((
            f"Waiting vs fixed timer: {adaptive_wait:.0f}s vs {baseline_wait:.0f}s "
            f"({pct:+.0f}% saved)"
            if saved >= 0 else
            f"Waiting vs fixed timer: {adaptive_wait:.0f}s vs {baseline_wait:.0f}s",
            (80, 255, 120),
        ))

    panel_w = 12 + max(
        cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] for t, _ in lines
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w + 12, 20 + 24 * len(lines)),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for index, (text, color) in enumerate(lines):
        _outlined_text(frame, text, (16, 30 + index * 24), color=color)

    width = frame.shape[1]
    height = frame.shape[0]
    draw_compact_signal(frame, str(signal_status["road1"]), (width - 118, 14), "Road 1")
    draw_compact_signal(frame, str(signal_status["road2"]), (width - 52, 14), "Road 2")

    # Identity labels drawn ON the roads, with direction-of-travel arrows.
    _outlined_text(frame, "Road 1", (width // 2 + 68, 26), color=ROAD1_COLOR)
    cv2.arrowedLine(frame, (width // 2 + 52, 12), (width // 2 + 52, 44),
                    ROAD1_COLOR, 2, tipLength=0.35)
    _outlined_text(frame, "Road 2", (10, height // 2 - 72), color=ROAD2_COLOR)
    cv2.arrowedLine(frame, (12, height // 2 - 62), (44, height // 2 - 62),
                    ROAD2_COLOR, 2, tipLength=0.35)
    return frame


#: Road identity colours (BGR families), matched to the Case 3 approach hues.
ROAD1_PALETTE = [(70, 180, 255), (60, 150, 230), (90, 200, 255)]   # oranges
ROAD2_PALETTE = [(255, 180, 70), (230, 150, 60), (255, 200, 100)]  # blues
ROAD1_COLOR = ROAD1_PALETTE[0]
ROAD2_COLOR = ROAD2_PALETTE[0]


class FixedTimeSignal:
    """A dumb fixed-cycle plan: the baseline the adaptive controller beats.

    20 s green per road + 3 s yellow + 2 s all-red, forever, regardless of
    demand — the standard behaviour of an untimed-study intersection.
    """

    GREEN = 20.0
    YELLOW = 3.0
    ALL_RED = 2.0

    def signals(self, now: float) -> Dict[str, str]:
        cycle = self.GREEN + self.YELLOW + self.ALL_RED
        phase = now % (2 * cycle)
        road, offset = ("road1", phase) if phase < cycle else ("road2", phase - cycle)
        other = "road2" if road == "road1" else "road1"
        if offset < self.GREEN:
            state = "GREEN"
        elif offset < self.GREEN + self.YELLOW:
            state = "YELLOW"
        else:
            state = "RED"
        return {road: state, other: "RED"}


class Case2Baseline:
    """Invisible twin world: identical traffic under the fixed-time plan.

    Both worlds consume identical random streams (separate ``Random``
    instances with the same seed, same call order), so every car that
    spawns in the adaptive world spawns here too — the wait-time difference
    is therefore a true like-for-like comparison.
    """

    def __init__(self, seed: int, frame_size, main_rate: float, side_rate: float) -> None:
        rng = random.Random(seed)
        self.road1 = SimulatedRoad(
            "vertical", frame_size, rng, spawn_rate=main_rate, max_vehicles=7,
            spawn_pause_chance=1.0, spawn_pause_duration=(4.0, 6.5),
            spawn_pause_cooldown=(4.5, 8.0), enforce_clearance=True,
            post_clear_pause=(0.8, 1.6), palette=ROAD1_PALETTE,
        )
        self.road2 = SimulatedRoad(
            "horizontal", frame_size, rng, spawn_rate=side_rate, max_vehicles=3,
            spawn_pause_chance=0.9, spawn_pause_duration=(2.5, 4.5),
            spawn_pause_cooldown=(3.0, 6.0), enforce_clearance=True,
            post_clear_pause=(0.9, 1.8), palette=ROAD2_PALETTE,
        )
        self.signal = FixedTimeSignal()
        self.elapsed = 0.0

    def step(self, dt: float) -> None:
        self.elapsed += dt
        signals = self.signal.signals(self.elapsed)
        self.road1.step(signals["road1"], dt)
        self.road2.step(signals["road2"], dt)

    def total_wait(self) -> float:
        return self.road1.total_wait + self.road2.total_wait


class SimulationTrafficSystem:
    """Generate synthetic frames and queue data without using a camera feed."""

    def __init__(
        self,
        fps: int = 30,
        frame_size: Tuple[int, int] = (480, 640),
        *,
        seed: Optional[int] = None,
        spawn_rate: float = 3.5,
        spawn_rate_road1: Optional[float] = None,
        spawn_rate_road2: Optional[float] = None,
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
        resolved_seed = seed if seed is not None else random.randrange(1 << 30)
        rng = random.Random(resolved_seed)

        main_spawn_rate = (
            spawn_rate_road1 if spawn_rate_road1 is not None else max(0.6, spawn_rate * 0.55)
        )
        side_spawn_rate = (
            spawn_rate_road2 if spawn_rate_road2 is not None else max(0.3, spawn_rate * 0.35)
        )

        self.road1 = SimulatedRoad(
            "vertical",
            frame_size,
            rng,
            spawn_rate=main_spawn_rate,
            max_vehicles=7,
            spawn_pause_chance=1.0,
            spawn_pause_duration=(4.0, 6.5),
            spawn_pause_cooldown=(4.5, 8.0),
            enforce_clearance=True,
            post_clear_pause=(0.8, 1.6),
            palette=ROAD1_PALETTE,
        )
        self.road2 = SimulatedRoad(
            "horizontal",
            frame_size,
            rng,
            spawn_rate=side_spawn_rate,
            max_vehicles=3,
            spawn_pause_chance=0.9,
            spawn_pause_duration=(2.5, 4.5),
            spawn_pause_cooldown=(3.0, 6.0),
            enforce_clearance=True,
            post_clear_pause=(0.9, 1.8),
            palette=ROAD2_PALETTE,
        )
        # Invisible twin running the SAME traffic under a fixed-time plan,
        # so the HUD can display the measured benefit live.
        self.baseline = Case2Baseline(
            resolved_seed, frame_size, main_spawn_rate, side_spawn_rate
        )
        self._scene_background = self._create_scene_background()

        def _line_ratio(position: int, dimension: int) -> float:
            ratio = position / float(max(1, dimension))
            return max(0.05, min(0.95, ratio))

        buffer_road1 = max(15, self.road1.vehicle_length // 2)
        buffer_road2 = max(15, self.road2.vehicle_length // 2)

        self.queue_analyzer_road1 = VehicleQueueAnalyzer(
            orientation="vertical",
            approach_threshold_ratio=_line_ratio(self.road1.stop_line, frame_size[0]),
            exit_margin=max(5, buffer_road1 // 2),
            line_contact_margin=max(1, buffer_road1 // 6),
        )
        self.queue_analyzer_road2 = VehicleQueueAnalyzer(
            orientation="horizontal",
            approach_threshold_ratio=_line_ratio(self.road2.stop_line, frame_size[1]),
            exit_margin=max(5, buffer_road2 // 2),
            line_contact_margin=max(1, buffer_road2 // 6),
        )

        # Drive the controller from simulated time so signal timing follows the
        # frame clock (dt per frame) instead of wall-clock time. This keeps
        # behaviour identical between display and headless runs.
        self._sim_time = 0.0
        self.controller = TrafficLightController(time_func=lambda: self._sim_time)
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
        fullscreen: bool = False,
    ) -> None:
        logger.info("Simulation mode initialised. Press 'q' to quit.")

        from demo_ui import draw_controls_hint, handle_display_keys, show_end_card

        dt = 1.0 / float(self.fps)
        frame_count = 0
        fullscreen_active = fullscreen if display_window else False
        action = None
        switches = 0
        previous_active = self._current_signal["active_road"]

        if display_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            target_state = cv2.WINDOW_FULLSCREEN if fullscreen_active else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, target_state)

        try:
            while max_frames is None or frame_count < max_frames:
                self._sim_time += dt
                self.road1.step(self._current_signal["road1"], dt)
                self.road2.step(self._current_signal["road2"], dt)
                self.baseline.step(dt)

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

                self._current_signal = self.controller.update_signal_timing(
                    metrics1.count,
                    metrics2.count,
                    road1_queue_pressure=metrics1.pressure,
                    road2_queue_pressure=metrics2.pressure,
                    road1_stopline_occupied=metrics1.stopline_occupied,
                    road2_stopline_occupied=metrics2.stopline_occupied,
                    road1_exit_ready=metrics1.count == 0,
                    road2_exit_ready=metrics2.count == 0,
                    road1_leading_edge=metrics1.leading_edge,
                    road2_leading_edge=metrics2.leading_edge,
                    road1_approach_line=metrics1.approach_line,
                    road2_approach_line=metrics2.approach_line,
                )

                if self._current_signal["active_road"] != previous_active:
                    switches += 1
                    previous_active = self._current_signal["active_road"]

                frame = self._scene_background.copy()
                self.road1.draw_vehicles(frame)
                self.road2.draw_vehicles(frame)

                frame = draw_vehicle_annotations(frame, metrics1)
                frame = draw_vehicle_annotations(frame, metrics2)
                frame = draw_threshold_lines(frame, metrics1, self.queue_analyzer_road1)
                frame = draw_threshold_lines(frame, metrics2, self.queue_analyzer_road2)

                frame = draw_case2_hud(
                    frame, self._current_signal, metrics1, metrics2,
                    switches, self._sim_time,
                    adaptive_wait=self.road1.total_wait + self.road2.total_wait,
                    baseline_wait=self.baseline.total_wait(),
                )

                if display_window:
                    draw_controls_hint(frame)
                    cv2.imshow(window_name, frame)
                    action, fullscreen_active = handle_display_keys(
                        window_name, int(1000 / self.fps), fullscreen_active
                    )
                    if action:
                        break

                frame_count += 1

        finally:
            if display_window and action != "exit":
                adaptive_wait = self.road1.total_wait + self.road2.total_wait
                baseline_wait = self.baseline.total_wait()
                saved_line = "Same traffic, fixed 20 s timer: no comparison yet"
                if baseline_wait > 1.0:
                    pct = 100.0 * (baseline_wait - adaptive_wait) / baseline_wait
                    saved_line = (
                        f"Waiting time vs fixed 20 s timer: {adaptive_wait:.0f} s "
                        f"vs {baseline_wait:.0f} s  ({pct:+.0f}%)"
                    )
                card_action = show_end_card(
                    window_name,
                    "Case 2 - Two-Road Intersection",
                    [
                        f"Simulated time: {self._sim_time:.0f} s",
                        f"Signal switches: {switches}",
                        saved_line,
                        "Identical cars ran in an invisible twin world under",
                        "a dumb fixed timer - that is the saving.",
                    ],
                )
                action = card_action or action
            if display_window:
                cv2.destroyAllWindows()

            logger.info(
                "Simulation complete. Avg vehicles per frame -> Road1: %.2f, Road2: %.2f",
                self.stats_road1.avg_vehicles_per_frame,
                self.stats_road2.avg_vehicles_per_frame,
            )
        return action

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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

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
    parser.add_argument(
        "--spawn-rate-road1",
        type=float,
        help="Override spawn rate for road 1 in simulation mode",
    )
    parser.add_argument(
        "--spawn-rate-road2",
        type=float,
        help="Override spawn rate for road 2 in simulation mode",
    )
    parser.add_argument("--max-frames", type=int, help="Limit frames processed in simulation mode")
    parser.add_argument("--seed", type=int, help="Random seed for simulation reproducibility")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run simulation without rendering a GUI window",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start the simulation window in fullscreen mode",
    )

    args = parser.parse_args()

    if args.mode == "simulation":
        try:
            simulation = SimulationTrafficSystem(
                fps=args.fps,
                seed=args.seed,
                spawn_rate=args.spawn_rate,
                spawn_rate_road1=args.spawn_rate_road1,
                spawn_rate_road2=args.spawn_rate_road2,
            )
            if args.fullscreen and args.no_display:
                logger.warning("Ignoring --fullscreen because --no-display was set.")
            simulation.run(
                max_frames=args.max_frames,
                display_window=not args.no_display,
                fullscreen=args.fullscreen and not args.no_display,
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
