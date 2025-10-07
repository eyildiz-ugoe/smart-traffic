"""Real-time traffic light automation that relies on YOLOv8 vehicle detection."""

from __future__ import annotations

import logging
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

    model_path: str | Path = "yolov8x.pt"
    confidence: float = 0.25
    iou: float = 0.5
    classes: Iterable[int] | None = None
    device: str | None = None
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

    def __init__(self, orientation: str = "vertical") -> None:
        self.sorter = VehicleSorter(orientation=orientation)
        self.counter = VehicleCounter()

    @property
    def orientation(self) -> str:
        return self.sorter.orientation

    def calculate_metrics(
        self, frame_shape: Tuple[int, int, int], detections: Sequence[VehicleDetection]
    ) -> QueueMetrics:
        sorted_detections = self.sorter.sort(detections)
        pressure = self._calculate_pressure(frame_shape, sorted_detections)
        count, class_breakdown = self.counter.summarize(sorted_detections)
        return QueueMetrics(
            count=count,
            sorted_detections=list(sorted_detections),
            pressure=pressure,
            class_breakdown=class_breakdown,
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
        self.queue_analyzer_road1 = VehicleQueueAnalyzer(orientation=orientation_road1)
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
        
    def draw_traffic_light(self, frame: np.ndarray, signal: str, position: str) -> np.ndarray:
        """
        Draw traffic light indicator on the frame.
        
        Args:
            frame: Input frame
            signal: Signal state ('GREEN', 'YELLOW', 'RED')
            position: Position of indicator ('top-left' or 'top-right')
            
        Returns:
            Frame with traffic light drawn
        """
        h, w = frame.shape[:2]
        
        # Determine position
        if position == 'top-left':
            x_offset, y_offset = 20, 20
        else:
            x_offset, y_offset = w - 120, 20
        
        # Draw traffic light background
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + 80, y_offset + 200), (50, 50, 50), -1)
        
        # Define light colors
        colors = {
            'RED': (0, 0, 255),
            'YELLOW': (0, 255, 255),
            'GREEN': (0, 255, 0)
        }
        
        # Draw lights
        light_positions = {'RED': 50, 'YELLOW': 110, 'GREEN': 170}

        for light, y_pos in light_positions.items():
            color = colors[light] if light == signal else (80, 80, 80)
            cv2.circle(frame, (x_offset + 40, y_offset + y_pos), 25, color, -1)
            cv2.circle(frame, (x_offset + 40, y_offset + y_pos), 25, (255, 255, 255), 2)

        return frame

    def draw_vehicle_annotations(self, frame: np.ndarray, metrics: QueueMetrics) -> np.ndarray:
        """Annotate the frame with bounding boxes ordered by queue priority."""

        for index, detection in enumerate(metrics.sorted_detections, start=1):
            x, y, w, h = detection.bbox
            color = (0, 255, 0) if index == 1 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"#{index} {detection.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame

    def draw_queue_summary(
        self, frame: np.ndarray, metrics: QueueMetrics, signal: str, anchor: Tuple[int, int]
    ) -> np.ndarray:
        """Overlay queue statistics such as vehicle count and pressure."""

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

    def _process_road(
        self, frame: np.ndarray, analyzer: VehicleQueueAnalyzer
    ) -> Tuple[QueueMetrics, np.ndarray]:
        detections = self.detector.detect_vehicles(frame)
        metrics = analyzer.calculate_metrics(frame.shape, detections)
        annotated_frame = self.draw_vehicle_annotations(frame, metrics)
        return metrics, annotated_frame

    def run(self):
        """
        Main loop to run the smart traffic system.
        """
        print("=" * 60)
        print("Smart Traffic Light Automation System")
        print("=" * 60)
        print("Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        
        try:
            while True:
                # Read frames from both videos
                ret1, frame1 = self.cap_road1.read()
                ret2, frame2 = self.cap_road2.read()
                
                # Check if we've reached the end of either video
                if not ret1 or not ret2:
                    print("\nEnd of video reached. Restarting...")
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
                )

                # Draw traffic lights and queue summaries
                frame1 = self.draw_traffic_light(frame1, signal_status['road1'], 'top-right')
                frame2 = self.draw_traffic_light(frame2, signal_status['road2'], 'top-right')

                frame1 = self.draw_queue_summary(
                    frame1,
                    metrics1,
                    signal_status['road1'],
                    (20, frame1.shape[0] - 60),
                )
                frame2 = self.draw_queue_summary(
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
                    print(f"\nFrame: {frame_count}")
                    print(
                        "Road 1: "
                        f"{metrics1.count} vehicles | "
                        f"pressure={metrics1.pressure:.2f} | "
                        f"{self.stats_road1.congestion_level} | "
                        f"signal={signal_status['road1']}"
                    )
                    print(
                        "Road 2: "
                        f"{metrics2.count} vehicles | "
                        f"pressure={metrics2.pressure:.2f} | "
                        f"{self.stats_road2.congestion_level} | "
                        f"signal={signal_status['road2']}"
                    )
                
                frame_count += 1
                
                # Check for quit command
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"\nError occurred: {e}")
            raise
        finally:
            # Cleanup
            self.cap_road1.release()
            self.cap_road2.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "=" * 60)
            print("FINAL STATISTICS")
            print("=" * 60)
            print(f"Road 1: Avg vehicles = {self.stats_road1.avg_vehicles_per_frame:.2f}")
            print(f"Road 2: Avg vehicles = {self.stats_road2.avg_vehicles_per_frame:.2f}")
            if self.last_metrics_road1 is not None:
                print(
                    f"Road 1 last pressure = {self.last_metrics_road1.pressure:.2f} | "
                    f"vehicles = {self.last_metrics_road1.count}"
                )
            if self.last_metrics_road2 is not None:
                print(
                    f"Road 2 last pressure = {self.last_metrics_road2.pressure:.2f} | "
                    f"vehicles = {self.last_metrics_road2.count}"
                )
            print("=" * 60)


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


def main():
    """
    Main function to run the smart traffic system.

    Usage:
        Replace 'road1.mp4' and 'road2.mp4' with your actual video files.
    """
    # Video file paths (update these with your actual video paths)
    VIDEO_ROAD1, VIDEO_ROAD2 = resolve_video_sources()

    try:
        # Initialize and run the system
        system = SmartTrafficSystem(VIDEO_ROAD1, VIDEO_ROAD2)
        system.run()
        
    except FileNotFoundError as e:
        print(f"Error: Video file not found - {e}")
        print("\nPlease ensure you have two video files:")
        print("  - videos/road1.mp4 or road1.mp4: Video of traffic on road 1")
        print("  - videos/road2.mp4 or road2.mp4: Video of traffic on road 2")
        print(
            "The application automatically looks inside the 'videos/' directory "
            "before falling back to the project root."
        )
    except ImportError as e:
        print("Error: Missing dependency -", e)
        print("\nInstall the Ultralytics package with:\n  pip install ultralytics")
        print("Ensure the YOLOv8 weights (e.g., yolov8n.pt) are available locally.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
