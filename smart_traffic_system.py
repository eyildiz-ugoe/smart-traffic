"""Real-time traffic light automation that relies on YOLOv8 vehicle detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from traffic_core import TrafficLightController, TrafficStats

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

    model_path: str | Path = "yolov8n.pt"
    confidence: float = 0.25
    iou: float = 0.5
    classes: Iterable[int] | None = None
    device: str | None = None


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

    def detect_vehicles(self, frame: np.ndarray) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        """Detect vehicles on a frame using YOLOv8."""

        results = self.model(
            frame,
            verbose=False,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.config.device,
        )[0]

        vehicles: List[Tuple[int, int, int, int]] = []

        if not hasattr(results, "boxes") or results.boxes is None:  # pragma: no cover - defensive
            return 0, vehicles

        for cls, conf, xyxy in zip(results.boxes.cls, results.boxes.conf, results.boxes.xyxy):
            if int(cls) not in self._target_classes:
                continue
            if float(conf) < self.config.confidence:
                continue

            x1, y1, x2, y2 = map(int, xyxy.tolist())
            vehicles.append((x1, y1, x2 - x1, y2 - y1))

        return len(vehicles), vehicles


class SmartTrafficSystem:
    """Main system that integrates YOLO detection with traffic light control."""

    def __init__(
        self,
        video_road1: str,
        video_road2: str,
        detector_config: DetectorConfig | None = None,
    ) -> None:
        """
        Initialize the smart traffic system.

        Args:
            video_road1: Path to video file for road 1
            video_road2: Path to video file for road 2
            detector_config: Optional configuration for the YOLO detector
        """
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
        
        # Initialize traffic light controller
        self.controller = TrafficLightController()
        
        # Initialize statistics
        self.stats_road1 = TrafficStats()
        self.stats_road2 = TrafficStats()
        
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
                    continue
                
                # Resize frames for better visualization
                frame1 = cv2.resize(frame1, (640, 480))
                frame2 = cv2.resize(frame2, (640, 480))
                
                # Detect vehicles on both roads
                count1, _ = self.detector.detect_vehicles(frame1)
                count2, _ = self.detector.detect_vehicles(frame2)
                
                # Update statistics
                self.stats_road1.update(count1)
                self.stats_road2.update(count2)
                
                # Update traffic signal timing
                signal_status = self.controller.update_signal_timing(count1, count2)
                
                # Draw traffic lights
                frame1 = self.draw_traffic_light(frame1, signal_status['road1'], 'top-right')
                frame2 = self.draw_traffic_light(frame2, signal_status['road2'], 'top-right')

                # Combine frames side by side
                combined_frame = np.hstack([frame1, frame2])
                
                # Display the result
                cv2.imshow('Smart Traffic Light System', combined_frame)
                
                # Print statistics every 30 frames
                if frame_count % 30 == 0:
                    print(f"\nFrame: {frame_count}")
                    print(f"Road 1: {count1} vehicles | {self.stats_road1.congestion_level} | {signal_status['road1']}")
                    print(f"Road 2: {count2} vehicles | {self.stats_road2.congestion_level} | {signal_status['road2']}")
                
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
            print("=" * 60)


def main():
    """
    Main function to run the smart traffic system.
    
    Usage:
        Replace 'road1.mp4' and 'road2.mp4' with your actual video files.
    """
    # Video file paths (update these with your actual video paths)
    VIDEO_ROAD1 = "videos/road1.mp4"
    VIDEO_ROAD2 = "videos/road2.mp4"
    
    try:
        # Initialize and run the system
        system = SmartTrafficSystem(VIDEO_ROAD1, VIDEO_ROAD2)
        system.run()
        
    except FileNotFoundError as e:
        print(f"Error: Video file not found - {e}")
        print("\nPlease ensure you have two video files:")
        print("  - road1.mp4: Video of traffic on road 1")
        print("  - road2.mp4: Video of traffic on road 2")
    except ImportError as e:
        print("Error: Missing dependency -", e)
        print("\nInstall the Ultralytics package with:\n  pip install ultralytics")
        print("Ensure the YOLOv8 weights (e.g., yolov8n.pt) are available locally.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
