"""Realistic vehicle counter using YOLOv8 detections over video feeds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import logging

from .base import CounterResult, VehicleCounter

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional runtime dependency
    import cv2
except Exception as exc:  # pragma: no cover - gracefully degrade when OpenCV missing
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _CV2_IMPORT_ERROR = None

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception as exc:  # pragma: no cover - gracefully degrade when numpy missing
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _NUMPY_IMPORT_ERROR = None

try:  # pragma: no cover - optional runtime dependency
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - gracefully degrade when YOLO missing
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _YOLO_IMPORT_ERROR = None


@dataclass(slots=True)
class CameraCounterConfig:
    """Specialised configuration for :class:`CameraCounter`."""

    video_path: Path
    detection_zone_size: float
    confidence: float = 0.25
    model_path: str | Path = "weights/yolov8n.pt"
    classes: Optional[Iterable[int]] = None


class CameraCounter(VehicleCounter):
    """Vehicle counter backed by YOLOv8 detections on video frames."""

    def __init__(self, config: CameraCounterConfig) -> None:
        if cv2 is None:  # pragma: no cover - executed when dependency missing
            raise RuntimeError(
                "OpenCV is required for CameraCounter but could not be imported"
            ) from _CV2_IMPORT_ERROR
        if np is None:  # pragma: no cover - executed when dependency missing
            raise RuntimeError(
                "NumPy is required for CameraCounter but could not be imported"
            ) from _NUMPY_IMPORT_ERROR
        if YOLO is None:  # pragma: no cover - executed when dependency missing
            raise RuntimeError(
                "ultralytics is required for CameraCounter but could not be imported"
            ) from _YOLO_IMPORT_ERROR

        self.config = config
        self.capture = cv2.VideoCapture(str(config.video_path))
        if not self.capture.isOpened():  # pragma: no cover - depends on runtime files
            raise FileNotFoundError(f"Unable to open video source: {config.video_path}")

        logger.info("Loading YOLO model from %s", config.model_path)
        self.model = YOLO(str(config.model_path))
        self._latest_count = 0
        self._latest_frame: Optional[np.ndarray] = None

    def _detection_zone(self, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        """Return the detection zone rectangle in pixel coordinates.

        The detection zone spans the full width of the frame and the bottom
        ``detection_zone_size`` proportion of the height.  Vehicles counted in
        this area are immediately upstream of the traffic signal.
        """

        height, width = frame_shape[:2]
        zone_height = max(1, int(height * self.config.detection_zone_size))
        top = height - zone_height
        return (0, top, width, zone_height)

    def _count_detections(self, frame: "np.ndarray") -> tuple[int, "np.ndarray"]:
        """Run YOLO on ``frame`` and return vehicle count with annotations."""

        zone = self._detection_zone(frame.shape)
        zx, zy, zw, zh = zone

        overlays: "np.ndarray" = frame.copy()
        cv2.rectangle(overlays, (zx, zy), (zx + zw, zy + zh), (0, 128, 255), 2)
        cv2.addWeighted(frame, 0.7, overlays, 0.3, 0, overlays)

        logger.debug("Running YOLO detection on frame")
        results = self.model.predict(
            overlays,
            conf=self.config.confidence,
            classes=self.config.classes,
            verbose=False,
        )

        detections_in_zone = 0
        annotated = overlays.copy()
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:  # pragma: no cover - depends on runtime detection
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                cls = int(box.cls)

                x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
                cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{conf:.2f}",
                    (x1_i, max(0, y1_i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                if y2_i >= zy and y1_i <= zy + zh:
                    detections_in_zone += 1

        cv2.putText(
            annotated,
            f"Vehicles: {detections_in_zone}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        return detections_in_zone, annotated

    def step(self) -> CounterResult:
        ok, frame = self.capture.read()
        if not ok:  # pragma: no cover - depends on runtime videos
            logger.warning("End of stream reached for %s", self.config.video_path)
            self._latest_frame = None
            self._latest_count = 0
            return CounterResult(count=0, frame=None)

        count, annotated = self._count_detections(frame)
        self._latest_count = count
        self._latest_frame = annotated
        return CounterResult(count=count, frame=annotated)

    def get_count(self) -> int:
        return self._latest_count

    def get_frame(self) -> Optional["np.ndarray"]:
        """Return the most recently annotated frame."""

        return self._latest_frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
