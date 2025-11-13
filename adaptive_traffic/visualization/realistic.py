"""OpenCV based visualization for realistic mode."""

from __future__ import annotations

import logging
from typing import Dict

from .base import VisualizationStrategy

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional runtime dependency
    import cv2
except Exception as exc:  # pragma: no cover - degrade gracefully
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _CV2_IMPORT_ERROR = None


class RealisticVisualization(VisualizationStrategy):
    """Visualises annotated camera feeds side by side using OpenCV windows."""

    def __init__(self) -> None:
        if cv2 is None:  # pragma: no cover - executed when dependency missing
            raise RuntimeError(
                "OpenCV is required for RealisticVisualization but could not be imported"
            ) from _CV2_IMPORT_ERROR

    def render(self, context: Dict[str, object]) -> None:
        frame_a = context.get("frame_a")
        frame_b = context.get("frame_b")
        lights = context.get("lights", ("red", "red"))

        if frame_a is None or frame_b is None:
            logger.debug("Skipping render because one of the frames is missing")
            return

        combined = cv2.hconcat([frame_a, frame_b])
        cv2.putText(
            combined,
            f"Road A: {lights[0].upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            f"Road B: {lights[1].upper()}",
            (frame_a.shape[1] + 30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Adaptive Traffic - Realistic", combined)
        cv2.waitKey(1)

    def close(self) -> None:
        if cv2 is not None:
            cv2.destroyWindow("Adaptive Traffic - Realistic")
