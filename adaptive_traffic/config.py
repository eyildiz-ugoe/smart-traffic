"""Configuration dataclasses for the adaptive traffic light system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ModeLiteral = Literal["realistic", "simulation"]


@dataclass(slots=True)
class TrafficConfig:
    """Runtime configuration for :class:`adaptive_traffic.system.TrafficSystem`.

    Parameters
    ----------
    mode:
        Operating mode. ``"realistic"`` consumes camera/video input while
        ``"simulation"`` renders an animated intersection.
    video_path_road_a, video_path_road_b:
        Absolute or relative paths to the video feeds used in realistic mode.
    spawn_rate_road_a, spawn_rate_road_b:
        Expected spawn rate for simulated vehicles (vehicles per minute).
    min_green_time, max_red_time, yellow_time:
        Timing constraints for the adaptive controller in seconds.
    detection_zone_size:
        Fractional height (0-1] of the frame that forms the upstream detection
        zone directly before the stop line. ``0.2`` monitors the bottom 20%% of
        the frame.
    detection_confidence:
        Minimum YOLO confidence threshold used when filtering detections.
    """

    mode: ModeLiteral = "simulation"
    video_path_road_a: str | Path | None = None
    video_path_road_b: str | Path | None = None
    spawn_rate_road_a: float = 12.0
    spawn_rate_road_b: float = 12.0
    min_green_time: float = 5.0
    max_red_time: float = 60.0
    yellow_time: float = 3.0
    detection_zone_size: float = 0.25
    detection_confidence: float = 0.25

    def ensure_paths(self) -> None:
        """Expand configured video paths to absolute :class:`~pathlib.Path`.

        Realistic mode works with OpenCV capture objects which require that the
        target file exists.  This helper normalises the configuration so callers
        can provide ``str`` or :class:`~pathlib.Path` interchangeably.
        """

        if self.video_path_road_a is not None:
            self.video_path_road_a = Path(self.video_path_road_a).expanduser().resolve()
        if self.video_path_road_b is not None:
            self.video_path_road_b = Path(self.video_path_road_b).expanduser().resolve()
