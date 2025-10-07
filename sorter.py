"""Utilities for ordering vehicle detections based on lane orientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from smart_traffic_system import VehicleDetection

__all__ = ["VehicleSorter"]


@dataclass(slots=True)
class VehicleSorter:
    """Sort detections to reflect the physical queue order for a lane."""

    orientation: str = "vertical"

    def __post_init__(self) -> None:
        orientation = self.orientation.lower()
        if orientation not in {"vertical", "horizontal"}:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
        self.orientation = orientation

    def sort(self, detections: Sequence["VehicleDetection"]) -> List["VehicleDetection"]:
        if self.orientation == "vertical":
            key_func = lambda det: det.bottom_edge
        else:
            key_func = lambda det: det.right_edge

        return sorted(detections, key=key_func, reverse=True)
