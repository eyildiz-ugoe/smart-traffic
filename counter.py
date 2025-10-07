"""Vehicle counting utilities for the smart traffic system."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from smart_traffic_system import VehicleDetection


__all__ = ["VehicleCounter"]


@dataclass(slots=True)
class VehicleCounter:
    """Keep track of per-frame and cumulative vehicle tallies."""

    total_detected: int = 0
    totals_per_class: Counter = field(default_factory=Counter)

    def summarize(self, detections: Sequence["VehicleDetection"]) -> Tuple[int, Dict[int, int]]:
        """Return the count and class histogram for the provided detections."""

        frame_counter: Counter = Counter(det.class_id for det in detections)
        frame_total = sum(frame_counter.values())

        if frame_total:
            self.total_detected += frame_total
            self.totals_per_class.update(frame_counter)

        return frame_total, dict(frame_counter)

    def reset(self) -> None:
        """Clear cumulative statistics (useful for restarting feeds)."""

        self.total_detected = 0
        self.totals_per_class.clear()
