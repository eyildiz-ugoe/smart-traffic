"""Vehicle counting abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional OpenCV dependency
    import numpy as np
except Exception:  # pragma: no cover - gracefully degrade when numpy unavailable
    np = None  # type: ignore[assignment]


@dataclass(slots=True)
class CounterResult:
    """Outcome returned by :meth:`VehicleCounter.step`.

    Attributes
    ----------
    count:
        Number of vehicles detected inside the detection zone for the most
        recent frame or simulation step.
    frame:
        Optional annotated frame.  Realistic mode provides a :class:`numpy.ndarray`
        while the simulation mode leaves this ``None``.
    """

    count: int
    frame: Optional["np.ndarray"] = None


class VehicleCounter(ABC):
    """Abstract interface for any component capable of counting vehicles."""

    @abstractmethod
    def step(self) -> CounterResult:
        """Advance the counter by one tick and return the most recent counts."""

    @abstractmethod
    def get_count(self) -> int:
        """Return the latest vehicle count detected within the monitored zone."""

    @abstractmethod
    def close(self) -> None:
        """Release any open resources such as video captures."""
