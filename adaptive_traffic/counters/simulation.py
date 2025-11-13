"""Vehicle counter using the internal simulation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import CounterResult, VehicleCounter
from ..simulation.world import Road, SimulationWorld


@dataclass(slots=True)
class SimulationCounterConfig:
    """Configuration for :class:`SimulatedCounter`."""

    road_name: str
    world: SimulationWorld


class SimulatedCounter(VehicleCounter):
    """Vehicle counter that reads from a :class:`SimulationWorld`."""

    def __init__(self, config: SimulationCounterConfig) -> None:
        self.config = config
        self._latest_count = 0

    @property
    def road(self) -> Road:
        return self.config.world.roads[self.config.road_name]

    def step(self) -> CounterResult:
        self._latest_count = self.road.count_in_detection()
        return CounterResult(count=self._latest_count, frame=None)

    def get_count(self) -> int:
        return self._latest_count

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None
