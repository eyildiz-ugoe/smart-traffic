"""High level orchestration of the adaptive traffic light system."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple

from .config import TrafficConfig
from .controller import AdaptiveController
from .counters.camera import CameraCounter, CameraCounterConfig
from .counters.simulation import SimulatedCounter, SimulationCounterConfig
from .simulation.world import Road, SimulationWorld
from .visualization.base import VisualizationStrategy
from .visualization.realistic import RealisticVisualization
from .visualization.simulation import SimulationVisualization

logger = logging.getLogger(__name__)


class ModeStrategy(ABC):
    """Strategy pattern implementation for running different modes."""

    def __init__(self, controller: AdaptiveController) -> None:
        self.controller = controller

    @abstractmethod
    def run(self) -> None:
        """Execute the strategy main loop."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources used by the strategy."""


class RealisticModeStrategy(ModeStrategy):
    """Strategy that processes camera feeds and displays OpenCV windows."""

    def __init__(self, controller: AdaptiveController, config: TrafficConfig) -> None:
        super().__init__(controller)
        config.ensure_paths()
        if config.video_path_road_a is None or config.video_path_road_b is None:
            raise ValueError("Realistic mode requires both video paths to be provided")
        self.counter_a = CameraCounter(
            CameraCounterConfig(
                video_path=config.video_path_road_a,  # type: ignore[arg-type]
                detection_zone_size=config.detection_zone_size,
                confidence=config.detection_confidence,
            )
        )
        self.counter_b = CameraCounter(
            CameraCounterConfig(
                video_path=config.video_path_road_b,  # type: ignore[arg-type]
                detection_zone_size=config.detection_zone_size,
                confidence=config.detection_confidence,
            )
        )
        self.visualization: VisualizationStrategy = RealisticVisualization()
        self._running = True

    def run(self) -> None:  # pragma: no cover - relies on real video streams
        while self._running:
            result_a = self.counter_a.step()
            result_b = self.counter_b.step()
            lights = self.controller.update(result_a.count, result_b.count)
            self.visualization.render(
                {
                    "frame_a": result_a.frame,
                    "frame_b": result_b.frame,
                    "lights": lights,
                }
            )

    def close(self) -> None:
        self._running = False
        self.counter_a.close()
        self.counter_b.close()
        self.visualization.close()


class SimulationModeStrategy(ModeStrategy):
    """Strategy that simulates vehicles and renders via pygame."""

    def __init__(self, controller: AdaptiveController, config: TrafficConfig) -> None:
        super().__init__(controller)
        road_a = Road(
            name="A",
            detection_length=70.0,
            spawn_rate_per_minute=config.spawn_rate_road_a,
        )
        road_b = Road(
            name="B",
            detection_length=70.0,
            spawn_rate_per_minute=config.spawn_rate_road_b,
        )
        self.world = SimulationWorld(road_a, road_b)
        self.counter_a = SimulatedCounter(
            SimulationCounterConfig(road_name="A", world=self.world)
        )
        self.counter_b = SimulatedCounter(
            SimulationCounterConfig(road_name="B", world=self.world)
        )
        self.visualization: VisualizationStrategy = SimulationVisualization()
        self._running = True
        self._lights: Tuple[str, str] = ("green", "red")
        self._last_time = time.perf_counter()

    def _greens(self) -> Tuple[str, ...]:
        return tuple(road for road, light in zip(("A", "B"), self._lights) if light == "green")

    def run(self) -> None:  # pragma: no cover - requires pygame event loop
        while self._running:
            now = time.perf_counter()
            dt = now - self._last_time
            self._last_time = now
            self.world.update(dt, self._greens())
            result_a = self.counter_a.step()
            result_b = self.counter_b.step()
            self._lights = self.controller.update(result_a.count, result_b.count)
            context = {
                "world": self.world,
                "lights": self._lights,
                "counts": {"A": result_a.count, "B": result_b.count},
                "waits": self.world.average_waits(),
            }
            self.visualization.render(context)

    def close(self) -> None:
        self._running = False
        self.visualization.close()


class TrafficSystem:
    """Main entry point orchestrating the adaptive traffic system."""

    def __init__(self, config: TrafficConfig) -> None:
        self.config = config
        self.controller = AdaptiveController(
            min_green_time=config.min_green_time,
            max_red_time=config.max_red_time,
            yellow_time=config.yellow_time,
        )
        if config.mode == "realistic":
            self.strategy: ModeStrategy = RealisticModeStrategy(self.controller, config)
        else:
            self.strategy = SimulationModeStrategy(self.controller, config)

    def run(self) -> None:
        """Run the configured strategy until interrupted."""

        try:
            self.strategy.run()
        except KeyboardInterrupt:
            logger.info("Traffic system interrupted by user")
        finally:
            self.strategy.close()
