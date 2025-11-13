"""Pygame visualization for the simulated intersection."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from .base import VisualizationStrategy
from ..simulation.world import SimulationWorld

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional runtime dependency
    import pygame
except Exception as exc:  # pragma: no cover - degrade gracefully
    pygame = None  # type: ignore[assignment]
    _PYGAME_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _PYGAME_IMPORT_ERROR = None

COLOR_ROAD = (80, 80, 80)
COLOR_BACKGROUND = (20, 20, 20)
COLOR_TEXT = (240, 240, 240)
COLOR_VEHICLE_A = (70, 180, 255)
COLOR_VEHICLE_B = (255, 180, 70)
COLOR_LIGHT_GREEN = (0, 200, 0)
COLOR_LIGHT_RED = (200, 0, 0)
COLOR_LIGHT_YELLOW = (230, 210, 0)


class SimulationVisualization(VisualizationStrategy):
    """Render the simulated traffic intersection using Pygame."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        if pygame is None:  # pragma: no cover - executed when dependency missing
            raise RuntimeError(
                "pygame is required for SimulationVisualization but could not be imported"
            ) from _PYGAME_IMPORT_ERROR

        pygame.init()
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Adaptive Traffic - Simulation")
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height

    def _draw_roads(self) -> None:
        road_width = 160
        pygame.draw.rect(
            self.surface,
            COLOR_ROAD,
            ((self.width - road_width) // 2, 0, road_width, self.height),
        )
        pygame.draw.rect(
            self.surface,
            COLOR_ROAD,
            (0, (self.height - road_width) // 2, self.width, road_width),
        )

    def _draw_light(self, position: Tuple[int, int], state: str) -> None:
        color = {
            "green": COLOR_LIGHT_GREEN,
            "yellow": COLOR_LIGHT_YELLOW,
            "red": COLOR_LIGHT_RED,
        }.get(state, COLOR_LIGHT_RED)
        pygame.draw.circle(self.surface, color, position, 20)

    def _draw_detection_zone(self) -> None:
        overlay = pygame.Surface((self.width, self.height), flags=pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        road_width = 160
        zone_depth = 120
        pygame.draw.rect(
            overlay,
            (0, 120, 200, 60),
            ((self.width - road_width) // 2, self.height // 2, road_width, zone_depth),
        )
        pygame.draw.rect(
            overlay,
            (200, 120, 0, 60),
            (self.width // 2 - zone_depth, (self.height - road_width) // 2, zone_depth, road_width),
        )
        self.surface.blit(overlay, (0, 0))

    def _draw_vehicles(self, world: SimulationWorld) -> None:
        road_width = 160
        lane_offset = road_width // 4
        for vehicle in world.roads["A"].vehicles:
            y = int(self.height // 2 + vehicle.position)
            rect = pygame.Rect(
                (self.width - lane_offset) // 2,
                y,
                lane_offset,
                18,
            )
            pygame.draw.rect(self.surface, COLOR_VEHICLE_A, rect)
        for vehicle in world.roads["B"].vehicles:
            x = int(self.width // 2 - vehicle.position)
            rect = pygame.Rect(
                x,
                (self.height - lane_offset) // 2,
                18,
                lane_offset,
            )
            pygame.draw.rect(self.surface, COLOR_VEHICLE_B, rect)

    def _draw_stats(self, context: Dict[str, object]) -> None:
        font = pygame.font.SysFont("Consolas", 18)
        counts = context.get("counts", {"A": 0, "B": 0})
        waits = context.get("waits", {"A": 0.0, "B": 0.0})
        text_lines = [
            f"Road A vehicles: {counts['A']} wait: {waits['A']:.1f}s",
            f"Road B vehicles: {counts['B']} wait: {waits['B']:.1f}s",
        ]
        for idx, text in enumerate(text_lines):
            surface = font.render(text, True, COLOR_TEXT)
            self.surface.blit(surface, (20, 20 + idx * 24))

    def render(self, context: Dict[str, object]) -> None:
        world = context["world"]
        lights = context.get("lights", ("red", "red"))

        for event in pygame.event.get():  # pragma: no cover - interactive loop
            if event.type == pygame.QUIT:
                raise SystemExit

        self.surface.fill(COLOR_BACKGROUND)
        self._draw_roads()
        self._draw_detection_zone()
        self._draw_vehicles(world)
        self._draw_light((self.width // 2 + 80, self.height // 2 - 80), lights[0])
        self._draw_light((self.width // 2 - 80, self.height // 2 + 80), lights[1])
        self._draw_stats(context)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self) -> None:
        if pygame is not None:
            pygame.quit()
