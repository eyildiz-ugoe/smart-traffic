"""Pygame visualization for the simulated intersection."""

from __future__ import annotations

import logging
from typing import Dict

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

COLOR_BACKGROUND = (25, 28, 33)
COLOR_ROAD = (72, 76, 83)
COLOR_ROAD_EDGE = (54, 58, 63)
COLOR_LANE_MARK = (150, 150, 150)
COLOR_TEXT = (235, 235, 235)
COLOR_VEHICLE_A = (70, 180, 255)
COLOR_VEHICLE_B = (255, 180, 70)
COLOR_VEHICLE_GLASS = (210, 230, 245)
COLOR_LIGHT_HOUSING = (32, 32, 36)
COLOR_LIGHT_OFF = (70, 70, 70)
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

        self.road_width = 140
        self.detection_depth = 140
        self.vehicle_length = 42
        self.vehicle_width = 28
        margin = 36
        self.intersection_size = self.road_width
        self.font_small = pygame.font.Font(None, 18)
        self.font_label = pygame.font.Font(None, 22)

        intersection_half = self.intersection_size // 2
        self.intersection_rect = pygame.Rect(
            self.width // 2 - intersection_half,
            self.height // 2 - intersection_half,
            self.intersection_size,
            self.intersection_size,
        )
        self.road_a_rect = pygame.Rect(
            self.intersection_rect.left,
            self.intersection_rect.centery,
            self.road_width,
            self.height - self.intersection_rect.centery,
        )
        self.road_b_rect = pygame.Rect(
            0,
            self.intersection_rect.top,
            self.intersection_rect.centerx,
            self.road_width,
        )
        self.light_box_a = pygame.Rect(
            self.intersection_rect.right + margin,
            self.intersection_rect.bottom - self.road_width // 4,
            36,
            96,
        )
        self.light_box_b = pygame.Rect(
            self.intersection_rect.left - (96 + margin),
            self.intersection_rect.top - 36,
            96,
            36,
        )

    def _draw_roads(self) -> None:
        pygame.draw.rect(self.surface, COLOR_ROAD_EDGE, self.road_a_rect.inflate(12, 12))
        pygame.draw.rect(self.surface, COLOR_ROAD_EDGE, self.road_b_rect.inflate(12, 12))
        pygame.draw.rect(self.surface, COLOR_ROAD, self.road_a_rect)
        pygame.draw.rect(self.surface, COLOR_ROAD, self.road_b_rect)
        pygame.draw.rect(self.surface, COLOR_ROAD, self.intersection_rect)

        lane_line_length = 24
        lane_gap = 18
        lane_x = self.road_a_rect.centerx
        start_y = self.intersection_rect.bottom + 12
        end_y = self.road_a_rect.bottom
        y = start_y
        while y < end_y:
            pygame.draw.line(
                self.surface,
                COLOR_LANE_MARK,
                (lane_x, y),
                (lane_x, min(y + lane_line_length, end_y)),
                2,
            )
            y += lane_line_length + lane_gap

        lane_y = self.road_b_rect.centery
        start_x = self.road_b_rect.left
        end_x = self.intersection_rect.left - 12
        x = start_x
        while x < end_x:
            pygame.draw.line(
                self.surface,
                COLOR_LANE_MARK,
                (x, lane_y),
                (min(x + lane_line_length, end_x), lane_y),
                2,
            )
            x += lane_line_length + lane_gap

    def _draw_light(
        self,
        rect: pygame.Rect,
        state: str,
        label: str,
        horizontal: bool,
        anchor: tuple[int, int],
    ) -> None:
        pygame.draw.rect(self.surface, COLOR_LIGHT_HOUSING, rect, border_radius=8)
        connector_start = rect.midright if horizontal else rect.midleft
        pygame.draw.line(self.surface, COLOR_LANE_MARK, connector_start, anchor, 3)
        pygame.draw.circle(self.surface, COLOR_LANE_MARK, anchor, 4)

        padding = 8
        radius = max(6, (min(rect.width, rect.height) - padding * 2) // 4)
        if horizontal:
            positions = [
                (rect.left + padding + radius, rect.centery),
                (rect.centerx, rect.centery),
                (rect.right - padding - radius, rect.centery),
            ]
        else:
            positions = [
                (rect.centerx, rect.top + padding + radius),
                (rect.centerx, rect.centery),
                (rect.centerx, rect.bottom - padding - radius),
            ]

        lights = [
            ("red", COLOR_LIGHT_RED, positions[0]),
            ("yellow", COLOR_LIGHT_YELLOW, positions[1]),
            ("green", COLOR_LIGHT_GREEN, positions[2]),
        ]

        for name, color, center in lights:
            pygame.draw.circle(
                self.surface,
                color if state == name else COLOR_LIGHT_OFF,
                center,
                radius,
            )
            pygame.draw.circle(self.surface, COLOR_LIGHT_HOUSING, center, radius, 2)

        label_surface = self.font_label.render(label, True, COLOR_TEXT)
        if horizontal:
            label_pos = label_surface.get_rect()
            label_pos.midtop = (rect.centerx, rect.bottom + 6)
        else:
            label_pos = label_surface.get_rect()
            label_pos.midleft = (rect.right + 8, rect.centery)
        self.surface.blit(label_surface, label_pos)

    def _draw_detection_zone(self) -> None:
        overlay = pygame.Surface((self.width, self.height), flags=pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        zone_a_height = min(self.detection_depth, self.height - self.intersection_rect.bottom)
        zone_a = pygame.Rect(
            self.road_a_rect.left,
            self.intersection_rect.bottom,
            self.road_width,
            zone_a_height,
        )
        zone_b_width = min(self.detection_depth, self.intersection_rect.left)
        zone_b = pygame.Rect(
            self.intersection_rect.left - zone_b_width,
            self.road_b_rect.top,
            zone_b_width,
            self.road_width,
        )
        pygame.draw.rect(overlay, (0, 140, 220, 65), zone_a)
        pygame.draw.rect(overlay, (220, 140, 0, 65), zone_b)
        self.surface.blit(overlay, (0, 0))

    def _draw_vehicles(self, world: SimulationWorld) -> None:
        lane_center_x = self.road_a_rect.centerx
        lane_center_y = self.road_b_rect.centery

        for vehicle in world.roads["A"].vehicles:
            front_position = max(-self.vehicle_length, vehicle.position)
            front_y = int(self.height // 2 + front_position)
            body_rect = pygame.Rect(
                lane_center_x - self.vehicle_width // 2,
                front_y - self.vehicle_length,
                self.vehicle_width,
                self.vehicle_length,
            )
            pygame.draw.rect(self.surface, COLOR_VEHICLE_A, body_rect, border_radius=6)
            hood = [
                (body_rect.left, body_rect.top),
                (body_rect.right, body_rect.top),
                (body_rect.centerx, body_rect.top - 8),
            ]
            pygame.draw.polygon(self.surface, COLOR_VEHICLE_A, hood)
            window_rect = body_rect.inflate(-8, -18)
            window_rect.height = max(6, window_rect.height)
            pygame.draw.rect(self.surface, COLOR_VEHICLE_GLASS, window_rect, border_radius=4)

        for vehicle in world.roads["B"].vehicles:
            front_position = max(-self.vehicle_length, vehicle.position)
            front_x = int(self.width // 2 - front_position)
            body_rect = pygame.Rect(
                front_x - self.vehicle_length,
                lane_center_y - self.vehicle_width // 2,
                self.vehicle_length,
                self.vehicle_width,
            )
            pygame.draw.rect(self.surface, COLOR_VEHICLE_B, body_rect, border_radius=6)
            hood = [
                (body_rect.right, body_rect.top),
                (body_rect.right, body_rect.bottom),
                (body_rect.right + 8, body_rect.centery),
            ]
            pygame.draw.polygon(self.surface, COLOR_VEHICLE_B, hood)
            window_rect = body_rect.inflate(-18, -8)
            window_rect.width = max(6, window_rect.width)
            pygame.draw.rect(self.surface, COLOR_VEHICLE_GLASS, window_rect, border_radius=4)

    def _draw_stats(self, context: Dict[str, object]) -> None:
        counts = context.get("counts", {"A": 0, "B": 0})
        waits = context.get("waits", {"A": 0.0, "B": 0.0})
        text_lines = [
            f"Road A vehicles: {counts['A']} wait: {waits['A']:.1f}s",
            f"Road B vehicles: {counts['B']} wait: {waits['B']:.1f}s",
        ]
        for idx, text in enumerate(text_lines):
            surface = self.font_small.render(text, True, COLOR_TEXT)
            self.surface.blit(surface, (20, 20 + idx * 22))

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
        anchor_a = (self.road_a_rect.centerx, self.intersection_rect.bottom)
        anchor_b = (self.intersection_rect.left, self.road_b_rect.centery)
        self._draw_light(self.light_box_a, lights[0], "Road A", horizontal=False, anchor=anchor_a)
        self._draw_light(self.light_box_b, lights[1], "Road B", horizontal=True, anchor=anchor_b)
        self._draw_stats(context)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self) -> None:
        if pygame is not None:
            pygame.quit()
