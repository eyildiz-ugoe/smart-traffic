"""Core simulation primitives for the adaptive traffic light system."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List


@dataclass(slots=True)
class SimVehicle:
    """Simple vehicle representation for the simulation."""

    position: float
    speed: float
    wait_time: float = 0.0

    def advance(self, dt: float, can_move: bool) -> None:
        """Update the vehicle position and accumulated wait time."""

        if can_move:
            self.position -= self.speed * dt
            self.wait_time = max(0.0, self.wait_time - dt)
        else:
            self.wait_time += dt


@dataclass(slots=True)
class Road:
    """Container describing a single approach in the simulation."""

    name: str
    length: float = 220.0
    detection_length: float = 70.0
    spawn_rate_per_minute: float = 12.0
    max_speed: float = 12.0
    min_speed: float = 8.0
    vehicles: List[SimVehicle] = field(default_factory=list)

    def spawn_probability(self, dt: float) -> float:
        return (self.spawn_rate_per_minute / 60.0) * dt

    def spawn_vehicle(self) -> None:
        speed = random.uniform(self.min_speed, self.max_speed)
        self.vehicles.append(SimVehicle(position=self.length, speed=speed))

    def remove_passed(self) -> None:
        self.vehicles = [vehicle for vehicle in self.vehicles if vehicle.position > -20]

    def count_in_detection(self) -> int:
        return sum(1 for vehicle in self.vehicles if vehicle.position <= self.detection_length)

    def average_wait(self) -> float:
        if not self.vehicles:
            return 0.0
        return sum(vehicle.wait_time for vehicle in self.vehicles) / len(self.vehicles)


class SimulationWorld:
    """Encapsulates the toy traffic world with two orthogonal roads."""

    def __init__(self, road_a: Road, road_b: Road) -> None:
        self.roads: Dict[str, Road] = {"A": road_a, "B": road_b}

    def update(self, dt: float, green_roads: Iterable[str]) -> None:
        """Advance the world by ``dt`` seconds.

        Parameters
        ----------
        dt:
            Simulation time step in seconds.
        green_roads:
            Names of roads currently showing a green signal.  Red or yellow
            signals prevent vehicles from crossing the stop line.
        """

        green = set(green_roads)
        for road in self.roads.values():
            if random.random() < road.spawn_probability(dt):
                road.spawn_vehicle()

        for name, road in self.roads.items():
            can_move = name in green
            for vehicle in road.vehicles:
                vehicle.advance(dt, can_move=can_move and vehicle.position >= -5)
            road.remove_passed()

    def counts(self) -> Dict[str, int]:
        return {name: road.count_in_detection() for name, road in self.roads.items()}

    def average_waits(self) -> Dict[str, float]:
        return {name: road.average_wait() for name, road in self.roads.items()}
