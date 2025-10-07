"""Core logic shared between the simulation and unit tests."""

from dataclasses import dataclass
from typing import Callable, Dict
import time


@dataclass
class TrafficStats:
    """Data class to store traffic statistics for a road."""

    vehicle_count: int = 0
    avg_vehicles_per_frame: float = 0.0
    congestion_level: str = "LOW"
    total_frames: int = 0

    def update(self, current_count: int) -> None:
        """Update statistics with new vehicle count."""

        self.total_frames += 1
        self.vehicle_count = current_count
        self.avg_vehicles_per_frame = (
            (self.avg_vehicles_per_frame * (self.total_frames - 1) + current_count)
            / self.total_frames
        )

        if current_count == 0:
            self.congestion_level = "CLEAR"
        elif current_count < 3:
            self.congestion_level = "LOW"
        elif current_count < 7:
            self.congestion_level = "MEDIUM"
        else:
            self.congestion_level = "HIGH"


class TrafficLightController:
    """Controls traffic light timing based on vehicle density on two roads."""

    STATE_ROAD1_GREEN = 0
    STATE_ROAD2_GREEN = 1

    MIN_GREEN_TIME = 5
    MAX_GREEN_TIME = 30
    YELLOW_TIME = 3
    RED_TIME = 2

    def __init__(self, time_func: Callable[[], float] | None = None) -> None:
        self._time_func = time_func or time.time
        self.current_state = self.STATE_ROAD1_GREEN
        self.state_start_time = self._time_func()
        self.green_time_road1 = self.MIN_GREEN_TIME
        self.green_time_road2 = self.MIN_GREEN_TIME

    def calculate_green_time(self, vehicle_count: int) -> int:
        if vehicle_count == 0:
            return self.MIN_GREEN_TIME
        if vehicle_count < 3:
            return 10
        if vehicle_count < 7:
            return 20
        return self.MAX_GREEN_TIME

    def update_signal_timing(self, road1_vehicles: int, road2_vehicles: int) -> Dict[str, object]:
        current_time = self._time_func()
        elapsed_time = current_time - self.state_start_time

        self.green_time_road1 = self.calculate_green_time(road1_vehicles)
        self.green_time_road2 = self.calculate_green_time(road2_vehicles)

        signal_status: Dict[str, object] = {
            "road1": "RED",
            "road2": "RED",
            "time_remaining": 0.0,
            "next_switch": False,
        }

        if self.current_state == self.STATE_ROAD1_GREEN:
            if elapsed_time < self.green_time_road1:
                signal_status["road1"] = "GREEN"
                signal_status["road2"] = "RED"
                signal_status["time_remaining"] = self.green_time_road1 - elapsed_time
            elif elapsed_time < self.green_time_road1 + self.YELLOW_TIME:
                signal_status["road1"] = "YELLOW"
                signal_status["road2"] = "RED"
                signal_status["time_remaining"] = (
                    self.green_time_road1 + self.YELLOW_TIME - elapsed_time
                )
            else:
                signal_status["next_switch"] = True
                self.current_state = self.STATE_ROAD2_GREEN
                self.state_start_time = current_time
                signal_status["road1"] = "RED"
                signal_status["road2"] = "GREEN"
                signal_status["time_remaining"] = self.green_time_road2
        else:
            if elapsed_time < self.green_time_road2:
                signal_status["road1"] = "RED"
                signal_status["road2"] = "GREEN"
                signal_status["time_remaining"] = self.green_time_road2 - elapsed_time
            elif elapsed_time < self.green_time_road2 + self.YELLOW_TIME:
                signal_status["road1"] = "RED"
                signal_status["road2"] = "YELLOW"
                signal_status["time_remaining"] = (
                    self.green_time_road2 + self.YELLOW_TIME - elapsed_time
                )
            else:
                signal_status["next_switch"] = True
                self.current_state = self.STATE_ROAD1_GREEN
                self.state_start_time = current_time
                signal_status["road1"] = "GREEN"
                signal_status["road2"] = "RED"
                signal_status["time_remaining"] = self.green_time_road1

        return signal_status
