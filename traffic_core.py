"""Core logic shared between the simulation and unit tests."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional
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
    PRESSURE_BONUS_CAP = 5.0
    PRESSURE_BONUS_MULTIPLIER = 2

    def __init__(self, time_func: Callable[[], float] | None = None) -> None:
        self._time_func = time_func or time.time
        self.current_state = self.STATE_ROAD1_GREEN
        self.state_start_time = self._time_func()
        self.green_time_road1 = self.MIN_GREEN_TIME
        self.green_time_road2 = self.MIN_GREEN_TIME
        self._pending_state: Optional[int] = None

    def calculate_green_time(
        self, vehicle_count: int, queue_pressure: Optional[float] = None
    ) -> int:
        """Return a green phase duration that reacts to demand and queue pressure."""

        if vehicle_count == 0:
            base_time = self.MIN_GREEN_TIME
        elif vehicle_count < 3:
            base_time = 10
        elif vehicle_count < 7:
            base_time = 20
        else:
            base_time = self.MAX_GREEN_TIME

        if queue_pressure is None or base_time >= self.MAX_GREEN_TIME:
            return base_time

        dynamic_bonus = max(0.0, queue_pressure - float(vehicle_count))
        if dynamic_bonus <= 0:
            return base_time

        bonus_seconds = min(
            self.MAX_GREEN_TIME - base_time,
            int(
                round(
                    min(dynamic_bonus, self.PRESSURE_BONUS_CAP)
                    * self.PRESSURE_BONUS_MULTIPLIER
                )
            ),
        )
        return base_time + bonus_seconds

    def update_signal_timing(
        self,
        road1_vehicles: int,
        road2_vehicles: int,
        road1_queue_pressure: Optional[float] = None,
        road2_queue_pressure: Optional[float] = None,
        *,
        road1_stopline_occupied: Optional[bool] = None,
        road2_stopline_occupied: Optional[bool] = None,
        road1_exit_ready: Optional[bool] = None,
        road2_exit_ready: Optional[bool] = None,
        road1_leading_edge: Optional[int] = None,
        road2_leading_edge: Optional[int] = None,
        road1_approach_line: Optional[int] = None,
        road2_approach_line: Optional[int] = None,
    ) -> Dict[str, object]:
        current_time = self._time_func()
        elapsed_time = current_time - self.state_start_time

        self.green_time_road1 = self.calculate_green_time(
            road1_vehicles, queue_pressure=road1_queue_pressure
        )
        self.green_time_road2 = self.calculate_green_time(
            road2_vehicles, queue_pressure=road2_queue_pressure
        )

        # Smart early switching: if current road is empty and other has traffic, switch early
        early_switch = False
        
        # Determine if roads have active demand beyond just vehicle count
        road1_has_demand = road1_vehicles > 0
        if road1_stopline_occupied is not None and road1_stopline_occupied:
            road1_has_demand = True
            
        road2_has_demand = road2_vehicles > 0
        if road2_stopline_occupied is not None and road2_stopline_occupied:
            road2_has_demand = True

        if self.current_state == self.STATE_ROAD1_GREEN:
            # Road 1 is green, check if it's empty and road 2 has traffic
            if road1_vehicles == 0 and road2_has_demand and elapsed_time >= self.MIN_GREEN_TIME:
                early_switch = True
        else:
            # Road 2 is green, check if it's empty and road 1 has traffic
            if road2_vehicles == 0 and road1_has_demand and elapsed_time >= self.MIN_GREEN_TIME:
                early_switch = True

        signal_status: Dict[str, object] = {
            "road1": "RED",
            "road2": "RED",
            "time_remaining": 0.0,
            "next_switch": False,
            "active_road": "road1"
            if self.current_state == self.STATE_ROAD1_GREEN
            else "road2",
            "green_durations": {
                "road1": self.green_time_road1,
                "road2": self.green_time_road2,
            },
            "queue_pressure": {
                "road1": road1_queue_pressure,
                "road2": road2_queue_pressure,
            },
            "early_switch_triggered": early_switch,
        }

        if self.current_state == self.STATE_ROAD1_GREEN:
            # Trigger early switch if road is empty
            if early_switch:
                signal_status["road1"] = "YELLOW"
                signal_status["road2"] = "RED"
                signal_status["time_remaining"] = self.YELLOW_TIME
                signal_status["next_switch"] = True
                # Force yellow transition
                if elapsed_time >= self.MIN_GREEN_TIME + self.YELLOW_TIME:
                    self.current_state = self.STATE_ROAD2_GREEN
                    self.state_start_time = current_time
                    signal_status["road1"] = "RED"
                    signal_status["road2"] = "GREEN"
                    signal_status["active_road"] = "road2"
                    signal_status["time_remaining"] = self.green_time_road2
            elif elapsed_time < self.green_time_road1:
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
                signal_status["active_road"] = "road2"
                signal_status["time_remaining"] = self.green_time_road2
        else:
            # Trigger early switch if road is empty
            if early_switch:
                signal_status["road1"] = "RED"
                signal_status["road2"] = "YELLOW"
                signal_status["time_remaining"] = self.YELLOW_TIME
                signal_status["next_switch"] = True
                # Force yellow transition
                if elapsed_time >= self.MIN_GREEN_TIME + self.YELLOW_TIME:
                    self.current_state = self.STATE_ROAD1_GREEN
                    self.state_start_time = current_time
                    signal_status["road1"] = "GREEN"
                    signal_status["road2"] = "RED"
                    signal_status["active_road"] = "road1"
                    signal_status["time_remaining"] = self.green_time_road1
            elif elapsed_time < self.green_time_road2:
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
                signal_status["active_road"] = "road1"
                signal_status["time_remaining"] = self.green_time_road1

        return signal_status

