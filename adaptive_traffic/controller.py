"""Adaptive traffic signal controller shared between both operating modes."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Tuple

LightState = str


@dataclass(slots=True)
class ControllerState:
    """Internal state representation used by :class:`AdaptiveController`."""

    phase: str
    current_green: str
    pending_green: str
    phase_start: float
    last_green_start: Dict[str, float]
    last_red_start: Dict[str, float]


class AdaptiveController:
    """Adaptive two-phase controller enforcing fairness and safety constraints.

    The controller maintains a simple state machine with two main phases for the
    currently green road (``"A"`` or ``"B"``) and an intermediate yellow phase
    during transitions.  The decision logic follows the specification:

    * Keep a minimum green time before switching to prevent rapid oscillations.
    * If the current green road becomes empty after the minimum green period,
      immediately begin the changeover sequence.
    * Ensure the red duration of each approach never exceeds ``max_red_time`` by
      forcing a switch when required.
    * Insert a fixed yellow phase before the new approach receives green.
    """

    def __init__(
        self,
        min_green_time: float = 5.0,
        max_red_time: float = 60.0,
        yellow_time: float = 3.0,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.min_green_time = min_green_time
        self.max_red_time = max_red_time
        self.yellow_time = yellow_time
        self.time_func = time_func or time.time

        now = self.time_func()
        self.state = ControllerState(
            phase="green",
            current_green="A",
            pending_green="A",
            phase_start=now,
            last_green_start={"A": now, "B": now - max_red_time / 2},
            last_red_start={"A": now, "B": now},
        )

    def _time_since(self, timestamp: float) -> float:
        return self.time_func() - timestamp

    def _other(self, road: str) -> str:
        return "B" if road == "A" else "A"

    def _should_force_switch(self, red_road: str) -> bool:
        red_elapsed = self._time_since(self.state.last_red_start[red_road])
        return red_elapsed >= self.max_red_time

    def _should_switch_for_empty(self, count_current: int, count_other: int) -> bool:
        """Decide if the controller should begin switching due to empty queue."""

        if self._time_since(self.state.phase_start) < self.min_green_time:
            return False
        if count_current > 0:
            return False
        return count_other > 0

    def update(self, count_a: int, count_b: int) -> Tuple[LightState, LightState]:
        """Update the controller state using the latest vehicle counts."""

        now = self.time_func()
        state = self.state

        if state.phase == "green":
            active = state.current_green
            other = self._other(active)
            active_count = count_a if active == "A" else count_b
            other_count = count_b if other == "B" else count_a

            if self._should_switch_for_empty(active_count, other_count) or self._should_force_switch(other):
                state.phase = "yellow"
                state.pending_green = other
                state.phase_start = now
                state.last_red_start[active] = now
        elif state.phase == "yellow":
            if self._time_since(state.phase_start) >= self.yellow_time:
                state.phase = "green"
                state.current_green = state.pending_green
                state.phase_start = now
                state.last_green_start[state.current_green] = now
                other = self._other(state.current_green)
                state.last_red_start[other] = now

        if state.phase == "green":
            if state.current_green == "A":
                return "green", "red"
            return "red", "green"

        # yellow phase keeps the current approach yellow until changeover completes
        if state.current_green == "A":
            return "yellow", "red"
        return "red", "yellow"
