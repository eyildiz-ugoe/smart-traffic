"""Tests for Case 1 — the adaptive pedestrian crossing."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

try:  # pragma: no cover - optional test dependency
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - headless environments
    cv2 = None
    np = None

from pedestrian_crossing import PedestrianSignalController


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def make_controller():
    clock = FakeClock()
    return clock, PedestrianSignalController(time_func=clock)


def test_cars_stay_green_forever_without_pedestrians():
    clock, controller = make_controller()

    for _ in range(50):
        clock.advance(10.0)
        status = controller.update(pedestrian_waiting=False, vehicle_count=3)

    assert status["state"] == "CAR_GREEN"
    assert status["car_signal"] == "GREEN"
    assert status["ped_signal"] == "DONT_WALK"


def test_pedestrian_triggers_full_walk_cycle():
    clock, controller = make_controller()

    status = controller.update(pedestrian_waiting=True, vehicle_count=0)
    assert status["state"] == "CAR_GREEN"  # min car green not yet served

    clock.advance(controller.MIN_CAR_GREEN + 0.1)
    status = controller.update(pedestrian_waiting=True, vehicle_count=0)
    assert status["state"] == "CAR_YELLOW"
    assert status["car_signal"] == "YELLOW"

    clock.advance(controller.YELLOW_TIME + 0.05)
    status = controller.update(pedestrian_waiting=True)
    assert status["state"] == "ALL_RED"
    assert status["car_signal"] == "RED"
    assert status["ped_signal"] == "DONT_WALK"

    clock.advance(controller.ALL_RED_TIME + 0.05)
    status = controller.update(pedestrian_waiting=True)
    assert status["state"] == "WALK"
    assert status["ped_signal"] == "WALK"
    assert status["car_signal"] == "RED"
    assert status["pedestrians_served"] == 1

    clock.advance(controller.WALK_TIME + 0.05)
    status = controller.update(pedestrian_waiting=False)
    assert status["state"] == "PED_CLEARANCE"
    assert status["car_signal"] == "RED"

    clock.advance(controller.PED_CLEARANCE_TIME + 0.05)
    status = controller.update(pedestrian_waiting=False)
    assert status["state"] == "CAR_GREEN"
    assert status["car_signal"] == "GREEN"


def test_dilemma_zone_defers_the_switch():
    clock, controller = make_controller()

    clock.advance(controller.MIN_CAR_GREEN + 1.0)
    status = controller.update(
        pedestrian_waiting=True, vehicle_in_dilemma_zone=True, vehicle_count=2
    )
    assert status["state"] == "CAR_GREEN"  # fast car too close: hold green

    status = controller.update(
        pedestrian_waiting=True, vehicle_in_dilemma_zone=False, vehicle_count=2
    )
    assert status["state"] == "CAR_YELLOW"  # gap found: begin changeover


def test_max_wait_overrides_constant_traffic():
    clock, controller = make_controller()

    controller.update(pedestrian_waiting=True, vehicle_in_dilemma_zone=True, vehicle_count=4)
    clock.advance(controller.MAX_PED_WAIT + 1.0)
    status = controller.update(
        pedestrian_waiting=True, vehicle_in_dilemma_zone=True, vehicle_count=4
    )
    assert status["state"] == "CAR_YELLOW"


def test_raw_zero_count_cannot_bypass_debounced_dilemma_guard():
    """Regression: a single-frame detection dropout (count 0) must not start
    the yellow while the debounced dilemma-zone flag still reports a car."""

    clock, controller = make_controller()

    controller.update(pedestrian_waiting=True, vehicle_in_dilemma_zone=True, vehicle_count=2)
    clock.advance(controller.MIN_CAR_GREEN + 1.0)
    status = controller.update(
        pedestrian_waiting=True, vehicle_in_dilemma_zone=True, vehicle_count=0
    )
    assert status["state"] == "CAR_GREEN"  # debounced flag wins over raw count


def test_brief_pedestrian_occlusion_does_not_reset_wait_clock():
    """Regression: occlusion dropouts shorter than PED_ABSENCE_RESET must not
    zero the MAX_PED_WAIT fairness clock (pedestrian starvation)."""

    clock, controller = make_controller()

    status = controller.update(
        pedestrian_waiting=True, vehicle_in_dilemma_zone=True, vehicle_count=3
    )
    # Frame-cadence updates (10 Hz): the pedestrian detection drops out for a
    # single frame every ~10 s; traffic never yields a gap.
    for i in range(600):
        clock.advance(0.1)
        waiting = (i % 100) != 50
        status = controller.update(
            pedestrian_waiting=waiting, vehicle_in_dilemma_zone=True, vehicle_count=3
        )
        if status["state"] == "CAR_YELLOW":
            break

    # The accumulated wait crossed MAX_PED_WAIT despite the flickers.
    assert status["state"] == "CAR_YELLOW"
    assert clock.current == pytest.approx(controller.MAX_PED_WAIT, abs=1.0)


def test_sustained_pedestrian_absence_clears_the_request():
    clock, controller = make_controller()

    controller.update(pedestrian_waiting=True, vehicle_in_dilemma_zone=True, vehicle_count=3)
    clock.advance(controller.PED_ABSENCE_RESET + 1.0)
    controller.update(pedestrian_waiting=False, vehicle_in_dilemma_zone=True, vehicle_count=3)
    # Second update after the sustained absence: request must be gone.
    status = controller.update(
        pedestrian_waiting=False, vehicle_in_dilemma_zone=False, vehicle_count=0
    )
    assert status["state"] == "CAR_GREEN"
    assert status["ped_wait_time"] == 0.0


def test_empty_road_serves_pedestrian_after_min_green():
    clock, controller = make_controller()

    controller.update(pedestrian_waiting=True, vehicle_count=0)
    clock.advance(controller.MIN_CAR_GREEN + 0.1)
    status = controller.update(pedestrian_waiting=True, vehicle_count=0)
    assert status["state"] == "CAR_YELLOW"


@pytest.mark.skipif(cv2 is None or np is None, reason="requires cv2 and numpy")
def test_simulation_runs_headless_and_serves_pedestrians():
    from pedestrian_crossing import PedestrianCrossingSimulation

    sim = PedestrianCrossingSimulation(fps=30, seed=3, pedestrian_rate=1.2, car_spawn_rate=0.2)
    sim.run(max_frames=30 * 60, display_window=False)  # 60 simulated seconds

    assert sim.controller.pedestrians_served >= 1
    # No pedestrian left stranded outside the visible area bounds.
    for ped in sim.pedestrians:
        assert sim._left_wait_x - 12 <= ped.x <= sim._right_wait_x + 12


@pytest.mark.skipif(cv2 is None or np is None, reason="requires cv2 and numpy")
def test_simulation_dilemma_zone_detection():
    from pedestrian_crossing import PedestrianCrossingSimulation
    from smart_traffic_system import SimulatedVehicle

    sim = PedestrianCrossingSimulation(fps=30, seed=1, car_spawn_rate=0.0, pedestrian_rate=0.0)
    stop_line = sim.road.stop_line

    # Vehicle far upstream: not in the dilemma zone.
    sim.road.vehicles = [
        SimulatedVehicle(position=stop_line - 400, speed=180.0, length=70, width=40, color=(0, 0, 0))
    ]
    assert not sim.vehicle_in_dilemma_zone()

    # Vehicle just before the stop line: inside the dilemma zone.
    sim.road.vehicles = [
        SimulatedVehicle(position=stop_line - 100, speed=180.0, length=70, width=40, color=(0, 0, 0))
    ]
    assert sim.vehicle_in_dilemma_zone()
