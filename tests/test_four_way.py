"""Tests for Case 3 — the adaptive four-way intersection."""

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

from four_way_intersection import Approach, ApproachVehicle, FourWayController


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def make_controller():
    clock = FakeClock()
    return clock, FourWayController(time_func=clock)


def test_green_axis_holds_while_cross_axis_is_empty():
    clock, controller = make_controller()

    for _ in range(20):
        clock.advance(5.0)
        status = controller.update({"N": 3, "S": 2, "E": 0, "W": 0})

    # 100 seconds of green far beyond MAX_GREEN, but nobody is waiting:
    # no reason to ever switch.
    assert status["active_axis"] == "NS"
    assert status["phase"] == "GREEN"
    assert status["signals"] == {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"}


def test_empty_axis_is_skipped_early_for_waiting_cross_traffic():
    clock, controller = make_controller()

    status = controller.update({"N": 0, "S": 0, "E": 2, "W": 1})
    assert status["phase"] == "GREEN"  # min green must still be served

    clock.advance(controller.MIN_GREEN + 0.1)
    status = controller.update({"N": 0, "S": 0, "E": 2, "W": 1})
    assert status["phase"] == "YELLOW"
    assert status["early_switches"] == 1

    clock.advance(controller.YELLOW_TIME + 0.05)
    status = controller.update({"N": 0, "S": 0, "E": 2, "W": 1})
    assert status["phase"] == "ALL_RED"
    assert status["signals"] == {"N": "RED", "S": "RED", "E": "RED", "W": "RED"}

    clock.advance(controller.ALL_RED_TIME + 0.05)
    status = controller.update({"N": 0, "S": 0, "E": 2, "W": 1})
    assert status["active_axis"] == "EW"
    assert status["phase"] == "GREEN"
    assert status["signals"] == {"N": "RED", "S": "RED", "E": "GREEN", "W": "GREEN"}


def test_max_green_caps_a_busy_axis_when_cross_traffic_waits():
    clock, controller = make_controller()

    counts = {"N": 5, "S": 5, "E": 1, "W": 0}
    status = controller.update(counts)
    assert status["phase"] == "GREEN"

    clock.advance(controller.MAX_GREEN + 0.1)
    status = controller.update(counts)
    assert status["phase"] == "YELLOW"
    assert status["early_switches"] == 0  # this was a fairness switch


def test_max_red_failsafe_serves_axis_with_no_measured_demand():
    """Detector-failure recall: even with zero measured cross demand (dead
    camera, mis-calibrated zone), the cross axis is served after MAX_RED."""

    clock, controller = make_controller()
    counts = {"N": 3, "S": 1, "E": 0, "W": 0}

    controller.update(counts)
    clock.advance(controller.MAX_RED - 1.0)
    status = controller.update(counts)
    assert status["phase"] == "GREEN"  # not yet

    clock.advance(1.1)
    status = controller.update(counts)
    assert status["phase"] == "YELLOW"  # recall fired without measured demand


def test_red_time_is_stamped_at_yellow_onset():
    """The losing axis's red clock starts when its yellow begins, not when
    the cross green starts — red time must not be under-counted."""

    clock, controller = make_controller()
    counts = {"N": 0, "S": 0, "E": 2, "W": 0}

    clock.advance(controller.MIN_GREEN + 0.1)
    status = controller.update(counts)
    assert status["phase"] == "YELLOW"
    yellow_onset = clock.current
    assert controller._red_start["NS"] == pytest.approx(yellow_onset)


def test_full_cycle_returns_to_first_axis():
    clock, controller = make_controller()
    counts = {"N": 1, "S": 0, "E": 1, "W": 0}

    def advance_to_axis(axis):
        for _ in range(100):
            clock.advance(1.0)
            status = controller.update(counts)
            if status["active_axis"] == axis and status["phase"] == "GREEN":
                return status
        raise AssertionError(f"never reached {axis} green")

    counts = {"N": 0, "S": 0, "E": 1, "W": 0}
    advance_to_axis("EW")
    counts = {"N": 1, "S": 0, "E": 0, "W": 0}
    status = advance_to_axis("NS")
    assert status["signals"]["N"] == "GREEN"
    assert controller.total_switches == 2


def test_approach_queues_and_clears_vehicles():
    approach = Approach(name="N", spawn_rate=0.0)
    approach.vehicles = [
        ApproachVehicle(distance=10.0, speed=80.0),
        ApproachVehicle(distance=80.0, speed=80.0),
    ]

    # Red: the leader stops at the line, the follower queues behind it.
    for _ in range(30):
        approach.step("RED", 0.1)
    leader, follower = sorted(approach.vehicles, key=lambda v: v.distance)
    assert leader.distance == pytest.approx(0.0)
    assert follower.distance >= leader.distance + leader.length
    assert follower.wait_time > 0

    # Green: everyone proceeds and eventually clears the intersection.
    for _ in range(80):
        approach.step("GREEN", 0.1)
    assert not approach.vehicles


def test_demand_count_only_sees_detection_zone():
    approach = Approach(name="E", spawn_rate=0.0, detection_length=110.0)
    approach.vehicles = [
        ApproachVehicle(distance=50.0, speed=70.0),    # counted
        ApproachVehicle(distance=200.0, speed=70.0),   # too far upstream
        ApproachVehicle(distance=-20.0, speed=70.0),   # already crossing
    ]
    assert approach.demand_count() == 1


@pytest.mark.skipif(cv2 is None or np is None, reason="requires cv2 and numpy")
def test_simulation_runs_headless_with_adaptive_switching():
    from four_way_intersection import FourWaySimulation

    sim = FourWaySimulation(
        fps=30,
        seed=11,
        spawn_rates={"N": 0.5, "S": 0.4, "E": 0.15, "W": 0.1},
    )
    sim.run(max_frames=30 * 90, display_window=False)  # 90 simulated seconds

    assert sim.controller.total_switches >= 2
    # Frame renders without errors and has the expected shape.
    frame = sim.render(sim.controller.status(sim.counts()))
    assert frame.shape == (sim.size, sim.size, 3)
