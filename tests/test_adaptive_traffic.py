"""Tests for the adaptive_traffic package (controller, world, configs)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from adaptive_traffic.controller import AdaptiveController
from adaptive_traffic.counters.camera import CameraCounterConfig
from adaptive_traffic.simulation.world import Road, SimulationWorld, SimVehicle


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def test_vehicles_past_stop_line_keep_moving_until_removed():
    """Regression: vehicles used to freeze forever once position < -5."""

    road_a = Road(name="A", clear_distance=30.0)
    road_b = Road(name="B", spawn_rate_per_minute=0.0)
    road_a.spawn_rate_per_minute = 0.0
    world = SimulationWorld(road_a, road_b)

    vehicle = SimVehicle(position=-6.0, speed=10.0)
    road_a.vehicles.append(vehicle)

    # Even with the light red, a vehicle already past the stop line clears.
    world.update(1.0, green_roads=())
    assert vehicle.position == pytest.approx(-16.0)

    world.update(1.0, green_roads=())
    world.update(1.0, green_roads=())
    assert not road_a.vehicles, "vehicle past clear_distance must be removed"


def test_count_in_detection_excludes_vehicles_past_stop_line():
    road = Road(name="A", detection_length=70.0)
    road.vehicles = [
        SimVehicle(position=50.0, speed=10.0),   # in detection zone
        SimVehicle(position=100.0, speed=10.0),  # upstream of zone
        SimVehicle(position=-3.0, speed=10.0),   # already crossed: not demand
    ]

    assert road.count_in_detection() == 1


def test_red_light_holds_vehicles_upstream_of_stop_line():
    road_a = Road(name="A", spawn_rate_per_minute=0.0)
    road_b = Road(name="B", spawn_rate_per_minute=0.0)
    world = SimulationWorld(road_a, road_b)

    waiting = SimVehicle(position=40.0, speed=10.0)
    road_a.vehicles.append(waiting)

    world.update(1.0, green_roads=())
    assert waiting.position == pytest.approx(40.0)
    assert waiting.wait_time == pytest.approx(1.0)

    world.update(1.0, green_roads=("A",))
    assert waiting.position == pytest.approx(30.0)


def test_adaptive_controller_inserts_yellow_between_greens():
    clock = FakeClock()
    controller = AdaptiveController(time_func=clock)

    lights = controller.update(count_a=0, count_b=3)
    assert lights == ("green", "red")

    clock.advance(controller.min_green_time + 0.1)
    lights = controller.update(count_a=0, count_b=3)
    assert lights == ("yellow", "red")

    clock.advance(controller.yellow_time + 0.1)
    lights = controller.update(count_a=0, count_b=3)
    assert lights == ("red", "green")


def test_camera_counter_defaults_to_vehicle_classes():
    """Regression: classes=None made YOLO count people and animals as cars."""

    config = CameraCounterConfig(video_path=Path("dummy.mp4"), detection_zone_size=0.25)
    assert tuple(config.classes) == (2, 3, 5, 7)


def test_resolve_detection_device_falls_back_without_cuda(monkeypatch):
    from smart_traffic_system import resolve_detection_device

    # Non-CUDA requests pass through untouched.
    assert resolve_detection_device(None) is None
    assert resolve_detection_device("cpu") == "cpu"

    # With torch missing entirely, CUDA requests fall back to CPU.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_detection_device("cuda") == "cpu"
