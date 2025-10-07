import pytest

from traffic_core import TrafficLightController, TrafficStats
from traffic_scenarios import TrafficScenario, load_predefined_scenarios


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def test_traffic_stats_update_tracks_average_and_congestion():
    stats = TrafficStats()

    stats.update(0)
    stats.update(2)
    stats.update(5)

    assert stats.vehicle_count == 5
    assert pytest.approx(stats.avg_vehicles_per_frame, rel=1e-3) == (0 + 2 + 5) / 3
    assert stats.congestion_level == "MEDIUM"


def test_controller_switches_after_green_and_yellow():
    clock = FakeClock()
    controller = TrafficLightController(time_func=clock)

    status = controller.update_signal_timing(road1_vehicles=5, road2_vehicles=1)
    assert status["road1"] == "GREEN"
    assert status["road2"] == "RED"

    clock.advance(controller.green_time_road1 + controller.YELLOW_TIME + 0.1)
    status = controller.update_signal_timing(road1_vehicles=1, road2_vehicles=6)

    assert controller.current_state == controller.STATE_ROAD2_GREEN
    assert status["road1"] == "RED"
    assert status["road2"] == "GREEN"
    assert status["time_remaining"] == controller.green_time_road2

    clock.advance(controller.green_time_road2 + controller.YELLOW_TIME + 0.1)
    status = controller.update_signal_timing(road1_vehicles=4, road2_vehicles=0)

    assert controller.current_state == controller.STATE_ROAD1_GREEN
    assert status["road1"] == "GREEN"
    assert status["road2"] == "RED"
    assert status["time_remaining"] == controller.green_time_road1


def test_scenario_validation_and_iteration():
    scenarios = load_predefined_scenarios()
    assert len(scenarios) >= 2

    scenario = scenarios[0]
    assert isinstance(scenario, TrafficScenario)

    steps = list(scenario.steps())
    assert steps
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in steps)


def test_scenario_requires_balanced_lengths():
    with pytest.raises(ValueError):
        TrafficScenario(
            name="invalid",
            description="mismatch",
            road1_counts=[1, 2, 3],
            road2_counts=[1, 2],
        )
