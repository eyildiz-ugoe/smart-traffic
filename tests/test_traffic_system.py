from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

try:  # pragma: no cover - optional test dependency
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - gracefully handle headless environments
    cv2 = None
    np = None

from traffic_core import TrafficLightController, TrafficStats
from traffic_scenarios import TrafficScenario, load_predefined_scenarios
from smart_traffic_system import (
    SimulationTrafficSystem,
    VehicleCounter,
    VehicleDetection,
    VehicleQueueAnalyzer,
    VehicleSorter,
    resolve_video_sources,
)


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


def test_queue_analyzer_ranks_vertical_detections_by_distance():
    analyzer = VehicleQueueAnalyzer(orientation="vertical")
    detections = [
        VehicleDetection((10, 260, 40, 80), confidence=0.75, class_id=2),  # bottom = 340
        VehicleDetection((30, 150, 35, 40), confidence=0.90, class_id=2),  # bottom = 190
        VehicleDetection((50, 200, 30, 80), confidence=0.80, class_id=2),  # bottom = 280
    ]

    metrics = analyzer.calculate_metrics((480, 640, 3), detections)

    assert metrics.count == 3
    assert [det.bbox for det in metrics.sorted_detections] == [
        (10, 260, 40, 80),
        (50, 200, 30, 80),
        (30, 150, 35, 40),
    ]
    assert metrics.class_breakdown == {2: 3}
    assert metrics.pressure > metrics.count  # distance weighting applied
    assert metrics.stopline_occupied is True
    assert metrics.exit_zone_active is False
    assert metrics.leading_edge == 340


def test_queue_analyzer_horizontal_orientation():
    analyzer = VehicleQueueAnalyzer(orientation="horizontal")
    detections = [
        VehicleDetection((100, 50, 40, 40), confidence=0.60, class_id=2),  # right = 140
        VehicleDetection((200, 60, 30, 40), confidence=0.70, class_id=2),  # right = 230
        VehicleDetection((50, 70, 20, 20), confidence=0.90, class_id=2),   # right = 70
    ]

    metrics = analyzer.calculate_metrics((480, 640, 3), detections)

    assert [det.bbox for det in metrics.sorted_detections] == [
        (200, 60, 30, 40),
        (100, 50, 40, 40),
        (50, 70, 20, 20),
    ]
    assert metrics.class_breakdown == {2: 3}
    assert not metrics.stopline_occupied
    assert not metrics.exit_zone_active


def test_queue_analyzer_reports_threshold_lines():
    analyzer = VehicleQueueAnalyzer(orientation="vertical", approach_threshold_ratio=0.5, exit_margin=10)
    detections = [
        VehicleDetection((0, 200, 20, 50), confidence=0.8, class_id=2),  # bottom = 250
    ]

    metrics = analyzer.calculate_metrics((400, 640, 3), detections)

    assert metrics.approach_line == 200
    assert metrics.exit_line == 389
    assert metrics.stopline_occupied
    assert not metrics.exit_zone_active

    # detection reaching exit zone
    detections.append(VehicleDetection((0, 360, 20, 40), confidence=0.7, class_id=2))
    metrics = analyzer.calculate_metrics((400, 640, 3), detections)
    assert metrics.exit_zone_active


def test_controller_threshold_priority_and_switching():
    clock = FakeClock()
    controller = TrafficLightController(time_func=clock)

    status = controller.update_signal_timing(
        road1_vehicles=1,
        road2_vehicles=0,
        road1_queue_pressure=2.0,
        road2_queue_pressure=0.0,
        road1_stopline_occupied=True,
        road2_stopline_occupied=False,
        road1_exit_ready=False,
        road2_exit_ready=True,
    )

    assert status["road1"] == "GREEN"
    assert status["road2"] == "RED"

    # Road2 requests but road1 has priority and is not yet clear
    status = controller.update_signal_timing(
        road1_vehicles=1,
        road2_vehicles=1,
        road1_queue_pressure=3.0,
        road2_queue_pressure=3.0,
        road1_stopline_occupied=True,
        road2_stopline_occupied=True,
        road1_exit_ready=False,
        road2_exit_ready=False,
    )

    assert status["road1"] == "GREEN"
    assert controller.current_state == controller.STATE_ROAD1_GREEN

    # Once road1 clears and road2 still requests, switch to road2
    # Advance time to ensure min green time has passed
    clock.advance(controller.MIN_GREEN_TIME)
    
    status = controller.update_signal_timing(
        road1_vehicles=0,
        road2_vehicles=1,
        road1_queue_pressure=0.0,
        road2_queue_pressure=3.0,
        road1_stopline_occupied=False,
        road2_stopline_occupied=True,
        road1_exit_ready=True,
        road2_exit_ready=False,
    )

    # Expect Yellow first
    assert status["road1"] == "YELLOW"
    assert status["road2"] == "RED"
    
    # Advance through yellow
    clock.advance(controller.YELLOW_TIME + 0.1)
    
    status = controller.update_signal_timing(
        road1_vehicles=0,
        road2_vehicles=1,
        road1_queue_pressure=0.0,
        road2_queue_pressure=3.0,
        road1_stopline_occupied=False,
        road2_stopline_occupied=True,
        road1_exit_ready=True,
        road2_exit_ready=False,
    )

    assert controller.current_state == controller.STATE_ROAD2_GREEN
    assert status["road2"] == "GREEN"

    # Road1 requests while road2 is active; road1 has priority but must wait until road2 clears
    # Road 2 just started green, so it should stay green for min time
    status = controller.update_signal_timing(
        road1_vehicles=1,
        road2_vehicles=1,
        road1_queue_pressure=3.5,
        road2_queue_pressure=3.5,
        road1_stopline_occupied=True,
        road2_stopline_occupied=True,
        road1_exit_ready=False,
        road2_exit_ready=False,
    )

    assert controller.current_state == controller.STATE_ROAD2_GREEN
    assert status["road2"] == "GREEN"

    # After road2 clears, priority should give road1 the green light
    clock.advance(controller.MIN_GREEN_TIME)
    
    status = controller.update_signal_timing(
        road1_vehicles=1,
        road2_vehicles=0,
        road1_queue_pressure=3.5,
        road2_queue_pressure=0.0,
        road1_stopline_occupied=True,
        road2_stopline_occupied=False,
        road1_exit_ready=False,
        road2_exit_ready=True,
    )
    
    # Expect Yellow
    assert status["road1"] == "RED"
    assert status["road2"] == "YELLOW"
    
    clock.advance(controller.YELLOW_TIME + 0.1)
    
    status = controller.update_signal_timing(
        road1_vehicles=1,
        road2_vehicles=0,
        road1_queue_pressure=3.5,
        road2_queue_pressure=0.0,
        road1_stopline_occupied=True,
        road2_stopline_occupied=False,
        road1_exit_ready=False,
        road2_exit_ready=True,
    )

    assert controller.current_state == controller.STATE_ROAD1_GREEN
    assert status["road1"] == "GREEN"


def test_controller_holds_secondary_road_until_main_is_clear():
    clock = FakeClock()
    controller = TrafficLightController(time_func=clock)

    status = controller.update_signal_timing(
        road1_vehicles=1,
        road2_vehicles=0,
        road1_queue_pressure=2.5,
        road2_queue_pressure=0.0,
        road1_stopline_occupied=False,
        road2_stopline_occupied=False,
        road1_exit_ready=False,
        road2_exit_ready=True,
        road1_leading_edge=180,
        road1_approach_line=200,
    )

    assert controller.current_state == controller.STATE_ROAD1_GREEN
    assert status["road1"] == "GREEN"
    assert status["road2"] == "RED"

    status = controller.update_signal_timing(
        road1_vehicles=1,
        road2_vehicles=1,
        road1_queue_pressure=3.0,
        road2_queue_pressure=3.2,
        road1_stopline_occupied=False,
        road2_stopline_occupied=True,
        road1_exit_ready=False,
        road2_exit_ready=False,
        road1_leading_edge=195,
        road1_approach_line=200,
    )

    assert controller.current_state == controller.STATE_ROAD1_GREEN
    assert status["road1"] == "GREEN"
    assert status["road2"] == "RED"

    clock.advance(controller.MIN_GREEN_TIME)

    status = controller.update_signal_timing(
        road1_vehicles=0,
        road2_vehicles=1,
        road1_queue_pressure=0.0,
        road2_queue_pressure=3.2,
        road1_stopline_occupied=False,
        road2_stopline_occupied=True,
        road1_exit_ready=True,
        road2_exit_ready=False,
        road1_leading_edge=None,
        road1_approach_line=200,
    )
    
    # Expect Yellow
    assert status["road1"] == "YELLOW"
    
    clock.advance(controller.YELLOW_TIME + 0.1)
    
    status = controller.update_signal_timing(
        road1_vehicles=0,
        road2_vehicles=1,
        road1_queue_pressure=0.0,
        road2_queue_pressure=3.2,
        road1_stopline_occupied=False,
        road2_stopline_occupied=True,
        road1_exit_ready=True,
        road2_exit_ready=False,
        road1_leading_edge=None,
        road1_approach_line=200,
    )

    assert controller.current_state == controller.STATE_ROAD2_GREEN
    assert status["road1"] == "RED"
    assert status["road2"] == "GREEN"


def test_queue_pressure_extends_green_time():
    clock = FakeClock()
    controller = TrafficLightController(time_func=clock)

    controller.update_signal_timing(
        road1_vehicles=2,
        road2_vehicles=2,
        road1_queue_pressure=6.0,
        road2_queue_pressure=2.5,
    )

    assert controller.green_time_road1 > controller.green_time_road2


def test_vehicle_counter_tracks_class_breakdown():
    counter = VehicleCounter()
    detections = [
        VehicleDetection((0, 0, 10, 10), confidence=0.9, class_id=2),
        VehicleDetection((0, 0, 10, 10), confidence=0.8, class_id=2),
        VehicleDetection((0, 0, 10, 10), confidence=0.7, class_id=5),
    ]

    frame_total, breakdown = counter.summarize(detections)

    assert frame_total == 3
    assert breakdown == {2: 2, 5: 1}
    assert counter.total_detected == 3
    assert counter.totals_per_class[2] == 2


def test_vehicle_sorter_respects_orientation():
    horizontal = VehicleSorter(orientation="horizontal")
    detections = [
        VehicleDetection((0, 0, 10, 10), confidence=0.9, class_id=2),
        VehicleDetection((20, 0, 10, 10), confidence=0.9, class_id=2),
        VehicleDetection((15, 0, 10, 10), confidence=0.9, class_id=2),
    ]

    sorted_detections = horizontal.sort(detections)

    assert [det.bbox for det in sorted_detections] == [
        (20, 0, 10, 10),
        (15, 0, 10, 10),
        (0, 0, 10, 10),
    ]


def test_resolve_video_sources_uses_setup_helper_when_missing(tmp_path):
    created_dir = {}

    class DummySetup:
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            created_dir["path"] = self.output_dir

        def verify_setup(self) -> bool:
            return False

        def create_test_videos(self) -> bool:
            for name in ("road1.mp4", "road2.mp4"):
                (self.output_dir / name).write_text("stub")
            return True

    def factory(output_dir):
        return DummySetup(output_dir)

    candidate_pairs = [(tmp_path / "road1.mp4", tmp_path / "road2.mp4")]

    path1, path2 = resolve_video_sources(candidate_pairs=candidate_pairs, setup_factory=factory)

    assert Path(path1).exists()
    assert Path(path2).exists()
    assert created_dir["path"] == tmp_path


@pytest.mark.skipif(cv2 is None or np is None, reason="simulation requires cv2 and numpy")
def test_simulation_mode_generates_metrics_without_display():
    simulation = SimulationTrafficSystem(fps=20, seed=42)
    simulation.run(max_frames=60, display_window=False)

    assert simulation.stats_road1.total_frames == 60
    assert simulation.stats_road2.total_frames == 60
    assert simulation.last_metrics_road1 is not None
    assert simulation.last_metrics_road2 is not None
    assert simulation.stats_road1.avg_vehicles_per_frame >= 0.0
    assert simulation.stats_road2.avg_vehicles_per_frame >= 0.0
