"""Test the smart traffic system with visual animation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from smart_traffic_system import SmartTrafficSystem, SimulationTrafficSystem, resolve_video_sources


@pytest.mark.skipif(cv2 is None or np is None, reason="Requires cv2 and numpy")
def test_simulation_mode():
    """Test the simulation mode without video files."""
    try:
        # Create simulation with fixed seed for reproducibility
        simulation = SimulationTrafficSystem(
            fps=30,
            seed=42,
            spawn_rate=3.5
        )
        
        # Run for limited frames without display
        max_test_frames = 60
        simulation.run(max_frames=max_test_frames, display_window=False)
        
        # Verify simulation ran
        assert simulation.stats_road1.total_frames == max_test_frames
        assert simulation.stats_road2.total_frames == max_test_frames
        
        # Verify metrics were captured
        assert simulation.last_metrics_road1 is not None
        assert simulation.last_metrics_road2 is not None
        
        # Verify vehicles were spawned (check average is positive)
        avg_vehicles = (
            simulation.stats_road1.avg_vehicles_per_frame + 
            simulation.stats_road2.avg_vehicles_per_frame
        )
        assert avg_vehicles > 0, "No vehicles were spawned in simulation"
        
        print(f"\n✓ Simulation completed successfully")
        print(f"  • Processed {max_test_frames} frames")
        print(f"  • Road1: avg vehicles={simulation.stats_road1.avg_vehicles_per_frame:.2f}")
        print(f"  • Road2: avg vehicles={simulation.stats_road2.avg_vehicles_per_frame:.2f}")
        print(f"  • Combined avg: {avg_vehicles:.2f} vehicles/frame")
        
    except Exception as e:
        pytest.fail(f"Simulation test failed: {e}")


@pytest.mark.skipif(cv2 is None or np is None, reason="Requires cv2 and numpy")
def test_system_runs_with_real_videos():
    """Test that the system can run with real video files."""
    try:
        video_road1, video_road2 = resolve_video_sources()
        
        # Verify files exist
        assert Path(video_road1).exists(), f"Video file not found: {video_road1}"
        assert Path(video_road2).exists(), f"Video file not found: {video_road2}"
        
        # Initialize system
        system = SmartTrafficSystem(
            video_road1,
            video_road2,
            orientation_road1="vertical",
            orientation_road2="vertical"
        )
        
        # Run for a limited number of frames
        frame_count = 0
        max_frames = 90  # Process 90 frames (3 seconds at 30fps)
        
        while frame_count < max_frames:
            ret1, frame1 = system.cap_road1.read()
            ret2, frame2 = system.cap_road2.read()
            
            if not ret1 or not ret2:
                break
            
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))
            
            metrics1, frame1 = system._process_road(frame1, system.queue_analyzer_road1)
            metrics2, frame2 = system._process_road(frame2, system.queue_analyzer_road2)
            
            system.stats_road1.update(metrics1.count)
            system.stats_road2.update(metrics2.count)
            
            signal_status = system.controller.update_signal_timing(
                metrics1.count,
                metrics2.count,
                road1_queue_pressure=metrics1.pressure,
                road2_queue_pressure=metrics2.pressure,
                road1_stopline_occupied=metrics1.stopline_occupied,
                road2_stopline_occupied=metrics2.stopline_occupied,
                road1_exit_ready=metrics1.exit_zone_active or metrics1.count == 0,
                road2_exit_ready=metrics2.exit_zone_active or metrics2.count == 0,
                road1_leading_edge=metrics1.leading_edge,
                road2_leading_edge=metrics2.leading_edge,
                road1_approach_line=metrics1.approach_line,
                road2_approach_line=metrics2.approach_line,
            )
            
            frame_count += 1
        
        # Verify we processed some frames
        assert frame_count > 0, "No frames were processed"
        assert system.stats_road1.total_frames == frame_count
        assert system.stats_road2.total_frames == frame_count
        
        # Cleanup
        system.cap_road1.release()
        system.cap_road2.release()
        
        print(f"\n✓ Successfully processed {frame_count} frames")
        print(f"  Road1: avg vehicles={system.stats_road1.avg_vehicles_per_frame:.2f}")
        print(f"  Road2: avg vehicles={system.stats_road2.avg_vehicles_per_frame:.2f}")
        
    except Exception as e:
        pytest.fail(f"System failed to run: {e}")


@pytest.mark.skipif(cv2 is None or np is None, reason="Requires cv2 and numpy")
def test_system_video_loop():
    """Test that the system can loop videos and restart properly."""
    try:
        video_road1, video_road2 = resolve_video_sources()
        
        system = SmartTrafficSystem(
            video_road1,
            video_road2,
            orientation_road1="vertical",
            orientation_road2="vertical"
        )
        
        # Get total frames in the videos
        total_frames_road1 = int(system.cap_road1.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_road2 = int(system.cap_road2.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Skip to near the end of the video to test restart quickly
        skip_to_frame = max(total_frames_road1 - 10, 0)
        system.cap_road1.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
        system.cap_road2.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
        
        # Process frames to trigger the restart logic
        max_frames = 30  # Process 30 frames (should cross the end and restart)
        frame_count = 0
        restart_triggered = False
        
        while frame_count < max_frames:
            ret1, frame1 = system.cap_road1.read()
            ret2, frame2 = system.cap_road2.read()
            
            # Test restart logic
            if not ret1 or not ret2:
                restart_triggered = True
                system.cap_road1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                system.cap_road2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = system.cap_road1.read()
                ret2, frame2 = system.cap_road2.read()
            
            if not ret1 or not ret2:
                break
            
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))
            
            metrics1, _ = system._process_road(frame1, system.queue_analyzer_road1)
            metrics2, _ = system._process_road(frame2, system.queue_analyzer_road2)
            
            system.stats_road1.update(metrics1.count)
            system.stats_road2.update(metrics2.count)
            
            frame_count += 1
        
        # Verify restart was triggered
        assert restart_triggered, "Video restart logic was not triggered"
        assert frame_count == max_frames, f"Expected {max_frames} frames, got {frame_count}"
        
        # Cleanup
        system.cap_road1.release()
        system.cap_road2.release()
        
        print(f"\n✓ Successfully tested video looping ({frame_count} frames processed)")
        print(f"  Video restart was triggered successfully")
        
    except Exception as e:
        pytest.fail(f"Video looping test failed: {e}")


@pytest.mark.skipif(cv2 is None or np is None, reason="Requires cv2 and numpy")
def test_visualization_components():
    """Test vehicle detection and annotation pipeline."""
    try:
        video_road1, video_road2 = resolve_video_sources()
        
        system = SmartTrafficSystem(
            video_road1,
            video_road2,
            orientation_road1="vertical",
            orientation_road2="vertical"
        )
        
        # Read one frame
        ret1, frame1 = system.cap_road1.read()
        assert ret1, "Failed to read frame from road1"
        
        frame1 = cv2.resize(frame1, (640, 480))
        original_shape = frame1.shape
        
        # Test vehicle detection and annotation pipeline
        metrics1, annotated_frame = system._process_road(frame1.copy(), system.queue_analyzer_road1)
        
        # Verify the annotated frame has the same shape
        assert annotated_frame.shape == original_shape
        
        # Verify metrics are calculated
        assert metrics1 is not None
        assert hasattr(metrics1, 'count')
        assert hasattr(metrics1, 'pressure')
        assert hasattr(metrics1, 'sorted_detections')
        
        # Verify queue metrics make sense
        assert metrics1.count >= 0
        assert metrics1.pressure >= 0
        assert len(metrics1.sorted_detections) == metrics1.count
        
        # Cleanup
        system.cap_road1.release()
        system.cap_road2.release()
        
        print(f"\n✓ Visualization test completed successfully")
        print(f"  Detected {metrics1.count} vehicles")
        print(f"  Queue pressure: {metrics1.pressure:.2f}")
        
    except Exception as e:
        pytest.fail(f"Visualization test failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
