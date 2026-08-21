"""Tests for the dwell-time motion filter (parked-car immunity)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from motion_filter import MotionFilter


def make_filter(**overrides):
    defaults = dict(
        move_radius=12.0, parked_after=120.0, never_moved_grace=60.0, forget_after=5.0
    )
    defaults.update(overrides)
    return MotionFilter(**defaults)


def test_never_moved_car_becomes_parked_after_grace():
    mf = make_filter()

    for t in range(0, 59, 2):
        mf.observe(1, (100.0, 100.0), float(t))
        assert not mf.is_parked(1, float(t))

    mf.observe(1, (101.0, 100.0), 61.0)  # jitter, below move_radius
    assert mf.is_parked(1, 61.0)


def test_parked_car_reactivates_when_it_moves():
    mf = make_filter()

    mf.observe(1, (100.0, 100.0), 0.0)
    mf.observe(1, (100.0, 100.0), 70.0)
    assert mf.is_parked(1, 70.0)

    # The car pulls out of its spot.
    mf.observe(1, (130.0, 100.0), 71.0)
    assert not mf.is_parked(1, 71.0)


def test_car_that_arrived_moving_gets_full_dwell_budget():
    """A car that drove into the frame and stopped is queued traffic; it may
    stand far longer than the never-moved grace before being called parked."""

    mf = make_filter()

    mf.observe(2, (50.0, 100.0), 0.0)
    mf.observe(2, (150.0, 100.0), 2.0)  # clearly moving
    mf.observe(2, (150.0, 100.0), 100.0)  # stopped (e.g. at a red light)
    assert not mf.is_parked(2, 100.0)  # 98s stationary < parked_after

    mf.observe(2, (150.0, 100.0), 125.0)
    assert mf.is_parked(2, 125.0)  # 123s stationary >= parked_after


def test_queue_creep_resets_the_dwell_clock():
    mf = make_filter(parked_after=30.0)

    mf.observe(3, (100.0, 300.0), 0.0)
    mf.observe(3, (100.0, 200.0), 1.0)  # entered moving
    # Every 20s the queue advances one car length (> move_radius).
    for step in range(1, 7):
        t = 1.0 + step * 20.0
        mf.observe(3, (100.0, 200.0 - step * 20.0), t)
        assert not mf.is_parked(3, t)


def test_detection_jitter_does_not_count_as_movement():
    mf = make_filter(never_moved_grace=10.0)

    for t in range(0, 12):
        wiggle = (t % 3) - 1  # +/- 1px around the anchor
        mf.observe(4, (200.0 + wiggle, 50.0), float(t))
    assert mf.is_parked(4, 11.0)


def test_unknown_track_is_never_parked():
    mf = make_filter()
    assert not mf.is_parked(99, 1000.0)


def test_prune_forgets_lost_tracks():
    mf = make_filter()

    mf.observe(1, (10.0, 10.0), 0.0)
    mf.observe(2, (20.0, 20.0), 8.0)
    mf.prune(10.0)  # track 1 unseen for 10s > forget_after

    assert mf.track_count() == 1
    assert not mf.is_parked(1, 10.0)  # forgotten, so treated as fresh


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        MotionFilter(move_radius=0.0)
    with pytest.raises(ValueError):
        MotionFilter(parked_after=-1.0)
