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


def test_discontinuity_preserves_parked_candidates_by_position():
    """Across a video-loop rewind, a never-moved (parked) car keeps its
    accumulated history even though the tracker assigns it a new ID."""

    mf = make_filter(never_moved_grace=60.0)

    mf.observe(1, (400.0, 300.0), 0.0)
    mf.observe(1, (400.0, 300.0), 40.0)
    assert not mf.is_parked(1, 40.0)  # grace not yet reached

    mf.handle_discontinuity(40.0)
    # After the rewind the same car re-appears at the same spot as ID 7.
    mf.observe(7, (402.0, 301.0), 41.0)
    mf.observe(7, (402.0, 301.0), 61.0)
    assert mf.is_parked(7, 61.0)  # first_seen carried across the seam


def test_discontinuity_drops_moving_tracks():
    """A car that was driving keeps NO state across the seam — whatever
    appears at its old position is a different vehicle and starts fresh."""

    mf = make_filter()

    mf.observe(2, (100.0, 100.0), 0.0)
    mf.observe(2, (200.0, 100.0), 1.0)  # moving
    mf.handle_discontinuity(1.0)

    mf.observe(9, (200.0, 100.0), 2.0)  # new car at the old position
    mf.observe(9, (200.0, 100.0), 100.0)
    # Fresh never-moved clock: parked only via its own grace, and 98s > 60s
    # means it is now (correctly) treated as never-moved parked.
    assert mf.is_parked(9, 100.0)
    # But crucially it was NOT parked shortly after the seam:
    mf2 = make_filter()
    mf2.observe(2, (100.0, 100.0), 0.0)
    mf2.observe(2, (200.0, 100.0), 1.0)
    mf2.handle_discontinuity(1.0)
    mf2.observe(9, (200.0, 100.0), 2.0)
    assert not mf2.is_parked(9, 2.0)


def test_orphans_expire_after_the_adoption_window():
    mf = make_filter()

    mf.observe(1, (400.0, 300.0), 0.0)
    mf.handle_discontinuity(0.0, adopt_window=3.0)

    # Nothing re-appears at that spot until well past the window.
    mf.observe(5, (400.0, 300.0), 10.0)
    mf.observe(5, (400.0, 300.0), 12.0)
    assert not mf.is_parked(5, 12.0)  # fresh track, no inherited history


def test_mid_stream_id_churn_bridged_for_parked_candidates():
    """Real trackers churn IDs on stationary cars (measured gaps up to ~90s
    on the bundled footage). A pruned never-moved track's history must be
    re-adopted by the new ID at the same position."""

    mf = make_filter(never_moved_grace=60.0, orphan_lifetime=120.0)

    mf.observe(1, (500.0, 200.0), 0.0)
    mf.observe(1, (500.0, 200.0), 40.0)
    mf.prune(50.0)  # track 1 unseen for 10s > forget_after -> orphaned

    # 30s later the tracker re-acquires the same car as ID 42.
    mf.observe(42, (501.0, 201.0), 80.0)
    assert mf.is_parked(42, 80.0)  # 80s since first_seen=0 >= 60s grace


def test_adoption_picks_the_nearest_orphan():
    """Two parked cars closer than move_radius must not swap histories."""

    mf = make_filter(move_radius=12.0)

    mf.observe(1, (100.0, 100.0), 0.0)   # car A
    mf.observe(2, (108.0, 100.0), 0.0)   # car B, 8px away
    mf.handle_discontinuity(10.0)

    # New detection lands 1px from B (9px from A): must adopt B's state.
    mf.observe(7, (107.0, 100.0), 10.5)
    state = mf._tracks[7]
    assert state.anchor == (108.0, 100.0)


def test_chained_discontinuities_keep_unconsumed_orphans():
    """A second seam before an orphan is re-adopted must not discard it."""

    mf = make_filter(never_moved_grace=60.0)

    mf.observe(1, (400.0, 300.0), 0.0)
    mf.observe(1, (400.0, 300.0), 40.0)
    mf.handle_discontinuity(40.0)
    # Second seam immediately after, before any re-detection.
    mf.handle_discontinuity(41.0)

    mf.observe(9, (400.0, 300.0), 42.0)
    mf.observe(9, (400.0, 300.0), 61.0)
    assert mf.is_parked(9, 61.0)  # history survived both seams


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        MotionFilter(move_radius=0.0)
    with pytest.raises(ValueError):
        MotionFilter(parked_after=-1.0)
