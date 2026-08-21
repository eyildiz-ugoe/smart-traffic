"""Tests for trajectory-based auto-calibration (pure analysis functions)."""

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from auto_calibrate import (
    Track,
    classify_approach,
    derive_zone,
    dwell_points,
    zone_iou,
)

DIAG = math.hypot(640, 480)


def make_track(start, end, n=20, t0=0.0, dt=0.1):
    x0, y0 = start
    x1, y1 = end
    return Track(points=[
        (t0 + i * dt, x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1))
        for i in range(n)
    ])


def test_classify_approach_by_travel_direction():
    # Moving down the frame => came from the North, and so on.
    assert classify_approach(make_track((320, 20), (320, 460)), DIAG) == "N"
    assert classify_approach(make_track((320, 460), (320, 20)), DIAG) == "S"
    assert classify_approach(make_track((620, 240), (20, 240)), DIAG) == "E"
    assert classify_approach(make_track((20, 240), (620, 240)), DIAG) == "W"


def test_turning_vehicle_keeps_its_entry_approach():
    """Regression from the Sherbrooke run: a car that arrives from the top
    and then turns right must classify as North, even though its NET
    displacement points sideways."""

    points = []
    t = 0.0
    for y in range(0, 240, 12):            # enters from the top, heading down
        points.append((t, 320.0, float(y)))
        t += 0.1
    for x in range(320, 640, 16):          # then turns and exits right
        points.append((t, float(x), 240.0))
        t += 0.1
    curved = Track(points=points)
    # Net displacement is dominated by the horizontal exit leg...
    dx, dy = curved.direction()
    assert abs(dx) > abs(dy)
    # ...but the approach classification follows the entry direction.
    assert classify_approach(curved, DIAG) == "N"


def test_parked_and_short_tracks_are_rejected():
    parked = make_track((100, 100), (102, 101))  # jitter only
    assert classify_approach(parked, DIAG) is None

    short = Track(points=[(0.0, 10.0, 10.0), (0.1, 400.0, 400.0)])  # 2 points
    assert classify_approach(short, DIAG) is None


def test_derive_zone_covers_upstream_portion():
    # Ten southbound tracks entering at the top of a 640x480 frame.
    tracks = [make_track((300 + i * 4, 0), (300 + i * 4, 480), n=40) for i in range(10)]
    zone = derive_zone(tracks, (480, 640))
    assert zone is not None
    fx, fy, fw, fh = zone
    # Upstream = the top part of the frame; the zone must sit there.
    assert fy < 0.15
    assert fy + fh < 0.6
    # Horizontally centred on the lane cluster.
    assert 0.4 < fx + fw / 2 < 0.6
    # And it is a valid fractional rectangle.
    assert 0 <= fx and fx + fw <= 1 and 0 <= fy and fy + fh <= 1


def test_derive_zone_needs_enough_data():
    assert derive_zone([], (480, 640)) is None
    assert derive_zone([Track(points=[(0, 1, 1)])], (480, 640)) is None


def test_dwell_points_mark_queueing_locations():
    # A track that drives, stops for a while at y=200, then continues.
    points = []
    t = 0.0
    for y in range(0, 200, 20):        # driving
        points.append((t, 320.0, float(y)))
        t += 0.1
    for _ in range(15):                # queued at the line
        points.append((t, 320.0, 200.0))
        t += 0.1
    for y in range(200, 480, 20):      # proceeds on green
        points.append((t, 320.0, float(y)))
        t += 0.1
    dwells = dwell_points([Track(points=points)], DIAG)
    assert len(dwells) >= 10
    assert all(abs(y - 200.0) < 1e-6 for _, y in dwells)

    # A free-flowing track produces no dwell points.
    flowing = make_track((320, 0), (320, 480), n=30)
    assert dwell_points([flowing], DIAG) == []


def test_zone_iou_basics():
    a = (0.1, 0.1, 0.4, 0.4)
    assert zone_iou(a, a) == pytest.approx(1.0)
    assert zone_iou(a, (0.6, 0.6, 0.3, 0.3)) == 0.0
    half = (0.1, 0.1, 0.2, 0.4)  # left half of a
    assert zone_iou(a, half) == pytest.approx(0.5)
