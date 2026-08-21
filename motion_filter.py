"""Dwell-time motion filter: separates parked vehicles from traffic demand.

A single camera frame cannot distinguish a parked car from one queued at a
red light — both are stationary. This filter uses each vehicle's *history*
(via tracker IDs) and three rules:

1. **Dwell time** — a queued car is stationary for at most one signal cycle;
   a car stationary longer than ``parked_after`` seconds is parked.
2. **Arrival history** — a car that has been present since it first appeared
   and has *never* been seen moving is parked once ``never_moved_grace``
   seconds pass (long enough for any genuine queue to have discharged).
   A car that drove into the frame and stopped keeps counting as demand
   until rule 1 applies.
3. **Re-activation** — the moment a flagged car moves again it counts as
   demand again.

Stationarity is measured with an *anchor*: a track's anchor position resets
whenever the vehicle strays more than ``move_radius`` pixels from it, so
slow queue creep resets the clock while detection jitter does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(slots=True)
class _TrackState:
    anchor: Tuple[float, float]
    anchor_time: float
    first_seen: float
    last_seen: float
    ever_moved: bool = False


class MotionFilter:
    """Classify tracked vehicles as active demand or parked."""

    #: Upper bound on remembered orphan states (memory safety valve).
    MAX_ORPHANS = 64

    def __init__(
        self,
        *,
        move_radius: float = 12.0,
        parked_after: float = 120.0,
        never_moved_grace: float = 60.0,
        forget_after: float = 5.0,
        orphan_lifetime: float = 120.0,
    ) -> None:
        if move_radius <= 0:
            raise ValueError("move_radius must be positive")
        if parked_after <= 0 or never_moved_grace <= 0 or forget_after <= 0:
            raise ValueError("time thresholds must be positive")
        if orphan_lifetime <= 0:
            raise ValueError("orphan_lifetime must be positive")
        self.move_radius = move_radius
        self.parked_after = parked_after
        self.never_moved_grace = never_moved_grace
        self.forget_after = forget_after
        #: How long a lost never-moved track's state stays adoptable. Real
        #: trackers churn IDs on parked cars (measured dropouts up to ~90s
        #: on the bundled footage), so this must comfortably exceed those
        #: gaps for the parked flag to stay stable.
        self.orphan_lifetime = orphan_lifetime
        self._tracks: Dict[int, _TrackState] = {}
        #: (state, expiry): position-keyed histories of lost never-moved
        #: tracks, adoptable by whatever ID next appears at the same spot.
        self._orphans: list[Tuple[_TrackState, float]] = []

    def _orphan_state(self, state: _TrackState, expiry: float) -> None:
        self._orphans.append((state, expiry))
        if len(self._orphans) > self.MAX_ORPHANS:
            self._orphans = self._orphans[-self.MAX_ORPHANS:]

    def handle_discontinuity(self, now: float, adopt_window: float = 3.0) -> None:
        """Playback jumped (e.g. a looping video rewound to frame 0).

        Moving vehicles teleport across the seam, so their tracks are
        dropped — they will be re-acquired fresh and fail safe (counted as
        demand). Never-moved tracks belong to position-stable parked
        candidates; the scene is the same after the rewind, so their state
        is kept as position-keyed *orphans* that the next detection at the
        same spot re-adopts, preserving accumulated dwell history. Orphans
        from a previous seam that were not yet re-adopted are kept, not
        discarded.
        """

        for state in self._tracks.values():
            if not state.ever_moved:
                self._orphan_state(state, now + adopt_window)
        self._tracks.clear()
        self._expire_orphans(now)

    def _expire_orphans(self, now: float) -> None:
        self._orphans = [(s, e) for s, e in self._orphans if e > now]

    def _adopt_orphan(self, center: Tuple[float, float], now: float):
        """Pop and return the NEAREST unexpired orphan within move_radius."""

        best_index = None
        best_dist = None
        for index, (state, expiry) in enumerate(self._orphans):
            if expiry <= now:
                continue
            distance = math.dist(center, state.anchor)
            if distance <= self.move_radius and (best_dist is None or distance < best_dist):
                best_index, best_dist = index, distance
        if best_index is None:
            return None
        state, _ = self._orphans.pop(best_index)
        return state

    def observe(self, track_id: int, center: Tuple[float, float], now: float) -> None:
        """Record the latest position of a tracked vehicle."""

        state = self._tracks.get(track_id)
        if state is None:
            orphan = self._adopt_orphan(center, now)
            if orphan is not None:
                orphan.last_seen = now
                self._tracks[track_id] = orphan
                return
            self._tracks[track_id] = _TrackState(
                anchor=center, anchor_time=now, first_seen=now, last_seen=now
            )
            return
        state.last_seen = now
        if math.dist(center, state.anchor) > self.move_radius:
            state.anchor = center
            state.anchor_time = now
            state.ever_moved = True

    def is_parked(self, track_id: int, now: float) -> bool:
        """True when this track should be excluded from demand counts."""

        state = self._tracks.get(track_id)
        if state is None:
            return False
        if not state.ever_moved:
            return now - state.first_seen >= self.never_moved_grace
        return now - state.anchor_time >= self.parked_after

    def prune(self, now: float) -> None:
        """Forget tracks not observed recently.

        Never-moved (parked-candidate) states are not simply discarded:
        real trackers churn IDs on stationary vehicles, so the state is
        kept as a position-keyed orphan for ``orphan_lifetime`` seconds
        and re-adopted by whatever new ID appears at the same spot —
        bridging ordinary mid-stream ID changes the same way loop seams
        are bridged.
        """

        stale = [
            track_id
            for track_id, state in self._tracks.items()
            if now - state.last_seen > self.forget_after
        ]
        for track_id in stale:
            state = self._tracks.pop(track_id)
            if not state.ever_moved:
                self._orphan_state(state, now + self.orphan_lifetime)
        self._expire_orphans(now)

    def track_count(self) -> int:
        return len(self._tracks)
