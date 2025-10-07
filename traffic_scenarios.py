"""Predefined traffic scenarios for the smart traffic light simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple


@dataclass(frozen=True)
class TrafficScenario:
    """Describes a repeatable traffic scenario for two intersecting roads."""

    name: str
    description: str
    road1_counts: List[int]
    road2_counts: List[int]
    step_seconds: float = 5.0

    def __post_init__(self) -> None:  # type: ignore[override]
        if len(self.road1_counts) != len(self.road2_counts):
            raise ValueError("Both roads must have the same number of steps")
        if not self.road1_counts:
            raise ValueError("A scenario must contain at least one step")

    def steps(self) -> Iterator[Tuple[int, int]]:
        """Yield the (road1, road2) vehicle counts for each time step."""

        for count1, count2 in zip(self.road1_counts, self.road2_counts):
            yield count1, count2


def load_predefined_scenarios() -> List[TrafficScenario]:
    """Return curated scenarios that cover common traffic patterns."""

    heavy_vs_clear = TrafficScenario(
        name="Heavy arterial vs. empty side road",
        description=(
            "Simulates a heavy flow on the main road with occasional vehicles on the "
            "secondary road. Mirrors the behavior in road1.mp4 (heavy) and "
            "road2.mp4 (light)."
        ),
        road1_counts=[8, 9, 10, 7, 6, 5, 4, 3, 2, 2],
        road2_counts=[0, 1, 0, 0, 1, 1, 2, 1, 0, 0],
    )

    alternating_surges = TrafficScenario(
        name="Alternating surges",
        description=(
            "Traffic alternates between the two roads, useful for validating "
            "green light switching logic."
        ),
        road1_counts=[0, 2, 6, 1, 0, 5, 7, 2],
        road2_counts=[7, 6, 1, 0, 4, 0, 1, 5],
    )

    return [heavy_vs_clear, alternating_surges]
