"""Visualization strategy abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class VisualizationStrategy(ABC):
    """Render the current state of the traffic system."""

    @abstractmethod
    def render(self, context: Dict[str, object]) -> None:
        """Render the traffic system using the provided context."""

    @abstractmethod
    def close(self) -> None:
        """Dispose of any resources such as windows or surfaces."""
