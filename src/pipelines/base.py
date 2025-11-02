"""Abstract base classes for pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class Pipeline(ABC):
    """Base class for experiment pipelines."""

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the pipeline and return metrics and metadata."""

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for the pipeline."""

