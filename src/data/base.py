"""Abstract base classes for data importers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


Sample = Tuple[str, str]
DatasetInfo = Dict[str, Any]


class DataImporter(ABC):
    """Base class for dataset importing strategies."""

    @abstractmethod
    def import_data(self, split: str = "validation") -> Tuple[List[Sample], DatasetInfo]:
        """Import dataset samples and accompanying metadata."""
