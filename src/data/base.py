"""Abstract base classes for data loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


Sample = Tuple[str, str]
DatasetInfo = Dict[str, Any]


class DataLoader(ABC):
    """Base class for dataset loading strategies."""

    @abstractmethod
    def load_data(self, split: str = "validation") -> Tuple[List[Sample], DatasetInfo]:
        """Load dataset samples and accompanying metadata."""

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for the data loader."""

