"""Factory for creating data loader instances."""

from __future__ import annotations

from typing import Callable, Dict, Type

from src.data_loaders.base import DataLoader
from src.data_loaders.line_loader import IAMLineLoader


class DataLoaderFactory:
    """Instantiate data loaders based on configuration names."""

    _registry: Dict[str, Type[DataLoader]] = {}

    # Pre-register built-in loaders
    for _name in ("teklia/iam-line", "iam", "iam-line"):
        _registry[_name] = IAMLineLoader

    @classmethod
    def register(cls, name: str, loader_class: Type[DataLoader]) -> None:
        """Register a new data loader under the given name."""

        cls._registry[name.lower()] = loader_class

    @classmethod
    def create(cls, dataset_name: str, **kwargs) -> DataLoader:
        """Create a data loader instance for the provided dataset name."""

        loader_class = cls._registry.get(dataset_name.lower())
        if loader_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            )

        return loader_class(**kwargs)

