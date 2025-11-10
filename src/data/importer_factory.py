"""Factory for creating data importer instances."""

from __future__ import annotations

from typing import Dict, Type

from src.data.base import DataImporter
from src.data.line_importer import IAMLineImporter
from src.data.local_importer import LocalLineImporter


class DataImporterFactory:
    """Instantiate data importers based on configuration names."""

    _registry: Dict[str, Type[DataImporter]] = {}

    # Pre-register built-in importers
    for _name in ("teklia/iam-line", "iam", "iam-line"):
        _registry[_name] = IAMLineImporter
    for _name in ("local_lines", "local-line-dataset"):
        _registry[_name] = LocalLineImporter

    @classmethod
    def register(cls, name: str, importer_class: Type[DataImporter]) -> None:
        """Register a new data importer under the given name."""

        cls._registry[name.lower()] = importer_class

    @classmethod
    def create(cls, dataset_name: str, **kwargs) -> DataImporter:
        """Create a data importer instance for the provided dataset name."""

        importer_class = cls._registry.get(dataset_name.lower())
        if importer_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            )

        return importer_class(**kwargs)

