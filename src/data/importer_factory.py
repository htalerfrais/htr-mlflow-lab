from __future__ import annotations

from typing import Dict, Type

from src.data.base import DataImporter
from src.data.line_importer import IAMLineImporter
from src.data.local_importer import LocalLineImporter
from src.data.rimes_importer import RIMESLineImporter


class DataImporterFactory:
    """Instantiate data importers based on configuration names."""

    _registry: Dict[str, Type[DataImporter]] = {
        "teklia/iam-line": IAMLineImporter,
        "teklia/rimes-line": RIMESLineImporter,
        "local_lines": LocalLineImporter,
    }

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

