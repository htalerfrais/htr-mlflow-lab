from __future__ import annotations

from typing import Dict, Type

from src.data.importer_factory import DataImporterFactory
from src.models.factory import ModelFactory
from src.pipelines.base import Pipeline
from src.pipelines.line_to_text import LineToTextPipeline


class PipelineFactory:
    """Instantiate pipelines based on configuration names."""

    _registry: Dict[str, Type[Pipeline]] = {
        "line_to_text": LineToTextPipeline,
    }

    @classmethod
    def create(cls, pipeline_name: str, config: Dict[str, object]) -> Pipeline:
        """Create a pipeline instance for the provided pipeline name."""

        # check if called pipeline_name from config file is in the registery of the factory
        pipeline_class = cls._registry.get(pipeline_name.lower())
        if pipeline_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(
                f"Unknown pipeline '{pipeline_name}'. Available pipelines: {available}"
            )

        dataset_name = config.get("dataset")
        if not isinstance(dataset_name, str):
            raise ValueError("Configuration must include a 'dataset' name")

        model_name = config.get("model")
        if not isinstance(model_name, str):
            raise ValueError("Configuration must include a 'model' name")

        model_params = config.get("params")
        if model_params is not None and not isinstance(model_params, dict):
            raise ValueError("Configuration field 'params' must be a dictionary if provided")

        importer_params = config.get("importer")
        if importer_params is not None and not isinstance(importer_params, dict):
            raise ValueError("Configuration field 'importer' must be a dictionary if provided")

        data_importer = DataImporterFactory.create(dataset_name, **(importer_params or {}))
        model = ModelFactory.create(model_name, model_params)

        return pipeline_class(data_importer=data_importer, model=model)

