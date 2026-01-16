from __future__ import annotations

from typing import Dict, Type

from src.data.importer_factory import DataImporterFactory
from src.models.factory import ModelFactory
from src.pipelines.base import Pipeline
from src.pipelines.line_to_text import LineToTextPipeline
from src.pre_processing.factory import PreprocessorFactory


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

        dataset_config = config.get("dataset")

        if isinstance(dataset_config, str):
            dataset_name = dataset_config
            dataset_params = {}
        elif isinstance(dataset_config, dict):
            dataset_name = dataset_config.get("name")
            if not isinstance(dataset_name, str):
                raise ValueError("Dataset configuration must include a 'name' field")
            dataset_params = {k: v for k, v in dataset_config.items() if k != "name"}
        else:
            raise ValueError("Configuration must include a 'dataset' name or dictionary")

        model_config = config.get("model")
        if not isinstance(model_config, dict):
            raise ValueError(
                "Configuration must include a 'model' dictionary with a 'name' field "
                "(e.g. model: {name: trocr_fr_finetuned, mlflow_run_id: ...})."
            )

        model_name = model_config.get("name")
        if not isinstance(model_name, str):
            raise ValueError("Model configuration must include a 'name' field")

        model_params = {k: v for k, v in model_config.items() if k != "name"}

        data_importer = DataImporterFactory.create(dataset_name, **(dataset_params or {}))
        model = ModelFactory.create(model_name, **model_params)

        # Create preprocessor if configured
        preprocessor_config = config.get("preprocessor")
        preprocessor = PreprocessorFactory.create(preprocessor_config)

        return pipeline_class(
            data_importer=data_importer,
            model=model,
            preprocessor=preprocessor,
        )

