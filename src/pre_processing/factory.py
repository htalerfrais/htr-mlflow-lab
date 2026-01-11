"""Factory for creating image preprocessors."""

from __future__ import annotations

from typing import Dict, Type, List, Union, Optional

from src.pre_processing.base import ImagePreprocessor
from src.pre_processing.binarize import BinarizePreprocessor
from src.pre_processing.identity import IdentityPreprocessor
from src.pre_processing.resize import ResizePreprocessor
from src.pre_processing.sequential import SequentialPreprocessor


class PreprocessorFactory:
    """Instantiate image preprocessors based on configuration."""

    _registry: Dict[str, Type[ImagePreprocessor]] = {
        "identity": IdentityPreprocessor,
        "resize": ResizePreprocessor,
        "binarize": BinarizePreprocessor,
        # Future preprocessors can be added here:
        # "crop": CropPreprocessor,
    }

    @classmethod
    def create(
        cls,
        preprocessor_config: Optional[Union[str, Dict, List[Union[str, Dict]]]] = None,
    ) -> Optional[ImagePreprocessor]:
        """
        Create a preprocessor instance.

        Args:
            preprocessor_config:
                - None: no preprocessor (returns None)
                - str: name of a single preprocessor ("identity", "resize")
                - Dict: config dict with "name" and optional params ({"name": "resize", "height": 128})
                - List: list of names or config dicts for SequentialPreprocessor
                       (["identity", {"name": "resize", "height": 128}])

        Returns:
            ImagePreprocessor or None if preprocessor_config is None

        Raises:
            ValueError: If preprocessor_config has an invalid type or contains
                       unknown preprocessor names
        """
        # Case 1: No preprocessor
        if preprocessor_config is None:
            return None

        # Case 2: List of preprocessors -> SequentialPreprocessor
        if isinstance(preprocessor_config, list):
            if not preprocessor_config:
                return None  # Empty list = no preprocessor

            # Create each individual preprocessor
            preprocessors = [
                cls._create_single(config_item) for config_item in preprocessor_config
            ]

            # If only one preprocessor in the list, return it directly
            if len(preprocessors) == 1:
                return preprocessors[0]

            # Otherwise, create a SequentialPreprocessor
            return SequentialPreprocessor(preprocessors)

        # Case 3: Dict config -> single preprocessor with params
        if isinstance(preprocessor_config, dict):
            return cls._create_single(preprocessor_config)

        # Case 4: Simple string -> single preprocessor
        if isinstance(preprocessor_config, str):
            return cls._create_single(preprocessor_config)

        # Invalid case
        raise ValueError(
            f"Invalid preprocessor config: {preprocessor_config}. "
            f"Expected None, str, Dict, or List[Union[str, Dict]]"
        )

    @classmethod
    def _create_single(cls, config: Union[str, Dict]) -> ImagePreprocessor:
        """
        Create a single preprocessor from its name or config dict.

        Args:
            config: Either a string name or a dict with "name" and optional params

        Returns:
            ImagePreprocessor: An instance of the requested preprocessor

        Raises:
            ValueError: If the preprocessor name is not in the registry
        """
        # Handle dict config
        if isinstance(config, dict):
            preprocessor_name = config.get("name")
            if not preprocessor_name:
                raise ValueError("Preprocessor config dict must contain 'name' key")
            params = {k: v for k, v in config.items() if k != "name"}
        else:
            # Handle string config (backward compatible)
            preprocessor_name = config
            params = {}

        preprocessor_class = cls._registry.get(preprocessor_name.lower())

        if preprocessor_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(
                f"Unknown preprocessor '{preprocessor_name}'. "
                f"Available preprocessors: {available}"
            )

        # Instantiate with params if provided, otherwise use defaults
        return preprocessor_class(**params)

