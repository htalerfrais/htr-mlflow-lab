"""Factory for creating image preprocessors."""

from __future__ import annotations

from typing import Dict, Type, List, Union, Optional

from src.pre_processing.base import ImagePreprocessor
from src.pre_processing.identity import IdentityPreprocessor
from src.pre_processing.sequential import SequentialPreprocessor


class PreprocessorFactory:
    """Instantiate image preprocessors based on configuration."""

    _registry: Dict[str, Type[ImagePreprocessor]] = {
        "identity": IdentityPreprocessor,
        # Future preprocessors can be added here:
        # "binary": BinaryPreprocessor,
        # "crop": CropPreprocessor,
    }

    @classmethod
    def create(
        cls,
        preprocessor_config: Optional[Union[str, List[str]]] = None,
    ) -> Optional[ImagePreprocessor]:
        """
        Create a preprocessor instance.

        Args:
            preprocessor_config:
                - None: no preprocessor (returns None)
                - str: name of a single preprocessor ("identity", "binary")
                - List[str]: list of names for a SequentialPreprocessor
                            (["identity", "binary"])

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
                cls._create_single(name) for name in preprocessor_config
            ]

            # If only one preprocessor in the list, return it directly
            if len(preprocessors) == 1:
                return preprocessors[0]

            # Otherwise, create a SequentialPreprocessor
            return SequentialPreprocessor(preprocessors)

        # Case 3: Simple string -> single preprocessor
        if isinstance(preprocessor_config, str):
            return cls._create_single(preprocessor_config)

        # Invalid case
        raise ValueError(
            f"Invalid preprocessor config: {preprocessor_config}. "
            f"Expected None, str, or List[str]"
        )

    @classmethod
    def _create_single(cls, preprocessor_name: str) -> ImagePreprocessor:
        """
        Create a single preprocessor from its name.

        Args:
            preprocessor_name: Name of the preprocessor to create

        Returns:
            ImagePreprocessor: An instance of the requested preprocessor

        Raises:
            ValueError: If the preprocessor name is not in the registry
        """
        preprocessor_class = cls._registry.get(preprocessor_name.lower())

        if preprocessor_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(
                f"Unknown preprocessor '{preprocessor_name}'. "
                f"Available preprocessors: {available}"
            )

        return preprocessor_class()

