"""Sequential preprocessor that chains multiple preprocessors."""

from __future__ import annotations

from typing import List
from PIL import Image

from src.pre_processing.base import ImagePreprocessor, ImageInput


class SequentialPreprocessor(ImagePreprocessor):
    """Applies multiple preprocessors in sequence."""

    def __init__(self, preprocessors: List[ImagePreprocessor]) -> None:
        """
        Initialize a sequential preprocessor.
        
        Args:
            preprocessors: Ordered list of preprocessors to apply
            
        Raises:
            ValueError: If the preprocessors list is empty
        """
        if not preprocessors:
            raise ValueError("SequentialPreprocessor requires at least one preprocessor")
        self._preprocessors = preprocessors

    def preprocess(self, image: ImageInput) -> Image.Image:
        """
        Apply each preprocessor in sequence.
        
        Args:
            image: Either a path to an image file or a PIL Image object
            
        Returns:
            PIL.Image: The final preprocessed image after all transformations
        """
        # Convert to PIL Image if path is provided
        result = Image.open(image).convert("RGB") if isinstance(image, str) else image

        # Apply each preprocessor in order
        for preprocessor in self._preprocessors:
            result = preprocessor.preprocess(result)

        return result

    def get_name(self) -> str:
        """Return a human-readable preprocessor name."""
        names = [p.get_name() for p in self._preprocessors]
        return f"Sequential({', '.join(names)})"

