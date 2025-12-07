"""Abstract base classes for image preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from PIL import Image


ImageInput = Union[str, Image.Image]


class ImagePreprocessor(ABC):
    """Base class for image preprocessing operations."""

    @abstractmethod
    def preprocess(self, image: ImageInput) -> Image.Image:
        """
        Apply preprocessing to the provided image input.
        
        Args:
            image: Either a path to an image file or a PIL Image object
            
        Returns:
            PIL.Image: The preprocessed image
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable preprocessor name."""

