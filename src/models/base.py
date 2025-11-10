"""Abstract base classes for OCR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from PIL import Image


ImageInput = Union[str, Image.Image]


class OCRModel(ABC):
    """Base class for optical character recognition models."""

    @abstractmethod
    def predict(self, image: ImageInput) -> str:
        """Run inference on the provided image input."""

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable model name."""


# plus tard des méthodes de training ici par exemple