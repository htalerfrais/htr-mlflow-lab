"""Factory for creating OCR model instances."""

from __future__ import annotations

from typing import Dict, Type

from src.models.base import OCRModel
from src.models.tesseract_model import TesseractModel
from src.models.trocr_model import TrOCRModel


class ModelFactory:
    """Instantiate OCR models based on configuration names."""

    _registry: Dict[str, Type[OCRModel]] = {
        "tesseract": TesseractModel,
        "trocr": TrOCRModel,
    }

    @classmethod
    def create(cls, model_name: str) -> OCRModel:
        """Create a model instance for the provided model name using sensible defaults."""

        model_class = cls._registry.get(model_name.lower())
        if model_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        return model_class()

