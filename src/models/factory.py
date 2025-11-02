"""Factory for creating OCR model instances."""

from __future__ import annotations

from typing import Dict, Type

from src.models.base import OCRModel
from src.models.tesseract_model import TesseractModel


class ModelFactory:
    """Instantiate OCR models based on configuration names."""

    _registry: Dict[str, Type[OCRModel]] = {}

    for _name in ("tesseract", "tesseract-ocr"):
        _registry[_name] = TesseractModel

    @classmethod
    def register(cls, name: str, model_class: Type[OCRModel]) -> None:
        """Register a new model under the given name."""

        cls._registry[name.lower()] = model_class

    @classmethod
    def create(cls, model_name: str, params: Dict[str, object] | None = None) -> OCRModel:
        """Create a model instance for the provided model name."""

        model_class = cls._registry.get(model_name.lower())
        if model_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        params = params or {}

        if model_class is TesseractModel:
            return TesseractModel(
                lang=params.get("tesseract_language", "eng"),
                engine_mode=params.get("tesseract_engine_mode", 3),
            )

        # Default instantiation for models that accept **params
        return model_class(**params)

