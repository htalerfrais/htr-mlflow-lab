from __future__ import annotations

from typing import Dict, Type

from src.models.base import OCRModel
from src.models.onnx_model import ONNXModel
from src.models.tesseract_model import TesseractModel
from src.models.trocr_model import TrOCRModel
from src.models.crnn_model import CRNNModel
from src.models.crnn_f_model import CRNNFModel


class ModelFactory:
    """Instantiate OCR models based on configuration names."""

    _registry: Dict[str, Type[OCRModel]] = {
        "tesseract": TesseractModel,
        "trocr": TrOCRModel,
        "onnx": ONNXModel,
        "crnn": CRNNModel,
        "crnn_f": CRNNFModel,
    }

    @classmethod
    def create(cls, model_name: str) -> OCRModel:
        """Create a model instance for the provided model name using sensible defaults."""

        model_class = cls._registry.get(model_name.lower())
        if model_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        return model_class()

    # a rendre plus dynamique en utilisant les arguments du fichier de config
    # on utilise des arguments plus précis de la liste de config des qu'on veut comparer des resultats obtenus avec des paramètres différents pour un meme model
    # se met en place facilement