from __future__ import annotations

from typing import Dict, Type

from src.models.base import OCRModel
from src.models.tesseract_model import TesseractModel
from src.models.trocr_model import TrOCRModel
from src.models.trocr_fr_model import TrOCRFrModel
from src.models.trocr_fr_finetuned import TrOCRFrFinetunedModel
from src.models.crnn_f_model import CRNNFModel
from src.models.qwen_vl import QwenVLModel


class ModelFactory:
    """Instantiate OCR models based on configuration names."""

    _registry: Dict[str, Type[OCRModel]] = {
        "tesseract": TesseractModel,
        "trocr": TrOCRModel,
        "trocr_fr": TrOCRFrModel,
        "trocr_fr_finetuned": TrOCRFrFinetunedModel,
        "crnn_f": CRNNFModel,
        "qwen_vl": QwenVLModel,
    }

    @classmethod
    def create(cls, model_name: str, **model_params) -> OCRModel:
        """Create a model instance for the provided model name."""
        
        model_class = cls._registry.get(model_name.lower())
        if model_class is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        return model_class(**model_params)

    # a rendre plus dynamique en utilisant les arguments du fichier de config
    # on utilise des arguments plus précis de la liste de config des qu'on veut comparer des resultats obtenus avec des paramètres différents pour un meme model
    # se met en place facilement