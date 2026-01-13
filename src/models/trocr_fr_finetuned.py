from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from src.models.base import OCRModel, ImageInput


class TrOCRFrFinetunedModel(OCRModel):
    """Inference wrapper for the LoRA fine-tuned TrOCR French model."""

    def __init__(
        self,
        adapters_dir: Path | str = Path("models_local/finetuned/adapters"),
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._adapters_dir = Path(adapters_dir)
        self._max_new_tokens = max_new_tokens

        base_model = VisionEncoderDecoderModel.from_pretrained(
            "agomberto/trocr-large-handwritten-fr"
        )
        self._model = PeftModel.from_pretrained(base_model, self._adapters_dir)
        self._model.to(self._device)
        self._model.eval()

        self._processor = TrOCRProcessor.from_pretrained(self._adapters_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(self._adapters_dir)

    def predict(self, image: ImageInput) -> str:
        pil_image = (
            Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        )

        inputs = self._processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values=pixel_values,
                max_new_tokens=self._max_new_tokens,
            )

        text = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def get_name(self) -> str:
        return "TrOCR-FR (LoRA)"
