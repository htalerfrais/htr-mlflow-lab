from __future__ import annotations

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer

from src.models.base import OCRModel, ImageInput


class TrOCRFrModel(OCRModel):
    """Wrapper around French fine-tuned Hugging Face TrOCR models."""

    def __init__(
        self,
        processor_name: str = "microsoft/trocr-large-handwritten",
        model_name: str = "agomberto/trocr-large-handwritten-fr",
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self._processor_name = processor_name
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._max_new_tokens = max_new_tokens

        self._processor = TrOCRProcessor.from_pretrained(processor_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            use_safetensors=False,  # Use pytorch_model.bin, ignore safetensors
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        
        # Force complete model loading by accessing all parameters
        # This ensures no lazy loading happens during inference
        _ = list(self._model.parameters())
        
        # Warmup: trigger any remaining lazy loading with a dummy forward pass
        # This ensures the model is fully loaded before real inference starts
        with torch.no_grad():
            dummy_input = torch.zeros((1, 3, 384, 384), device=self._device)
            try:
                _ = self._model.generate(dummy_input, max_length=10)
            except Exception:
                # If warmup fails, it's okay - model will load during first real inference
                pass

    def predict(self, image: ImageInput) -> str:
        """
        Run OCR inference on an image.
        
        Note: This method applies model-specific preprocessing via TrOCRProcessor
        (resize to 384x384, normalization, tensor conversion). Common preprocessing
        (binarization, cropping, etc.) should be applied before calling this method
        via the pipeline's preprocessor.
        
        Args:
            image: Either a path to an image file or a PIL Image object
                  (may already be preprocessed by the common preprocessor)
        
        Returns:
            str: The recognized text from the image
        """
        pil_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")

        with torch.no_grad():
            # Model-specific preprocessing: TrOCRProcessor handles resize to 384x384,
            # normalization, and tensor conversion internally
            pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values.to(self._device)
            generated_ids = self._model.generate(pixel_values, max_new_tokens=self._max_new_tokens)

        # Use the tokenizer from the fine-tuned model for decoding
        text = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def get_name(self) -> str:
        return f"TrOCR-FR(model={self._model_name}, processor={self._processor_name})"
