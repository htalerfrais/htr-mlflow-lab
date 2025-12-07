from __future__ import annotations

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.models.base import OCRModel, ImageInput


class TrOCRModel(OCRModel):
    """Wrapper around Hugging Face TrOCR models."""

    def __init__(
        self,
        pretrained_model_name: str = "microsoft/trocr-base-handwritten",
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self._pretrained_model_name = pretrained_model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._max_new_tokens = max_new_tokens

        self._processor = TrOCRProcessor.from_pretrained(pretrained_model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name)
        self._model.to(self._device)
        self._model.eval()

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
        pil_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB") # convert PIL image into RGB format (handles images and paths of images)

        with torch.no_grad():
            # Model-specific preprocessing: TrOCRProcessor handles resize to 384x384,
            # normalization, and tensor conversion internally
            inputs = self._processor(images=pil_image, return_tensors="pt").pixel_values.to(self._device) # convert image to tensor and send to device
            generated_ids = self._model.generate(inputs, max_new_tokens=self._max_new_tokens)

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0] #decode vocabulary ids into text tokens. 
        return text.strip()

    def get_name(self) -> str:
        return f"TrOCR(pretrained={self._pretrained_model_name})"
