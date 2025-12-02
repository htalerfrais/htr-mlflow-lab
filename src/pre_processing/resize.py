"""Resize preprocessor that resizes images to a fixed height."""

from __future__ import annotations

from PIL import Image

from src.pre_processing.base import ImagePreprocessor, ImageInput


class ResizePreprocessor(ImagePreprocessor):
    """Preprocessor that resizes images to a fixed height using Lanczos interpolation."""

    def __init__(self, height: int = 128):
        self._height = height

    def preprocess(self, image: ImageInput) -> Image.Image:

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        
        # Calculate new width maintaining aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(self._height * aspect_ratio)
        
        return img.resize((new_width, self._height), Image.LANCZOS)

    def get_name(self) -> str:
        return f"Resize(height={self._height})"

