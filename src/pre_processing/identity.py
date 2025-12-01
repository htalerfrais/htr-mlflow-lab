"""Identity preprocessor that returns the image unchanged."""

from __future__ import annotations

from PIL import Image

from src.pre_processing.base import ImagePreprocessor, ImageInput


class IdentityPreprocessor(ImagePreprocessor):
    """Preprocessor that returns the image unchanged (pass-through)."""

    def preprocess(self, image: ImageInput) -> Image.Image:
        """
        Return the image unchanged.
        
        Args:
            image: Either a path to an image file or a PIL Image object
            
        Returns:
            PIL.Image: The same image (converted to RGB if needed)
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def get_name(self) -> str:
        """Return a human-readable preprocessor name."""
        return "Identity"

