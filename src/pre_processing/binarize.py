"""Binarization preprocessor that converts images to binary (black and white)."""

from __future__ import annotations

from PIL import Image, ImageOps

from src.pre_processing.base import ImagePreprocessor, ImageInput


class BinarizePreprocessor(ImagePreprocessor):
    """Preprocessor that converts images to binary (black and white) using thresholding."""

    def __init__(self, threshold: int = 128):
        """
        Initialize the binarization preprocessor.
        
        Args:
            threshold: Threshold value for binarization (0-255). 
                      Pixels above this value become white, below become black.
                      Default is 128.
        """
        if not 0 <= threshold <= 255:
            raise ValueError(f"Threshold must be between 0 and 255, got {threshold}")
        self._threshold = threshold

    def preprocess(self, image: ImageInput) -> Image.Image:
        """
        Convert the image to binary (black and white).
        
        Args:
            image: Either a path to an image file or a PIL Image object
            
        Returns:
            PIL.Image: The binarized image (mode '1' - 1-bit pixels)
        """
        # Convert to PIL Image if path is provided
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        
        # Convert to grayscale first
        img_gray = img.convert("L")
        
        # Apply threshold to create binary image
        # Pixels >= threshold become white (255), others become black (0)
        img_binary = img_gray.point(lambda p: 255 if p >= self._threshold else 0, mode='1')
        
        return img_binary

    def get_name(self) -> str:
        """Return a human-readable preprocessor name."""
        return f"Binarize(threshold={self._threshold})"
