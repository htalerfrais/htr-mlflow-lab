"""Tesseract OCR model implementation."""

from __future__ import annotations

import logging
from typing import Union

import pytesseract
from PIL import Image

from src.models.base import OCRModel, ImageInput


logger = logging.getLogger(__name__)


class TesseractModel(OCRModel):
    """Wrapper around pytesseract with a consistent interface."""

    def __init__(self, lang: str = "eng", engine_mode: int = 3) -> None:
        self._lang = lang
        self._engine_mode = engine_mode

    def predict(self, image: ImageInput) -> str:
        try:
            pil_image = Image.open(image) if isinstance(image, str) else image

            config = f"--oem {self._engine_mode} -l {self._lang}"
            text = pytesseract.image_to_string(pil_image, config=config).strip()

            logger.info("Tesseract OCR completed")
            return text

        except Exception as exc:
            logger.error("Tesseract OCR failed: %s", exc)
            raise Exception(f"Tesseract OCR failed: {exc}") from exc

    def get_name(self) -> str:
        return f"Tesseract(lang={self._lang}, oem={self._engine_mode})"

