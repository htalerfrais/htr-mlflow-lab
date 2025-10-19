# src/utils/tesseract_executor.py
import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def run_tesseract_ocr(image_path: str, lang: str = "eng", engine_mode: int = 3) -> str:
    """Run Tesseract OCR on an image and return the transcribed text.
    
    Args:
        image_path: Path to the input image
        lang: Language code for Tesseract (default: "eng")
        engine_mode: Tesseract engine mode (default: 3 for OSD and OCR)
        
    Returns:
        Transcribed text as string
        
    Raises:
        Exception: If Tesseract fails to process the image
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Configure Tesseract
        config = f'--oem {engine_mode} -l {lang}'
        
        # Run OCR
        text = pytesseract.image_to_string(image, config=config)
        
        # Clean up the text (remove extra whitespace)
        text = text.strip()
        
        logger.info(f"Tesseract OCR completed for {image_path}")
        return text
        
    except Exception as e:
        logger.error(f"Tesseract OCR failed for {image_path}: {str(e)}")
        raise Exception(f"Tesseract OCR failed: {str(e)}")
