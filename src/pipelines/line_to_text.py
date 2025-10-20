# src/pipelines/line_to_text.py
import logging
from typing import Dict, Any

from src.data_loaders.line_loader import load_iam_lines
from src.utils.tesseract_executor import run_tesseract_ocr
from src.utils.metrics import calculate_cer, calculate_wer

logger = logging.getLogger(__name__)


def run_pipeline(config: Dict[str, Any]) -> Dict[str, float]:
    """Run the line-to-text pipeline for Tesseract baseline.
    
    Args:
        config: Configuration dictionary from YAML file
        
    Returns:
        Dictionary containing final metrics: {"final_cer": float, "final_wer": float}
    """
    logger.info("Starting line-to-text pipeline")
    
    # Check if this is a Tesseract experiment
    if config.get("model") != "Tesseract":
        raise ValueError(f"Pipeline only supports Tesseract model, got: {config.get('model')}")
    
    # Load test data
    data_path = config.get("data_path", "data/iam")
    logger.info(f"Loading data from: {data_path}")
    
    try:
        samples = load_iam_lines(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Get Tesseract parameters
    tesseract_params = config.get("params", {})
    lang = tesseract_params.get("tesseract_language", "eng")
    engine_mode = tesseract_params.get("tesseract_engine_mode", 3)
    
    logger.info(f"Tesseract config: lang={lang}, engine_mode={engine_mode}")
    
    # Process each sample
    total_cer = 0.0
    total_wer = 0.0
    num_samples = len(samples)
    
    for i, (image_path, ground_truth) in enumerate(samples):
        logger.info(f"Processing sample {i+1}/{num_samples}: {image_path}")
        
        try:
            # Run Tesseract OCR
            prediction = run_tesseract_ocr(image_path, lang, engine_mode)
            
            # Calculate metrics
            cer = calculate_cer(ground_truth, prediction)
            wer = calculate_wer(ground_truth, prediction)
            
            total_cer += cer
            total_wer += wer
            
            logger.info(f"Sample {i+1} - CER: {cer:.4f}, WER: {wer:.4f}")
            logger.info(f"Ground truth: '{ground_truth}'")
            logger.info(f"Prediction: '{prediction}'")
            
        except Exception as e:
            logger.error(f"Failed to process sample {i+1}: {e}")
            raise
    
    # Calculate final metrics
    final_cer = total_cer / num_samples
    final_wer = total_wer / num_samples
    
    logger.info(f"Pipeline completed - Final CER: {final_cer:.4f}, Final WER: {final_wer:.4f}")
    
    return {
        "final_cer": final_cer,
        "final_wer": final_wer
    }
