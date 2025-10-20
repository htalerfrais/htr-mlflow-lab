# src/data_loaders/line_loader.py
import logging
from typing import List, Tuple
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_iam_lines(data_path: str = None) -> List[Tuple[str, str]]:
    """Load IAM line images and their ground truth text from Hugging Face.
    
    For minimal implementation, this loads a single test sample.
    
    Args:
        data_path: Ignored (kept for compatibility), uses Hugging Face dataset
        
    Returns:
        List of tuples: [(image_path, ground_truth_text), ...]
        
    Raises:
        Exception: If dataset loading fails
    """
    try:
        # Load IAM dataset from Hugging Face
        logger.info("Loading IAM dataset from Hugging Face...")
        dataset = load_dataset("iam-handwriting-database", "lines")
        
        # Get the first sample for minimal testing
        first_sample = dataset['train'][0]
        
        # Extract image and text
        image = first_sample['image']
        text = first_sample['text']
        
        # Save image temporarily and return path
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "sample.png")
        image.save(image_path)
        
        samples = [(image_path, text)]
        logger.info(f"Loaded sample: {text}")
        logger.info(f"Loaded {len(samples)} sample(s) from IAM dataset")
        
        return samples
        
    except Exception as e:
        logger.error(f"Failed to load IAM dataset: {e}")
        raise Exception(f"Failed to load IAM dataset: {e}")
