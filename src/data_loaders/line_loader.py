# src/data_loaders/line_loader.py
import logging
import tempfile
import os
from typing import List, Tuple
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_iam_lines(data_path: str = None) -> List[Tuple[str, str]]:
    try:
        # Load IAM from Hugging Face
        dataset = load_dataset("Teklia/IAM-line", "default")
        
        # Get the first 
        first_sample = dataset['validation'][0]
        
        # Extract image and text
        image = first_sample['image']
        text = first_sample['text']
        
        # Save image temporily and return path
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "sample.png")
        image.save(image_path)
        
        samples = [(image_path, text)]
        logger.info(f"Loaded sample: {text}")
        logger.info(f"Loaded {len(samples)} sample(s) from IAM dataset")
        
        return samples
        
    except Exception as e:
        raise Exception(f"Failed to load IAM dataset: {e}")
