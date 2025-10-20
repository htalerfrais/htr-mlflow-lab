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
        
        # Create temporary directory 
        temp_dir = tempfile.mkdtemp() # create tempo directory
        
        samples = []
        validation_data = dataset['validation'] #take a separate split 
        
        # Process all samples in the validation set
        for i, sample in enumerate(validation_data): # loop creating an index for each sample
            image = sample['image']
            text = sample['text']
            
            # Save image temporarily and return path
            image_path = os.path.join(temp_dir, f"sample_{i}.png") 
            image.save(image_path) # save image in the tempo dir 
            
            samples.append((image_path, text))
        
        logger.info(f"Loaded {len(samples)} sample(s) from IAM dataset")
        
        return samples
        
    except Exception as e:
        raise Exception(f"Failed to load IAM dataset: {e}")
