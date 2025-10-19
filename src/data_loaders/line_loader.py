# src/data_loaders/line_loader.py
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def load_iam_lines(data_path: str) -> List[Tuple[str, str]]:
    """Load IAM line images and their ground truth text.
    
    For minimal implementation, this loads a single test sample.
    Expected structure:
    data_path/
    ├── lines/           # Line images
    │   ├── a01-000u-00.png
    │   └── ...
    └── lines.txt        # Ground truth text file
    
    Args:
        data_path: Path to the IAM dataset directory
        
    Returns:
        List of tuples: [(image_path, ground_truth_text), ...]
        
    Raises:
        Exception: If data files are not found
    """
    lines_dir = os.path.join(data_path, "lines")
    lines_txt = os.path.join(data_path, "lines.txt")
    
    if not os.path.exists(lines_dir):
        raise Exception(f"Lines directory not found: {lines_dir}")
    if not os.path.exists(lines_txt):
        raise Exception(f"Lines text file not found: {lines_txt}")
    
    # Load ground truth text
    ground_truth = {}
    with open(lines_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    line_id, text = parts
                    ground_truth[line_id] = text
    
    # Load image files and match with ground truth
    samples = []
    image_files = [f for f in os.listdir(lines_dir) if f.endswith('.png')]
    
    # For minimal implementation, take only the first sample
    if image_files:
        image_file = image_files[0]
        line_id = image_file.replace('.png', '')
        
        if line_id in ground_truth:
            image_path = os.path.join(lines_dir, image_file)
            samples.append((image_path, ground_truth[line_id]))
            logger.info(f"Loaded sample: {line_id} -> {ground_truth[line_id]}")
        else:
            logger.warning(f"No ground truth found for {line_id}")
    
    if not samples:
        raise Exception("No valid samples found in the dataset")
    
    logger.info(f"Loaded {len(samples)} sample(s) from IAM dataset")
    return samples
