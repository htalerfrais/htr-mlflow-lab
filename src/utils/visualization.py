"""Utility functions for creating visualizations."""

from __future__ import annotations

import os
import tempfile
import logging
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

logger = logging.getLogger(__name__)


def create_preprocessing_visualization(pipeline, num_samples: int = 5) -> Optional[str]:
    """
    Generate preprocessing visualization using the pipeline's data importer and preprocessor.
    
    Args:
        pipeline: Pipeline instance (must have _data_importer and _preprocessor attributes)
        num_samples: Number of samples to visualize (default: 5)
        
    Returns:
        Path to the saved visualization image file, or None if visualization failed
    """
    try:
        # Load samples using pipeline's data importer
        samples, _ = pipeline._data_importer.import_data()
        num_samples = min(num_samples, len(samples))
        
        if num_samples == 0:
            logger.warning("No samples available for preprocessing visualization")
            return None
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        preprocessor_name = pipeline._preprocessor.get_name() if pipeline._preprocessor else "None"
        
        # Process each sample
        for idx, (image_path, ground_truth) in enumerate(samples[:num_samples]):
            original = Image.open(image_path).convert("RGB")
            preprocessed = pipeline._preprocessor.preprocess(image_path) if pipeline._preprocessor else original
            
            axes[idx, 0].imshow(original, cmap='gray' if original.mode == 'L' else None)
            axes[idx, 0].set_title(f"Original\n{os.path.basename(image_path)}", fontsize=9)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(preprocessed, cmap='gray' if preprocessed.mode == 'L' else None)
            axes[idx, 1].set_title(f"After: {preprocessor_name}", fontsize=9)
            axes[idx, 1].axis('off')
            
            fig.text(0.5, 0.98 - (idx + 0.5) / num_samples * 0.9,
                    f"GT: {ground_truth[:50]}..." if len(ground_truth) > 50 else f"GT: {ground_truth}",
                    ha='center', fontsize=8, style='italic')
        
        fig.suptitle(f'Preprocessing: {preprocessor_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            plt.savefig(tmp_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        logger.info(f"Created preprocessing visualization with {num_samples} samples")
        return tmp_path
        
    except Exception as e:
        logger.warning(f"Failed to create preprocessing visualization: {e}")
        return None

