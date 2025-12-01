"""
Utilities for fine-tuning OCR models
"""

from typing import Dict, List, Any
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from Levenshtein import distance as levenshtein_distance


@dataclass
class TrOCRDataCollator:
    """
    Data collator for TrOCR fine-tuning
    Handles padding and batching of image-text pairs
    """
    
    processor: Any
    padding: bool = True
    max_length: int = 64
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of samples with 'pixel_values' and 'labels'
            
        Returns:
            Batched tensors
        """
        # Extract pixel values and labels
        pixel_values = [item['pixel_values'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Stack pixel values
        pixel_values = torch.stack(pixel_values)
        
        # Pad labels
        max_label_length = max(len(label) for label in labels)
        padded_labels = []
        
        for label in labels:
            padding_length = max_label_length - len(label)
            # Pad with -100 (ignored by loss function)
            padded_label = label + [-100] * padding_length
            padded_labels.append(padded_label)
        
        labels = torch.tensor(padded_labels, dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }


def compute_cer(pred_str: str, label_str: str) -> float:
    """
    Compute Character Error Rate
    
    Args:
        pred_str: Predicted text
        label_str: Ground truth text
        
    Returns:
        CER as float
    """
    if len(label_str) == 0:
        return 1.0 if len(pred_str) > 0 else 0.0
    
    return levenshtein_distance(pred_str, label_str) / len(label_str)


def compute_wer(pred_str: str, label_str: str) -> float:
    """
    Compute Word Error Rate
    
    Args:
        pred_str: Predicted text
        label_str: Ground truth text
        
    Returns:
        WER as float
    """
    pred_words = pred_str.split()
    label_words = label_str.split()
    
    if len(label_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    return levenshtein_distance(pred_words, label_words) / len(label_words)


def compute_metrics(pred_ids: List[List[int]], label_ids: List[List[int]], 
                   processor: Any) -> Dict[str, float]:
    """
    Compute evaluation metrics for OCR
    
    Args:
        pred_ids: Predicted token IDs
        label_ids: Ground truth token IDs
        processor: TrOCR processor for decoding
        
    Returns:
        Dictionary with CER and WER
    """
    # Decode predictions and labels
    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding)
    label_ids = [[id if id != -100 else processor.tokenizer.pad_token_id 
                  for id in label] for label in label_ids]
    label_strs = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute CER and WER
    cers = [compute_cer(pred, label) for pred, label in zip(pred_strs, label_strs)]
    wers = [compute_wer(pred, label) for pred, label in zip(pred_strs, label_strs)]
    
    return {
        'cer': sum(cers) / len(cers) if cers else 0.0,
        'wer': sum(wers) / len(wers) if wers else 0.0,
    }


class EarlyStoppingCallback:
    """
    Early stopping callback for training
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to consider as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, current_loss: float) -> bool:
        """
        Check if training should stop
        
        Args:
            current_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            return False
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
