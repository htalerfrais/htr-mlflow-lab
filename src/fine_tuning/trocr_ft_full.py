from __future__ import annotations

import logging
import random
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import mlflow
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    default_data_collator,
    EvalPrediction,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    GenerationConfig,
)
import albumentations as A

from src.data.local_importer import LocalLineImporter
from src.utils.metrics import calculate_cer
from dotenv import load_dotenv
from datasets import load_dataset


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class TrOCRLineDataset(Dataset):
    """Dataset that turns (image_path, transcription) pairs into model inputs."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        processor: TrOCRProcessor,
        max_target_length: int = 256,
        augmentation_pipeline=None,
    ):
        self.samples = samples
        self.processor = processor
        self.max_target_length = max_target_length
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, transcription = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Apply data augmentation if configured
        if self.augmentation_pipeline is not None:
            image_np = np.array(image)
            augmented = self.augmentation_pipeline(image=image_np)
            image = Image.fromarray(augmented["image"])
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(transcription, padding="max_length", max_length=self.max_target_length, return_tensors="pt").input_ids.squeeze(0)
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": labels}


class HFTrOCRDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets (e.g., RIMES)."""

    def __init__(
        self,
        hf_dataset,
        processor: TrOCRProcessor,
        max_target_length: int = 256,
        augmentation_pipeline=None,
    ):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        
        # Image is already a PIL Image in HF datasets
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply data augmentation if configured
        if self.augmentation_pipeline is not None:
            image_np = np.array(image)
            augmented = self.augmentation_pipeline(image=image_np)
            image = Image.fromarray(augmented["image"])
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Get text (field might be "text" or "ground_truth")
        transcription = item.get("text") or item.get("ground_truth", "")
        labels = self.processor.tokenizer(transcription, padding="max_length", max_length=self.max_target_length, return_tensors="pt").input_ids.squeeze(0)
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": labels}


def load_samples(images_dir: Path, ground_truth_path: Path) -> List[Tuple[str, str]]:
    """Load the (image_path, transcription) tuples from the local dataset."""
    importer = LocalLineImporter(
        images_dir=str(images_dir),
        ground_truth_path=str(ground_truth_path),
        image_template="*line_{id:04d}.jpg",
    )
    samples, dataset_info = importer.import_data()

    if not samples:
        raise RuntimeError("No samples found for fine-tuning. Check dataset paths.")

    logger.info("Loaded %d samples from %s", len(samples), dataset_info["dataset_root"])
    return samples


def compute_cer(processor: TrOCRProcessor):
    """Returns a function that computes CER given a processor."""
    def _compute_cer(prediction: EvalPrediction) -> dict:
        generated_ids = prediction.predictions
        labels_ids = prediction.label_ids

        decoded_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        decoded_labels = processor.batch_decode(labels_ids, skip_special_tokens=True)

        # Debug: print first prediction
        if len(decoded_preds) > 0:
            logger.info(f"[EVAL] Label: '{decoded_labels[0]}'")
            logger.info(f"[EVAL] Pred:  '{decoded_preds[0]}'")

        cer_scores = [calculate_cer(ref, hyp) for ref, hyp in zip(decoded_labels, decoded_preds)]
        average_cer = float(np.mean(cer_scores)) if cer_scores else float("nan")
        return {"cer": average_cer}
    return _compute_cer


def create_augmentation_pipeline(config: dict):
    if not config.get("enabled", False):
        return None
    
    transforms = []
    
    # Rotation (very light for OCR)
    if config.get("rotation_enabled", False):
        transforms.append(A.Rotate(
            limit=config.get("rotation_limit", 2),
            p=config.get("rotation_p", 0.5),
            border_mode=0,
        ))
    
    # Brightness & Contrast
    if config.get("brightness_contrast_enabled", False):
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=config.get("brightness_limit", 0.2),
            contrast_limit=config.get("contrast_limit", 0.2),
            p=config.get("brightness_contrast_p", 0.5),
        ))
    
    # Gaussian Blur
    if config.get("gaussian_blur_enabled", False):
        transforms.append(A.GaussianBlur(
            blur_limit=config.get("gaussian_blur_limit", (3, 3)),
            p=config.get("gaussian_blur_p", 0.3),
        ))
    
    # Gaussian Noise
    if config.get("gaussian_noise_enabled", False):
        transforms.append(A.GaussNoise(
            std_range=config.get("gaussian_noise_std_range", (0.01, 0.05)),
            mean_range=config.get("gaussian_noise_mean_range", (0.0, 0.0)),
            p=config.get("gaussian_noise_p", 0.3),
        ))
    
    if not transforms:
        logger.warning("Data augmentation enabled but no transforms are configured.")
        return None
    
    return A.Compose(transforms)


def main():
    # ===== HYPERPARAMETERS - FULL FINE-TUNING =====
    # Training hyperparameters (optimized for full fine-tuning)
    per_device_train_batch_size = 8   # Smaller batch for full fine-tuning
    per_device_eval_batch_size = 8
    learning_rate = 5e-6              # Much smaller LR for full fine-tuning
    num_train_epochs = 5              # Fewer epochs to avoid overfitting
    weight_decay = 0.05               # Higher weight decay for regularization
    warmup_ratio = 0.3                # More warmup
    max_target_length = 128           # Good for RIMES
    logging_steps = 50
    seed = 42
    fp16 = True                       # Enable mixed precision for speed
    gradient_accumulation_steps = 2   # Effective batch size = 16
    
    # Data augmentation configuration (light for clean datasets)
    augmentation_config = {
        "enabled": True,
        "rotation_enabled": True,
        "rotation_limit": 1,
        "rotation_p": 0.3,
        "brightness_contrast_enabled": True,
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "brightness_contrast_p": 0.4,
        "gaussian_blur_enabled": False,
        "gaussian_noise_enabled": True,
        "gaussian_noise_std_range": (0.005, 0.02),
        "gaussian_noise_p": 0.2,
    }
    
    # ===== PATHS AND CONFIGURATION =====
    images_dir = Path("data_local/perso_dataset/hector_200_more_lines_extended/lines_out_sorted")
    ground_truth_path = Path("data_local/perso_dataset/hector_200_more_lines_extended/gt_hector_pages_lines.json")
    output_dir = Path("models_local/finetuned/full_ft")  # Different output directory
    mlflow_experiment_name = "trocr-full-finetuning"     # Different experiment name

    # Load env vars
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    random.seed(seed)
    torch.manual_seed(seed)

    images_dir = images_dir.expanduser().resolve()
    ground_truth_path = ground_truth_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    model_dir = output_dir / "final_model"
    tb_dir = output_dir / "tb"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {ground_truth_path}")

    # ----- IMPORTING DATA -----
    logger.info("Loading RIMES dataset from HuggingFace...")
    hf_dataset = load_dataset("Teklia/RIMES-2011-line")
    
    # Reduce dataset size for faster experimentation
    N_train_samples = 2000  # Use more samples for full fine-tuning
    train_data = hf_dataset["train"].shuffle(seed=seed).select(range(N_train_samples))
    val_data = hf_dataset["validation"]
    
    logger.info(f"Using {len(train_data)} train samples (out of {len(hf_dataset['train'])} total)")
    logger.info(f"Using {len(val_data)} validation samples")
    
    # ----- LOADING MODEL (NO LoRA) -----
    logger.info("Loading TrOCR model for FULL fine-tuning (no LoRA)...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Configuration des tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Create explicit generation config
    generation_config = GenerationConfig(
        max_length=max_target_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        eos_token_id=processor.tokenizer.sep_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        decoder_start_token_id=processor.tokenizer.cls_token_id,
    )
    model.generation_config = generation_config

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} (100%)")
    logger.info(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB")

    logger.info(f"Tokenizer special tokens:")
    logger.info(f"  - pad_token_id: {processor.tokenizer.pad_token_id}")
    logger.info(f"  - cls_token_id: {processor.tokenizer.cls_token_id}")
    logger.info(f"  - sep_token_id: {processor.tokenizer.sep_token_id}")

    # ----- CREATING DATASETS -----
    augmentation_pipeline = create_augmentation_pipeline(augmentation_config)
    
    train_dataset = HFTrOCRDataset(
        train_data,
        processor, 
        max_target_length=max_target_length,
        augmentation_pipeline=augmentation_pipeline
    )
    eval_dataset = HFTrOCRDataset(
        val_data,
        processor, 
        max_target_length=max_target_length,
        augmentation_pipeline=None
    )

    data_collator = default_data_collator 
    report_to = ["mlflow", "tensorboard"]

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_dir=str(tb_dir),
        report_to=report_to,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        optim="adamw_torch",
        save_total_limit=2,  # Keep only 2 best checkpoints
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "model": "trocr-base-handwritten",
            "fine_tuning_method": "full",
            "train_samples": len(train_data),
            "batch_size": per_device_train_batch_size,
            "gradient_accumulation": gradient_accumulation_steps,
            "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_target_length": max_target_length,
            "fp16": fp16,
        })
        
        # Log augmentation config
        aug_params = {f"aug_{k}": v for k, v in augmentation_config.items()}
        mlflow.log_params(aug_params)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
            compute_metrics=compute_cer(processor),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),  # Stop if no improvement
            ],
        )

        logger.info("Starting full fine-tuning...")
        trainer.train()

        # Save final model
        logger.info(f"Saving final model to {model_dir}")
        model.save_pretrained(model_dir)
        processor.save_pretrained(model_dir)

        # Log artifacts to MLflow
        if model_dir.exists():
            mlflow.log_artifacts(str(model_dir), artifact_path="final_model")
        if tb_dir.exists():
            mlflow.log_artifacts(str(tb_dir), artifact_path="tensorboard")

        active_run = mlflow.active_run()
        if active_run is not None:
            logger.info("MLflow run_id: %s", active_run.info.run_id)
            logger.info("Full fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
