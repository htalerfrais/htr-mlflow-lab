from __future__ import annotations

import logging
import random
import os
import importlib.util
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import mlflow
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    default_data_collator,
    EvalPrediction,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
import albumentations as A

from peft import LoraConfig, TaskType, get_peft_model
from src.data.local_importer import LocalLineImporter
from src.utils.metrics import calculate_cer
from dotenv import load_dotenv


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
            # Albumentations expects numpy array
            image_np = np.array(image)
            augmented = self.augmentation_pipeline(image=image_np)
            image = Image.fromarray(augmented["image"])
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0) #squeeze(0) removes the first dimension of the tensor (batch dimension)

        labels = self.processor.tokenizer(transcription, padding="max_length", max_length=self.max_target_length).input_ids.squeeze(0) #squeeze(0) removes the first dimension of the tensor (batch dimension)
        # make pad token = -100 so that it is ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values, "labels": labels} # pixel_values : [3, 384, 384]
        return encoding


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



def compute_cer(prediction: EvalPrediction, processor: TrOCRProcessor) -> dict:
    generated_ids = prediction.predictions
    labels_ids = prediction.label_ids

    
    decoded_preds = processor.batch_decode(generated_ids, skip_special_tokens=True) # returning a list (batch)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    decoded_labels = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer_scores = [calculate_cer(ref, hyp) for ref, hyp in zip(decoded_labels, decoded_preds)]
    average_cer = float(np.mean(cer_scores)) if cer_scores else float("nan")
    return average_cer
    

def create_augmentation_pipeline(config: dict):
    if not config.get("enabled", False):
        return None
    
    transforms = []
    
    # Rotation (very light for OCR)
    if config.get("rotation_enabled", False):
        transforms.append(A.Rotate(
            limit=config.get("rotation_limit", 2),
            p=config.get("rotation_p", 0.5),
            border_mode=0,  # constant border with zeros
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
            std_range=config.get("gaussian_noise_std_range", (0.01, 0.05)),  # Normalisé entre 0 et 1
            mean_range=config.get("gaussian_noise_mean_range", (0.0, 0.0)),
            p=config.get("gaussian_noise_p", 0.3),
        ))
    
    if not transforms:
        logger.warning("Data augmentation enabled but no transforms are configured.")
        return None
    
    return A.Compose(transforms)



def main():
    # ===== HYPERPARAMETERS (hardcoded) =====
    # Training hyperparameters
    per_device_train_batch_size = 16
    per_device_eval_batch_size = 16
    learning_rate = 1e-5 
    num_train_epochs = 30
    weight_decay = 0.01
    warmup_ratio = 0.2     # 10% du temps pour monter en puissance
    max_target_length = 256
    logging_steps = 50
    seed = 42
    fp16 = False 
    train_ratio = 0.9
    
    # LoRA configuration
    lora_r = 16
    lora_alpha = lora_r
    lora_dropout = 0.1
    target_modules = [
        "query", "key", "value", "dense",                   # Encodeur (ViT)
        "q_proj", "k_proj", "v_proj", "out_proj",           # Décodeur (Attention)
        "fc1", "fc2"                                        # Couches Feed-Forward
    ]
    
    # Data augmentation configuration (OCR-friendly transforms)
    augmentation_config = {
        "enabled": True,
        "rotation_enabled": True,
        "rotation_limit": 2,
        "rotation_p": 0.5,
        "brightness_contrast_enabled": True,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "brightness_contrast_p": 0.5,
        "gaussian_blur_enabled": True,
        "gaussian_blur_limit": (3, 3),
        "gaussian_blur_p": 0.3,
        "gaussian_noise_enabled": True,
        "gaussian_noise_std_range": (0.01, 0.05),  # Normalisé entre 0 et 1 (1% à 5% de l'intensité max)
        "gaussian_noise_mean_range": (0.0, 0.0),  # centre du bruit
        "gaussian_noise_p": 0.3,
    }
    
    # ===== PATHS AND CONFIGURATION (hardcoded) =====
    images_dir = Path("data_local/perso_dataset/hector_200_more_lines_extended/lines_out_sorted")
    ground_truth_path = Path("data_local/perso_dataset/hector_200_more_lines_extended/gt_hector_pages_lines.json")
    output_dir = Path("models_local/finetuned/adapters")
    mlflow_experiment_name = "trocr-finetuning"  # MLflow experiment name

    # Load env vars from project root .env (if present), without forcing users to pass secrets via CLI.
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    random.seed(seed)
    torch.manual_seed(seed)

    images_dir = images_dir.expanduser().resolve()
    ground_truth_path = ground_truth_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    adapters_dir = output_dir / "adapters"
    tb_dir = output_dir / "tb"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {ground_truth_path}")



    # ----- IMPORTING DATA -----

    # loading, shuffling samples and spliting them in train & val samples
    samples = load_samples(images_dir, ground_truth_path)
    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * train_ratio))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    
    # ----- LOADING MODEL -----
    # loading processor, model, lora config, tokenizer
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-large-handwritten")  # ← AJOUTER CETTE LIGNE

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Configure model for training
    # For the base TrOCR model, special tokens are already well-defined
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info(f"Tokenizer special tokens:")
    logger.info(f"  - pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"  - bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"  - eos_token_id: {tokenizer.eos_token_id}")
    logger.info(f"  - decoder_start_token_id: {model.config.decoder_start_token_id}")

    # ----- CREATING DATASETS -----
    # Create augmentation pipeline (only for training)
    augmentation_pipeline = create_augmentation_pipeline(augmentation_config)
    if augmentation_pipeline:
        logger.info("Data augmentation enabled for training")
    else:
        logger.info("Data augmentation disabled")
    
    # on vient créer des datasets ingérables par le modèle respecrant le format Hugging Face
    train_dataset = TrOCRLineDataset(
        train_samples, 
        processor, 
        tokenizer, 
        max_target_length=max_target_length,
        augmentation_pipeline=augmentation_pipeline  # Apply augmentation only to train
    )
    eval_dataset = TrOCRLineDataset(
        eval_samples, 
        processor, 
        tokenizer, 
        max_target_length=max_target_length,
        augmentation_pipeline=None  # No augmentation for eval
    )

    data_collator = default_data_collator 

    report_to = ["mlflow", "tensorboard"]

    # Training arguments (hardcoded)
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
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        optim="adamw_torch",
        save_total_limit=3,
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    with mlflow.start_run():
        # Log data augmentation configuration to MLflow
        aug_params = {f"aug_{k}": v for k, v in augmentation_config.items()}
        mlflow.log_params(aug_params)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_cer_metrics(tokenizer),
            callbacks=[
                # EarlyStoppingCallback(early_stopping_patience=3),
            ],
        )

        trainer.train()

        # Save adapters locally first, then log them to MLflow (so the artifact path exists for inference).
        model.save_pretrained(adapters_dir)
        tokenizer.save_pretrained(adapters_dir)
        processor.save_pretrained(adapters_dir)

        if adapters_dir.exists():
            mlflow.log_artifacts(str(adapters_dir), artifact_path="adapters")
        if tb_dir.exists():
            mlflow.log_artifacts(str(tb_dir), artifact_path="tensorboard")

        active_run = mlflow.active_run()
        if active_run is not None:
            logger.info("MLflow run_id: %s", active_run.info.run_id)


if __name__ == "__main__":
    main()