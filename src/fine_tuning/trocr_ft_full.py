from __future__ import annotations

import logging
import os
import random
import torch
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    default_data_collator,
    EvalPrediction,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
import evaluate
import mlflow
from dotenv import load_dotenv
from src.data.local_importer import LocalLineTextImporter
from src.utils.metrics import calculate_cer
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class LocalTrOCRDataset(Dataset):
    """Dataset wrapper for local line images with text transcriptions."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        processor: TrOCRProcessor,
        max_target_length: int = 256,
    ):
        self.samples = samples
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, transcription = self.samples[idx]
        
        image = Image.open(image_path).convert("RGB")
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(transcription, padding="max_length", max_length=self.max_target_length, return_tensors="pt").input_ids.squeeze(0)
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def load_samples(lines_dir: Path, ground_truth_txt_path: Path, dataset_name: str = "local_dataset") -> List[Tuple[str, str]]:
    """Load the (image_path, transcription) tuples from the local dataset."""
    importer = LocalLineTextImporter(
        lines_dir=str(lines_dir),
        ground_truth_txt_path=str(ground_truth_txt_path),
        display_name=dataset_name,
    )
    samples, dataset_info = importer.import_data()

    if not samples:
        raise RuntimeError("No samples found for fine-tuning. Check dataset paths.")

    logger.info("Loaded %d samples from %s", len(samples), dataset_info["dataset_root"])
    return samples


cer_metric = evaluate.load("cer")

def compute_metrics(processor: TrOCRProcessor, pred: EvalPrediction):
    # Récupère les ids de prédiction
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # Décodage des prédictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Prépare labels et décodage
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    print(f"\n--- EXEMPLES DE PRÉDICTIONS (Total: {len(label_str)}) ---")
    for i in range(min(8, len(label_str))): # On en affiche 5
        print(f"REF: {label_str[i]}")
        print(f"PRED: {pred_str[i]}")
        print("-" * 20)

    # Calcul du CER via Hugging Face
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def main():
    # Minimal hyperparameters
    per_device_train_batch_size = 16
    per_device_eval_batch_size = 16
    learning_rate = 5e-6
    num_train_epochs = 10
    max_target_length = 150
    seed = 42
    train_ratio = 0.85

    # Dataset paths
    lines_dir = Path("data_local/perso_dataset/fatima_full_1100_lines/dataset_train_val/lines")
    ground_truth_txt_path = Path("data_local/perso_dataset/fatima_full_1100_lines/dataset_train_val/gt_train_val.txt")
    dataset_name = "fatima_full_1100_lines"
    
    # Output paths
    output_dir = Path("models_local/full_finetuned")
    mlflow_experiment_name = "full-fine-tune"

    # Load env vars from project root .env
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    torch.manual_seed(seed)
    random.seed(seed)

    # Resolve paths
    lines_dir = lines_dir.expanduser().resolve()
    ground_truth_txt_path = ground_truth_txt_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tb"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Configure MLflow
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # Load and split data
    logger.info("Loading local dataset...")
    if not lines_dir.exists():
        raise FileNotFoundError(f"Lines directory does not exist: {lines_dir}")
    if not ground_truth_txt_path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {ground_truth_txt_path}")

    samples = load_samples(lines_dir, ground_truth_txt_path, dataset_name)
    random.shuffle(samples)
    
    split_idx = max(1, int(len(samples) * train_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    logger.info(f"Using {len(train_samples)} train samples")
    logger.info(f"Using {len(val_samples)} validation samples")
    
    # Load model
    logger.info("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Configure tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Configure generation using GenerationConfig
    generation_config = GenerationConfig(
        max_length=max_target_length,
        num_beams=4,
        early_stopping=False,
        no_repeat_ngram_size=2,
        length_penalty=1,
        eos_token_id=processor.tokenizer.sep_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        decoder_start_token_id=processor.tokenizer.cls_token_id,
    )
    model.generation_config = generation_config

    # Create datasets
    train_dataset = LocalTrOCRDataset(
        train_samples,
        processor, 
        max_target_length=max_target_length,
    )
    eval_dataset = LocalTrOCRDataset(
        val_samples,
        processor, 
        max_target_length=max_target_length,
    )
    
    # Training arguments (using defaults for most parameters)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_strategy="steps",
        logging_steps=50,
        logging_dir=str(tb_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=["mlflow", "tensorboard"],
        save_total_limit=3,
    )
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "max_target_length": max_target_length,
            "train_ratio": train_ratio,
            "dataset_name": dataset_name,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
        })
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            tokenizer=processor,
            compute_metrics=lambda pred: compute_metrics(processor, pred),
        )

        logger.info("Evaluating model before training...")
        initial_metrics = trainer.evaluate()
        logger.info(f"Initial CER: {initial_metrics['eval_cer']:.4f}")
        
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")
        
        # Save model and processor
        final_model_dir = output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_model_dir)
        processor.save_pretrained(final_model_dir)
        
        # Log artifacts to MLflow
        if final_model_dir.exists():
            mlflow.log_artifacts(str(final_model_dir), artifact_path="model")
        if tb_dir.exists():
            mlflow.log_artifacts(str(tb_dir), artifact_path="tensorboard")
        
        active_run = mlflow.active_run()
        if active_run is not None:
            logger.info("MLflow run_id: %s", active_run.info.run_id)


if __name__ == "__main__":
    main()
