from __future__ import annotations

import logging
import torch
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
from datasets import load_dataset
import evaluate
from src.utils.metrics import calculate_cer
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class HFTrOCRDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets (e.g., RIMES, IAM)."""

    def __init__(
        self,
        hf_dataset,
        processor: TrOCRProcessor,
        max_target_length: int = 256,
    ):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        transcription = item.get("text") or item.get("ground_truth", "")
        labels = self.processor.tokenizer(transcription, padding="max_length", max_length=self.max_target_length, return_tensors="pt").input_ids.squeeze(0)
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


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

    torch.manual_seed(seed)

    # Load data
    logger.info("Loading IAM-line dataset from HuggingFace...")
    hf_dataset = load_dataset("Teklia/RIMES-2011-line")
    
    N_train_samples = 5000
    train_data = hf_dataset["train"].shuffle(seed=seed).select(range(N_train_samples))
    val_data = hf_dataset["validation"]

    
    logger.info(f"Using {len(train_data)} train samples")
    logger.info(f"Using {len(val_data)} validation samples")
    
    # Load model
    logger.info("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

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
    train_dataset = HFTrOCRDataset(
        train_data,
        processor, 
        max_target_length=max_target_length,
    )
    eval_dataset = HFTrOCRDataset(
        val_data,
        processor, 
        max_target_length=max_target_length,
    )
    
    # Training arguments (using defaults for most parameters)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
    )
    
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


if __name__ == "__main__":
    main()
