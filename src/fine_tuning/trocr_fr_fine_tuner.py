from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from peft import LoraConfig, TaskType, get_peft_model

from src.data.local_importer import LocalLineImporter
from src.utils.metrics import calculate_cer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class TrOCRLineDataset(Dataset):
    """Dataset that turns (image_path, transcription) pairs into model inputs."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        processor: TrOCRProcessor,
        tokenizer: AutoTokenizer,
        max_target_length: int = 256,
    ):
        self.samples = samples
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, transcription = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        tokenized = self.tokenizer(
            transcription,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_attention_mask=False,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id for Seq2Seq training.")

        labels = torch.where(labels == pad_token_id, -100, labels)

        return {"pixel_values": pixel_values, "labels": labels}


class TrOCRDataCollator:
    """Collator that batches pixel values and labels while handling padding."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[dict]) -> dict:
        batch_pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_labels = torch.where(
            batch_labels == self.tokenizer.pad_token_id, -100, batch_labels
        )
        return {"pixel_values": batch_pixel_values, "labels": batch_labels}


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


def compute_cer_metrics(tokenizer: AutoTokenizer):
    """Build the metrics callback that returns the CER."""

    def _compute(prediction: EvalPrediction) -> dict:
        generated_ids = prediction.predictions
        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels = prediction.label_ids
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        cer_scores = [
            calculate_cer(ref, hyp) for ref, hyp in zip(decoded_labels, decoded_preds)
        ]
        average_cer = float(np.mean(cer_scores)) if cer_scores else float("nan")
        return {"cer": average_cer}

    return _compute


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR-FR on a local dataset.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data_local/perso_dataset/hector_pages_lines_3/lines_out_sorted"),
        help="Directory that contains the line images.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data_local/perso_dataset/hector_pages_lines_3/gt_hector_pages_lines.json"),
        help="JSON file with the ground-truth transcriptions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models_local/finetuned/adapters"),
        help="Directory where LoRA adapters will be saved.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Proportion of samples used for training.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-train-epochs", type=int, default=6)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling).")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate.")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    images_dir = args.images_dir.expanduser().resolve()
    ground_truth_path = args.ground_truth.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {ground_truth_path}")

    samples = load_samples(images_dir, ground_truth_path)
    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * args.train_ratio))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    if not eval_samples:
        eval_samples = train_samples[-1:]

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("agomberto/trocr-large-handwritten-fr")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained("agomberto/trocr-large-handwritten-fr")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define pad or EOS token id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = tokenizer.vocab_size

    train_dataset = TrOCRLineDataset(
        train_samples, processor, tokenizer, max_target_length=args.max_target_length
    )
    eval_dataset = TrOCRLineDataset(
        eval_samples, processor, tokenizer, max_target_length=args.max_target_length
    )

    data_collator = TrOCRDataCollator(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        save_total_limit=3,
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_cer_metrics(tokenizer),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
