"""
TrOCR fine-tuning trainer
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import mlflow

from .base_trainer import BaseFineTuner
from .utils import TrOCRDataCollator, compute_metrics


class TrOCRFineTuner(BaseFineTuner):
    """
    Fine-tuner for TrOCR models on handwriting recognition tasks
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        output_dir: str = "./fine_tuned_trocr",
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = "TrOCR-FineTuning",
        **kwargs
    ):
        """
        Initialize TrOCR fine-tuner
        
        Args:
            model_name: Pretrained TrOCR model name
            output_dir: Directory to save fine-tuned model
            mlflow_tracking_uri: MLflow tracking server URI
            mlflow_experiment_name: MLflow experiment name
        """
        super().__init__(
            model_name=model_name,
            output_dir=output_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            **kwargs
        )
        
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup_model(self, **kwargs) -> None:
        """
        Load and configure TrOCR model for fine-tuning
        """
        print(f"Loading TrOCR model: {self.model_name}")
        
        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        
        # Set special tokens
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Configure generation
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = kwargs.get('max_length', 64)
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = kwargs.get('num_beams', 4)
        
        print("✅ Model and processor loaded")
    
    def prepare_dataset(
        self,
        dataset_name: str = "Teklia/IAM-line",
        train_split: str = "train",
        eval_split: str = "validation",
        max_samples: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Load and prepare IAM dataset for fine-tuning
        
        Args:
            dataset_name: HuggingFace dataset name
            train_split: Name of training split
            eval_split: Name of evaluation split
            max_samples: Maximum samples per split (for testing)
        """
        print(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Select splits
        train_data = dataset[train_split]
        eval_data = dataset[eval_split]
        
        # Limit samples if specified
        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
            eval_data = eval_data.select(range(min(max_samples, len(eval_data))))
        
        print(f"Train samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data)}")
        
        # Preprocess datasets
        self.train_dataset = train_data.map(
            self._preprocess_function,
            remove_columns=train_data.column_names,
            desc="Processing train dataset"
        )
        
        self.eval_dataset = eval_data.map(
            self._preprocess_function,
            remove_columns=eval_data.column_names,
            desc="Processing eval dataset"
        )
        
        print("✅ Datasets prepared")
    
    def _preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single example
        
        Args:
            examples: Raw example from dataset
            
        Returns:
            Processed example with pixel_values and labels
        """
        # Process image
        image = examples['image'].convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Encode text
        labels = self.processor.tokenizer(
            examples['text'],
            padding="max_length",
            max_length=self.model.config.max_length,
            truncation=True,
        ).input_ids
        
        # Replace pad token id with -100 (ignored by loss)
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]
        
        return {
            'pixel_values': pixel_values.squeeze(),
            'labels': labels
        }
    
    def train(
        self,
        num_train_epochs: int = 5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        logging_steps: int = 100,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "cer",
        greater_is_better: bool = False,
        fp16: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run fine-tuning training
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size
            per_device_eval_batch_size: Evaluation batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            evaluation_strategy: When to evaluate
            save_strategy: When to save checkpoints
            logging_steps: Logging frequency
            load_best_model_at_end: Load best model after training
            metric_for_best_model: Metric to track for best model
            greater_is_better: Whether higher metric is better
            fp16: Use mixed precision training
            
        Returns:
            Training metrics
        """
        if self.model is None or self.train_dataset is None:
            raise ValueError("Model and dataset must be set up first")
        
        print("\n" + "="*60)
        print("Starting TrOCR Fine-Tuning")
        print("="*60)
        
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training device: {device}")
        
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            fp16 = fp16 and torch.cuda.is_available()
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy=evaluation_strategy,  # Changed from evaluation_strategy
            save_strategy=save_strategy,
            logging_steps=logging_steps,
            logging_dir=str(self.output_dir / "logs"),
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,
            predict_with_generate=True,
            report_to=["mlflow"] if mlflow.active_run() else [],
            **kwargs
        )
        
        # Log parameters to MLflow
        self.log_params({
            "model_name": self.model_name,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "batch_size": per_device_train_batch_size,
            "train_samples": len(self.train_dataset),
            "eval_samples": len(self.eval_dataset),
        })
        
        # Data collator
        data_collator = TrOCRDataCollator(processor=self.processor)
        
        # Metrics function for Trainer
        def compute_metrics_fn(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            return compute_metrics(pred_ids, label_ids, self.processor)
        
        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
        
        # Train
        print("\n🚀 Starting training...")
        train_result = trainer.train()
        
        # Evaluate
        print("\n📊 Evaluating...")
        eval_metrics = trainer.evaluate()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Final CER: {eval_metrics.get('eval_cer', 'N/A'):.4f}")
        print(f"Final WER: {eval_metrics.get('eval_wer', 'N/A'):.4f}")
        print("="*60 + "\n")
        
        # Log final metrics
        self.log_metrics(eval_metrics)
        
        return {
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics
        }
    
    def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate fine-tuned model
        
        Args:
            eval_dataset: Optional evaluation dataset (uses self.eval_dataset if None)
            
        Returns:
            Evaluation metrics
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            raise ValueError("No evaluation dataset available")
        
        print("Evaluating model...")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=8,
            predict_with_generate=True,
        )
        
        data_collator = TrOCRDataCollator(processor=self.processor)
        
        def compute_metrics_fn(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            return compute_metrics(pred_ids, label_ids, self.processor)
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
        
        metrics = trainer.evaluate()
        
        print(f"CER: {metrics.get('eval_cer', 'N/A'):.4f}")
        print(f"WER: {metrics.get('eval_wer', 'N/A'):.4f}")
        
        return metrics
    
    def _save_model_impl(self, save_path: Path) -> None:
        """
        Save fine-tuned TrOCR model
        
        Args:
            save_path: Path to save model
        """
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        # Also log to MLflow
        try:
            mlflow.pytorch.log_model(self.model, "model")
        except Exception as e:
            print(f"⚠️  Failed to log model to MLflow: {e}")
