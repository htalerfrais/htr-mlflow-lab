"""
Base trainer class for fine-tuning OCR models
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow


class BaseFineTuner(ABC):
    """
    Abstract base class for fine-tuning OCR models
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base fine-tuner
        
        Args:
            model_name: Name/path of pretrained model
            output_dir: Directory to save fine-tuned model
            mlflow_tracking_uri: MLflow tracking server URI
            mlflow_experiment_name: MLflow experiment name
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MLflow setup
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)
    
    @abstractmethod
    def prepare_dataset(self, dataset_name: str, **kwargs) -> Any:
        """
        Load and prepare dataset for fine-tuning
        
        Args:
            dataset_name: Name of dataset to load
            **kwargs: Additional dataset parameters
            
        Returns:
            Prepared dataset
        """
        pass
    
    @abstractmethod
    def setup_model(self, **kwargs) -> None:
        """
        Load and configure model for fine-tuning
        
        Args:
            **kwargs: Model configuration parameters
        """
        pass
    
    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Run fine-tuning training loop
        
        Args:
            **kwargs: Training parameters
            
        Returns:
            Training metrics and results
        """
        pass
    
    @abstractmethod
    def evaluate(self, eval_dataset: Any) -> Dict[str, float]:
        """
        Evaluate fine-tuned model
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        pass
    
    def save_model(self, save_path: Optional[str] = None) -> None:
        """
        Save fine-tuned model
        
        Args:
            save_path: Path to save model (defaults to output_dir)
        """
        if save_path is None:
            save_path = self.output_dir
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self._save_model_impl(save_path)
        print(f"✅ Model saved to {save_path}")
    
    @abstractmethod
    def _save_model_impl(self, save_path: Path) -> None:
        """
        Implementation-specific model saving
        
        Args:
            save_path: Path to save model
        """
        pass
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional training step
        """
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"⚠️  Failed to log metrics to MLflow: {e}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
        except Exception as e:
            print(f"⚠️  Failed to log params to MLflow: {e}")
