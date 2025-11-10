"""Run experiments with MLflow tracking."""

import yaml
import mlflow
import logging
import sys
import os
from typing import Dict, Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.git_utils import get_current_git_commit
from src.pipelines.factory import PipelineFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = "http://13.60.230.97:5000/"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def setup_mlflow_experiment(experiment_name: str) -> str:
    """Get or create MLflow experiment and return its ID."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Connected to MLflow tracking server")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name}")
    
    return experiment_id


def log_config_to_mlflow(config: Dict[str, Any]) -> None:
    """Log configuration parameters and git commit to MLflow."""
    # Log git commit
    try:
        git_commit = get_current_git_commit()
        mlflow.log_param("git_commit", git_commit)
    except Exception as e:
        logger.warning(f"Could not get git commit: {e}")
        mlflow.log_param("git_commit", "unknown")
    
    # Log all configuration parameters
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                mlflow.log_param(f"{key}.{sub_key}", sub_value)
        else:
            mlflow.log_param(key, value)
    
    logger.info("Logged configuration parameters to MLflow")


def log_metrics_to_mlflow(metrics: Dict[str, Any]) -> None:
    """Log metrics to MLflow, handling special cases like dataset_info."""
    for metric_name, metric_value in metrics.items():
        if metric_name == "dataset_info":
            # Log dataset information as parameters
            for key, value in metric_value.items():
                mlflow.log_param(f"dataset.{key}", value)
        else:
            # Log regular metrics
            mlflow.log_metric(metric_name, metric_value)
    
    logger.info(f"Logged metrics to MLflow: {list(metrics.keys())}")


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create and run the pipeline, returning metrics."""
    pipeline_name = config.get("pipeline", "line_to_text")
    pipeline = PipelineFactory.create(pipeline_name, config)
    
    logger.info("Running pipeline: %s", pipeline.get_name())
    metrics = pipeline.run()
    
    return metrics


def run_experiment(config_path: str) -> None:
    """Main experiment execution function."""
    config = load_config(config_path)
    
    experiment_name = config.get("experiment_name", "Default Experiment")
    experiment_id = setup_mlflow_experiment(experiment_name)
    
    run_name = config.get("run_name", "unnamed_run")
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        try:
            log_config_to_mlflow(config)
            metrics = run_pipeline(config)
            log_metrics_to_mlflow(metrics)
            
            logger.info(f"Experiment completed successfully. Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise


def main() -> None:
    """Entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)
    
    try:
        run_experiment(config_file)
        print("Experiment completed successfully!")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
