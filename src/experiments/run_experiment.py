# src/experiments/run_experiment.py
import yaml
import mlflow
import logging
import sys
import os

"""Ensure the project root is on sys.path so that 'from src.*' imports work
when running this file directly (e.g., `python src/experiments/run_experiment.py`).
This adds the parent directory of `src/` to sys.path.
"""
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders.line_loader import load_iam_lines
from src.pipelines.line_to_text import run_pipeline
from src.utils.git_utils import get_current_git_commit
from src.utils.metrics import calculate_cer, calculate_wer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def run_experiment(config_path: str):
    # first we load the configs
    config = load_config(config_path)
    
    # set the mlflow tracking uri to the remote server we created
    mlflow.set_tracking_uri("http://13.60.230.97:5000/")
    logger.info("Connected to MLflow tracking server")
    
    # instantiate experiment (get or create)
    # if the experiùent already exists, we get the already ewisting id 
    experiment_name = config.get("experiment_name", "Default Experiment")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to setup experiment: {e}")
        raise
    

    # ---------------------------------
    # Start MLflow run
    # ---------------------------------

    run_name = config.get("run_name", "unnamed_run")
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        try:
            # Log git commit
            try:
                git_commit = get_current_git_commit() # from the utils functions
                mlflow.log_param("git_commit", git_commit)
            except Exception as e:
                logger.warning(f"Could not get git commit: {e}")
                mlflow.log_param("git_commit", "unknown")
            
            # log all configuration parameters from the dict key and subkeys
            for key, value in config.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_param(f"{key}.{sub_key}", sub_value)
                else:
                    mlflow.log_param(key, value)
            
            logger.info("Logged configuration parameters to MLflow")
            
            # run the pipeline based on configuration
            pipeline_name = config.get("pipeline", "line_to_text")
            
            if pipeline_name == "line_to_text":
                metrics = run_pipeline(config)
            else:
                raise ValueError(f"Unknown pipeline: {pipeline_name}")
            
            # Log metrics (metrics is returned by run_pipeline from pipeline/)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Experiment completed successfully. Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise


# entrypoint , with args config_file
if __name__ == "__main__":
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
