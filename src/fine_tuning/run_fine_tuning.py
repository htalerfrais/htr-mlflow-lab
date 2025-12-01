#!/usr/bin/env python
"""
Main script to run fine-tuning experiments
"""

import sys
import yaml
import mlflow
from pathlib import Path
from argparse import ArgumentParser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fine_tuning.trocr_trainer import TrOCRFineTuner


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_fine_tuning(config_path: str):
    """
    Run fine-tuning experiment from config file
    
    Args:
        config_path: Path to YAML config file
    """
    print("="*60)
    print("TrOCR Fine-Tuning")
    print("="*60)
    
    # Load config
    config = load_config(config_path)
    print(f"\nConfig: {config_path}")
    
    # Extract sections
    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})
    mlflow_config = config.get('mlflow', {})
    output_config = config.get('output', {})
    
    # Setup MLflow
    mlflow_uri = mlflow_config.get('tracking_uri')
    experiment_name = mlflow_config.get('experiment_name', 'TrOCR-FineTuning')
    
    # Start MLflow run
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=config.get('experiment', {}).get('name', 'trocr-finetune')):
        
        # Log config
        mlflow.log_params({
            'config_file': config_path,
            'model_name': model_config.get('pretrained_model_name'),
            'dataset': dataset_config.get('name'),
        })
        
        # Initialize trainer
        trainer = TrOCRFineTuner(
            model_name=model_config.get('pretrained_model_name', 'microsoft/trocr-base-handwritten'),
            output_dir=output_config.get('dir', './fine_tuned_trocr'),
            mlflow_tracking_uri=mlflow_uri,
            mlflow_experiment_name=experiment_name,
        )
        
        # Setup model
        trainer.setup_model(**model_config.get('params', {}))
        
        # Prepare dataset
        trainer.prepare_dataset(
            dataset_name=dataset_config.get('name', 'Teklia/IAM-line'),
            train_split=dataset_config.get('train_split', 'train'),
            eval_split=dataset_config.get('eval_split', 'validation'),
            max_samples=dataset_config.get('max_samples'),
        )
        
        # Train
        results = trainer.train(**training_config)
        
        # Save model
        save_path = output_config.get('dir', './fine_tuned_trocr')
        trainer.save_model(save_path)
        
        print("\n✅ Fine-tuning complete!")
        print(f"Model saved to: {save_path}")
        
        if mlflow_uri:
            print(f"MLflow: {mlflow_uri}")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return results


def main():
    """Main entry point"""
    parser = ArgumentParser(description='Fine-tune TrOCR model')
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML config file'
    )
    
    args = parser.parse_args()
    
    # Check config exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Run fine-tuning
    try:
        run_fine_tuning(args.config)
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
