import yaml
import mlflow
import os
from pathlib import Path
import logging

class MLflowSetup:
    def __init__(self, config_path):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self._setup_mlflow()
    
    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded config with tracking URI: {config['mlflow']['tracking_uri']}")
                return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
    
    def _setup_mlflow(self):
        try:
            # Get tracking URI from config
            tracking_uri = self.config['mlflow']['tracking_uri']
            
            # Create the mlruns directory if it doesn't exist
            mlruns_dir = os.path.dirname(tracking_uri.replace('sqlite:///', ''))
            os.makedirs(mlruns_dir, exist_ok=True)
            
            # Set the tracking URI
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            
            # Set up experiment
            experiment_name = self.config['mlflow']['experiment_name']
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.logger.info(f"Experiment '{experiment_name}' does not exist. Creating it.")
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow experiment set to: {experiment_name}")
            
        except Exception as e:
            self.logger.error(f"Error setting up MLflow: {e}")
            raise
    
    def get_config(self):
        return self.config
