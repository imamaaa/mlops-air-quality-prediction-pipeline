import mlflow
import argparse
import numpy as np
import pandas as pd
from utils.mlflow_setup import MLflowSetup
from data_prep.preprocess import EnhancedDataPreprocessor
from data_prep.feature_eng import TimeSeriesFeatureEngineer
from models.arima import ArimaModel
from models.lstm import LSTMModel
import logging
from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models(config):
    try:
        # Ensure no active runs before starting
        if mlflow.active_run():
            logger.warning("Ending any active MLflow run before starting.")
            mlflow.end_run()

        # Start parent run with a descriptive name
        run_name = f"{config['mlflow']['run_name_prefix']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            logger.info(f"Started parent run '{run_name}'.")

            # Log parent run parameters
            mlflow.log_params({
                "data_path_weather": config['data']['weather_path'],
                "data_path_pollution": config['data']['pollution_path'],
                "experiment_name": config['mlflow']['experiment_name']
            })

            # Load and preprocess data
            preprocessor = EnhancedDataPreprocessor()
            weather_df, pollution_df = preprocessor.load_data(
                config['data']['weather_path'],
                config['data']['pollution_path']
            )
            processed_df = preprocessor.preprocess(weather_df, pollution_df)

            # Feature engineering
            feature_engineer = TimeSeriesFeatureEngineer(
                target_col='aqi',
                time_windows=[3, 6, 12, 24],
                lag_windows=[1, 3, 6, 12]
            )
            final_data = feature_engineer.fit_transform(processed_df)

            # ARIMA Training in nested run
            with mlflow.start_run(run_name="arima_training", nested=True):
                logger.info("Started ARIMA nested run.")

                # Prepare ARIMA data
                target = final_data['aqi']
                exog_columns = config['model_params']['arima'].get('exog_columns', [])
                target = target.replace([float('inf'), float('-inf')], np.nan).dropna()

                if exog_columns:
                    exog = final_data[exog_columns].replace([float('inf'), float('-inf')], np.nan).dropna()
                    common_indices = target.index.intersection(exog.index)
                    target = target.loc[common_indices]
                    exog = exog.loc[common_indices]

                    logger.info(f"ARIMA - Target shape: {target.shape}")
                    logger.info(f"ARIMA - Exogenous shape: {exog.shape}")

                    # Train ARIMA
                    arima = ArimaModel(**config['model_params']['arima'])
                    arima_metrics = arima.fit(target, exog)
                else:
                    logger.info(f"ARIMA - Target shape: {target.shape}")
                    arima = ArimaModel(**config['model_params']['arima'])
                    arima_metrics = arima.fit(target, None)

                # Log ARIMA metrics
                mlflow.log_metrics({f"arima_{k}": v for k, v in arima_metrics.items()})
                logger.info(f"ARIMA metrics logged: {arima_metrics}")

            # LSTM Training in nested run
            with mlflow.start_run(run_name="lstm_training", nested=True):
                logger.info("Started LSTM nested run.")

                # Prepare LSTM data
                lstm_features = ['aqi'] + exog_columns
                X_lstm = final_data[lstm_features].values
                y_lstm = final_data['aqi'].values

                sequence_length = config['model_params']['lstm']['sequence_length']
                X_sequences = []
                y_sequences = []

                for i in range(len(X_lstm) - sequence_length):
                    X_sequences.append(X_lstm[i:(i + sequence_length)])
                    y_sequences.append(y_lstm[i + sequence_length])

                X_lstm = np.array(X_sequences)
                y_lstm = np.array(y_sequences)

                logger.info(f"LSTM - Input shape: {X_lstm.shape}")
                logger.info(f"LSTM - Target shape: {y_lstm.shape}")

                # Train LSTM
                lstm = LSTMModel(**config['model_params']['lstm'])
                lstm_metrics = lstm.fit(
                    X_lstm,
                    y_lstm,
                    validation_split=0.1,
                    epochs=15,
                    batch_size=2
                )

                # Log LSTM metrics
                mlflow.log_metrics({f"lstm_{k}": v for k, v in lstm_metrics.items()})
                logger.info(f"LSTM metrics logged: {lstm_metrics}")

            logger.info("Training completed successfully.")

    except Exception as e:
        logger.error(f"Error in training models: {e}")
        raise
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        logger.info("All MLflow runs properly ended.")

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='config/config.yaml')
        args = parser.parse_args()

        # Setup MLflow
        mlflow_setup = MLflowSetup(args.config)
        config = mlflow_setup.get_config()

        # Set tracking URI from config
        tracking_uri = config['mlflow']['tracking_uri']
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

        # Set up experiment
        experiment_name = config['mlflow']['experiment_name']
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment '{experiment_name}'")
        mlflow.set_experiment(experiment_name)

        # Train models
        train_models(config)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        logger.info("MLflow run ended.")

if __name__ == "__main__":
    main()
