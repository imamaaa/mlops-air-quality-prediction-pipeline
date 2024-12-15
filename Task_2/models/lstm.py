import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, Tuple, List, Optional
import mlflow

class LSTMModel:
    def __init__(self,
                 sequence_length: int = 24,
                 n_features: Optional[int] = None,
                 units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def build_model(self):
        try:
            self.model = Sequential()
            
            # First LSTM layer
            self.model.add(LSTM(
                units=self.units[0],
                activation='relu',
                return_sequences=len(self.units) > 1,
                input_shape=(self.sequence_length, self.n_features)
            ))
            self.model.add(Dropout(self.dropout_rate))
            
            # Additional LSTM layers
            for i in range(1, len(self.units)):
                self.model.add(LSTM(
                    units=self.units[i],
                    activation='relu',
                    return_sequences=i < len(self.units) - 1
                ))
                self.model.add(Dropout(self.dropout_rate))
            
            # Output layer
            self.model.add(Dense(1))
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info("LSTM model built successfully")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {e}")
            raise

def fit(self, X_train: np.ndarray, y_train: np.ndarray,
        validation_split: float = 0.1,
        epochs: int = 20,
        batch_size: int = 2) -> Dict:
    try:
        with mlflow.start_run(nested=True) as run:
            self.logger.info(f"Started MLflow run for LSTM: {run.info.run_id}")
            
            if self.n_features is None:
                self.n_features = X_train.shape[2]
            
            # Build model if not already built
            if self.model is None:
                self.build_model()
            
            # Log model parameters
            mlflow.log_params({
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'units': self.units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'batch_size': batch_size,
                'epochs': epochs
            })
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Calculate and log training metrics
            train_pred = self.model.predict(X_train)
            train_metrics = self.calculate_metrics(y_train, train_pred.flatten())
            
            # Log metrics
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            
            # Log validation metrics if available
            if history.history.get('val_loss'):
                mlflow.log_metrics({
                    'val_loss': float(history.history['val_loss'][-1]),
                    'val_mae': float(history.history['val_mae'][-1])
                })
            
            # Save model artifacts
            mlflow.tensorflow.log_model(self.model, "lstm_model")
            
            # Save history as JSON
            mlflow.log_dict(history.history, "training_history.json")
            
            return train_metrics
            
    except Exception as e:
        self.logger.error(f"Error in fitting LSTM model: {e}")
        raise

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate all evaluation metrics"""
        try:
            # Ensure arrays are 1D
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            
            # Calculate RMSE
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            
            # Calculate MAE
            mae = float(mean_absolute_error(y_true, y_pred))
            
            # Calculate R2
            r2 = float(r2_score(y_true, y_pred))
            
            # Calculate accuracy (predictions within 10% of actual values)
            percentage_errors = np.abs(y_true - y_pred) / np.where(y_true != 0, np.abs(y_true), 1)
            accuracy = float(np.mean(percentage_errors <= 0.10) * 100)  # Convert to percentage
            
            self.logger.info(f"LSTM Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Accuracy: {accuracy:.2f}%")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise

    