import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, Tuple, Optional
import mlflow

class ArimaModel:
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 exog_columns: Optional[list] = None):
        self.order = order
        self.exog_columns = exog_columns
        self.model = None
        self.fitted_model = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        """
        try:
            # Ensure arrays are 1D
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Calculate MAE
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate R-squared
            r2 = r2_score(y_true, y_pred)
            
            # Calculate accuracy (within 10% of true value)
            percentage_errors = np.abs(y_true - y_pred) / np.where(y_true != 0, np.abs(y_true), 1)
            accuracy = np.mean(percentage_errors <= 0.10) * 100  # Convert to percentage
            
            self.logger.info(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Accuracy: {accuracy:.2f}%")
            
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'accuracy': float(accuracy)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise

    def fit(self, endog: pd.Series, exog: Optional[pd.DataFrame] = None) -> Dict:
        """
        Fit the ARIMA model and log metrics.
        """
        try:
            # Start an MLflow run
            with mlflow.start_run(nested=True) as run:
                self.logger.info(f"Started MLflow run: {run.info.run_id}")
                
                # Convert to numpy array if needed
                if isinstance(endog, pd.Series):
                    endog = endog.values
                if isinstance(exog, pd.DataFrame):
                    exog = exog.values
                
                self.logger.info(f"Endog shape: {endog.shape}")
                if exog is not None:
                    self.logger.info(f"Exog shape: {exog.shape}")

                # Initialize and fit the model
                self.model = ARIMA(endog, exog=exog, order=self.order)
                self.logger.info("Fitting ARIMA model...")
                self.fitted_model = self.model.fit()

                # Get predictions for the training period
                predictions = self.fitted_model.predict(start=0, end=len(endog)-1, exog=exog)
                
                # Calculate metrics
                train_metrics = self.calculate_metrics(endog, predictions)
                
                # Add model information metrics
                train_metrics.update({
                    'aic': float(self.fitted_model.aic),
                    'bic': float(self.fitted_model.bic)
                })

                # Log parameters
                mlflow.log_params({
                    'order_p': self.order[0],
                    'order_d': self.order[1],
                    'order_q': self.order[2],
                    'n_exog_variables': 0 if exog is None else exog.shape[1]
                })

                # Log metrics
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                
                self.logger.info("Training metrics logged to MLflow")
                return train_metrics

        except Exception as e:
            self.logger.error(f"Error in fitting ARIMA model: {e}")
            raise

    def predict(self, test_data: pd.DataFrame, forecast_horizon: int = 24) -> Tuple[np.ndarray, Dict]:
        """
        Generate forecasts and log test metrics.
        """
        try:
            with mlflow.start_run(nested=True) as run:
                self.logger.info(f"Started MLflow prediction run: {run.info.run_id}")
                
                target, exog = self.prepare_data(test_data)
                
                # Convert to numpy arrays
                if isinstance(target, pd.Series):
                    target = target.values
                if isinstance(exog, pd.DataFrame):
                    exog = exog.values
                
                # Generate predictions
                predictions = self.fitted_model.forecast(
                    steps=forecast_horizon,
                    exog=exog[:forecast_horizon] if exog is not None else None
                )
                
                # Calculate metrics
                test_metrics = self.calculate_metrics(
                    target[:forecast_horizon], 
                    predictions
                )
                
                # Log test metrics
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                
                self.logger.info("Test metrics logged to MLflow")
                return predictions, test_metrics
                
        except Exception as e:
            self.logger.error(f"Error in making predictions: {e}")
            raise