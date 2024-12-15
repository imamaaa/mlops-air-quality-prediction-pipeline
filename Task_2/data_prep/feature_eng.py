from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import logging
from pathlib import Path

class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str = 'aqi', 
                 time_windows: List[int] = [3, 6, 12, 24],
                 lag_windows: List[int] = [1, 3, 6, 12, 24]):
        """
        Initialize feature engineering pipeline
        
        Args:
            target_col: Target column for prediction
            time_windows: Windows for rolling statistics (in hours)
            lag_windows: Windows for lag features (in hours)
        """
        self.target_col = target_col
        self.time_windows = time_windows
        self.lag_windows = lag_windows
        self.feature_stats = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Store feature statistics"""
        try:
            # Store statistics for future reference
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            self.feature_stats = {
                col: {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max()
                } for col in numeric_cols
            }
            
            # Log feature statistics with MLflow
            with mlflow.start_run(nested=True):
                for col, stats in self.feature_stats.items():
                    for stat_name, value in stats.items():
                        mlflow.log_metric(f"{col}_{stat_name}", value)
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error in fit: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features"""
        try:
            X_transformed = X.copy()
            
            # Add all feature groups
            X_transformed = self.add_time_features(X_transformed)
            X_transformed = self.add_lag_features(X_transformed)
            X_transformed = self.add_rolling_features(X_transformed)
            X_transformed = self.add_interaction_features(X_transformed)
            X_transformed = self.add_rate_of_change_features(X_transformed)
            
            # Log feature creation with MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_param("n_features_created", 
                               len(X_transformed.columns) - len(X.columns))
            
            return X_transformed
            
        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            raise

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features"""
        try:
            # First ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                else:
                    raise ValueError("No timestamp column or datetime index found")

            # Extract time components first
            df['hour'] = df.index.hour
            df['day'] = df.index.day
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month

            # Then create cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
            
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
            
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            
            # Optionally drop the original time components if not needed
            df = df.drop(['hour', 'day', 'day_of_week', 'month'], axis=1)
            
            return df
                
        except Exception as e:
            self.logger.error(f"Error in add_time_features: {e}")
            raise

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series"""
        # Add lags for target variable
        for lag in self.lag_windows:
            df[f'{self.target_col}_lag_{lag}h'] = df[self.target_col].shift(lag)
        
        # Add lags for important predictors
        important_predictors = ['temperature', 'humidity', 'wind_speed']
        for col in important_predictors:
            if col in df.columns:
                df[f'{col}_lag_1h'] = df[col].shift(1)
                df[f'{col}_lag_24h'] = df[col].shift(24)
        
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        # Add rolling statistics for target
        for window in self.time_windows:
            df[f'{self.target_col}_rolling_mean_{window}h'] = (
                df[self.target_col].rolling(window=window, min_periods=1).mean()
            )
            df[f'{self.target_col}_rolling_std_{window}h'] = (
                df[self.target_col].rolling(window=window, min_periods=1).std()
            )
            df[f'{self.target_col}_rolling_max_{window}h'] = (
                df[self.target_col].rolling(window=window, min_periods=1).max()
            )
        
        # Add rolling features for weather parameters
        weather_params = ['temperature', 'humidity', 'wind_speed']
        for param in weather_params:
            if param in df.columns:
                df[f'{param}_rolling_mean_24h'] = (
                    df[param].rolling(window=24, min_periods=1).mean()
                )
        
        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between variables"""
        # Weather interactions
        if all(col in df.columns for col in ['temperature', 'humidity']):
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        # Pollution interactions
        pollution_params = ['pm2_5', 'pm10', 'no2', 'o3']
        for param in pollution_params:
            if param in df.columns:
                df[f'{self.target_col}_{param}_ratio'] = df[self.target_col] / df[param]
        
        return df

    def add_rate_of_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change features"""
        try:
            # Calculate rate of change for target
            df[f'{self.target_col}_rate_of_change'] = df[self.target_col].diff()
            
            # Calculate rate of change for weather parameters
            weather_params = ['temperature', 'humidity', 'wind_speed', 'pressure']
            for param in weather_params:
                if param in df.columns:
                    df[f'{param}_rate_of_change'] = df[param].diff()
                    # Add percentage change
                    df[f'{param}_pct_change'] = df[param].pct_change().fillna(0)  # Fill initial NaN with 0
            
            # Calculate rate of change for pollution parameters
            pollution_params = ['co', 'no2', 'o3', 'pm2_5', 'pm10']
            for param in pollution_params:
                if param in df.columns:
                    df[f'{param}_rate_of_change'] = df[param].diff()
                    # Fill missing values before calculating pct_change
                    df[param] = df[param].fillna(0)  # Replace NaN with 0 or any other value suitable for your dataset
                    df[f'{param}_pct_change'] = df[param].pct_change(fill_method=None)  # Disable padding

            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in add_rate_of_change_features: {e}")
            raise

    def prepare_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare final features for modeling"""
        try:
            # Remove rows with NaN values created by lag features
            df_clean = df.dropna()
            
            # Separate features and target
            target = df_clean[self.target_col]
            features = df_clean.drop([self.target_col, 'timestamp'], axis=1)
            
            # Log feature preparation metrics
            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    "final_features_count": features.shape[1],
                    "samples_after_cleaning": features.shape[0]
                })
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error in prepare_for_modeling: {e}")
            raise

if __name__ == "__main__":
    try:
        # Initialize preprocessor
        from preprocess import EnhancedDataPreprocessor
        preprocessor = EnhancedDataPreprocessor()
        
        # Load and preprocess data
        weather_df, pollution_df = preprocessor.load_data(
            weather_folder="E:\\Semester 8\\MLOps\\Project_Task_1\\data\\weather",
            pollution_folder="E:\\Semester 8\\MLOps\\Project_Task_1\\data\\pollution"
        )
        processed_df = preprocessor.preprocess(weather_df, pollution_df)
        
        # Initialize feature engineer
        feature_engineer = TimeSeriesFeatureEngineer(
            target_col='aqi',
            time_windows=[3, 6, 12, 24],
            lag_windows=[1, 3, 6, 12, 24]
        )
        
        # Fit and transform features
        feature_engineer.fit(processed_df)
        engineered_df = feature_engineer.transform(processed_df)
        
        # Prepare for modeling
        X, y = feature_engineer.prepare_for_modeling(engineered_df)
        
        # Print summary
        print("\nFeature Engineering Summary:")
        print(f"Number of features created: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        print("\nFeature list:")
        for col in X.columns:
            print(f"- {col}")
        
        # Save feature statistics
        feature_stats_df = pd.DataFrame.from_dict(
            feature_engineer.feature_stats,
            orient='index'
        )
        feature_stats_df.to_csv('feature_statistics.csv')
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        raise
