import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import mlflow
from scipy import stats
import logging
from typing import Tuple, List, Dict


class DataValidator:
    def __init__(self):
        self.expected_weather_cols = ['timestamp', 'temperature', 'humidity', 
                                    'pressure', 'wind_speed']
        self.expected_pollution_cols = ['timestamp', 'aqi', 'co', 'no2', 'o3', 
                                      'pm2_5', 'pm10']
        self.validation_results = {'passed': True, 'warnings': [], 'errors': []}

    def validate_datasets(self, weather_df: pd.DataFrame, pollution_df: pd.DataFrame) -> dict:
        """Validate both weather and pollution datasets"""
        # Check if dataframes are empty
        if weather_df.empty or pollution_df.empty:
            self.validation_results['passed'] = False
            self.validation_results['errors'].append("Empty dataframe detected")
            return self.validation_results

        # Validate columns
        self._validate_columns(weather_df, 'weather')
        self._validate_columns(pollution_df, 'pollution')

        # Validate data types and ranges
        self._validate_data_types(weather_df, pollution_df)
        
        # Validate time continuity
        self._validate_time_continuity(weather_df, pollution_df)

        return self.validation_results

    def _validate_columns(self, df: pd.DataFrame, data_type: str):
        """Validate column presence and naming"""
        expected_cols = (self.expected_weather_cols if data_type == 'weather' 
                        else self.expected_pollution_cols)
        
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            self.validation_results['passed'] = False
            self.validation_results['errors'].append(
                f"Missing columns in {data_type} data: {missing_cols}"
            )

    def _validate_data_types(self, weather_df: pd.DataFrame, pollution_df: pd.DataFrame):
        """Validate data types and basic ranges"""
        # Check timestamp type
        for df, name in [(weather_df, 'weather'), (pollution_df, 'pollution')]:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                self.validation_results['warnings'].append(
                    f"Timestamp in {name} data is not datetime type"
                )

    def _validate_time_continuity(self, weather_df: pd.DataFrame, pollution_df: pd.DataFrame):
        """Validate time series continuity and alignment"""
        # Check for time gaps
        for df, name in [(weather_df, 'weather'), (pollution_df, 'pollution')]:
            time_diff = df['timestamp'].diff().dropna()
            large_gaps = time_diff[time_diff > pd.Timedelta(hours=2)]
            if not large_gaps.empty:
                self.validation_results['warnings'].append(
                    f"Found {len(large_gaps)} gaps > 2 hours in {name} data"
                )


class EnhancedDataPreprocessor:
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.validator = DataValidator()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using forward fill and backward fill for time series"""
        try:
            # Sort by timestamp first
            df = df.sort_values('timestamp')
            
            # Forward fill then backward fill
            df = df.ffill().bfill()
            
            self.logger.info("Missing values handled successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            raise

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using rolling median instead of z-score for time series"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'aqi':  # Preserve AQI values
                    # Calculate rolling median and std
                    rolling_median = df[col].rolling(window=24, center=True, min_periods=1).median()
                    rolling_std = df[col].rolling(window=24, center=True, min_periods=1).std()
                    
                    # Define bounds
                    lower_bound = rolling_median - 3 * rolling_std
                    upper_bound = rolling_median + 3 * rolling_std
                    
                    # Replace outliers with rolling median
                    mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    df.loc[mask, col] = rolling_median[mask]
            
            return df
        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}")
            raise

    def preprocess(self, weather_df: pd.DataFrame, pollution_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data specifically for time series analysis"""
        try:
            # Validate data
            validation_results = self.validator.validate_datasets(weather_df, pollution_df)
            if not validation_results['passed']:
                raise ValueError(f"Data validation failed: {validation_results['errors']}")

            # Convert timestamps
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            pollution_df['timestamp'] = pd.to_datetime(pollution_df['timestamp'])

            # Sort by timestamp
            weather_df = weather_df.sort_values('timestamp')
            pollution_df = pollution_df.sort_values('timestamp')

            # Handle missing values
            weather_df = self.handle_missing_values(weather_df)
            pollution_df = self.handle_missing_values(pollution_df)

            # Handle outliers
            weather_df = self.handle_outliers(weather_df)
            pollution_df = self.handle_outliers(pollution_df)

            # Merge datasets with proper timestamp alignment
            merged_df = pd.merge_asof(
                pollution_df,
                weather_df,
                on='timestamp',
                tolerance=pd.Timedelta('1h'),
                direction='nearest'
            )

            # Set timestamp as index
            merged_df.set_index('timestamp', inplace=True)

            # Fill any remaining missing values
            merged_df = merged_df.ffill().bfill()

            # Scale only the weather features (not AQI)
            weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
            merged_df[weather_features] = self.scaler.fit_transform(merged_df[weather_features])

            self.logger.info(f"Final dataset shape: {merged_df.shape}")
            self.logger.info(f"Time range: {merged_df.index.min()} to {merged_df.index.max()}")
            
            return merged_df

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

    def aggregate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates data to daily frequency.
        - Takes the mean for numeric columns while preserving non-numeric data.
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)

            # Separate numeric and non-numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            non_numeric_data = data.select_dtypes(exclude=[np.number])

            # Resample numeric data and take mean
            numeric_aggregated = numeric_data.resample('D').mean()

            # For non-numeric data, take the first valid value per day
            non_numeric_aggregated = non_numeric_data.resample('D').first()

            # Combine numeric and non-numeric data back together
            aggregated_data = pd.concat([numeric_aggregated, non_numeric_aggregated], axis=1)

            # Reset the index
            aggregated_data.reset_index(inplace=True)

            self.logger.info("Data successfully aggregated to daily frequency.")
            return aggregated_data
        except Exception as e:
            self.logger.error(f"Error in data aggregation: {e}")
            raise


    def load_data(self, weather_folder: str, pollution_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and combine multiple JSON files"""
        try:
            # Load weather data
            weather_files = list(Path(weather_folder).glob('*.json'))
            weather_dfs = []
            for file in weather_files:
                try:
                    df = pd.read_json(file)
                    weather_dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading weather file {file}: {e}")
            
            weather_df = pd.concat(weather_dfs, ignore_index=True)
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            weather_df = weather_df.sort_values('timestamp')

            # Load pollution data
            pollution_files = list(Path(pollution_folder).glob('*.json'))
            pollution_dfs = []
            for file in pollution_files:
                try:
                    df = pd.read_json(file)
                    pollution_dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading pollution file {file}: {e}")
            
            pollution_df = pd.concat(pollution_dfs, ignore_index=True)
            pollution_df['timestamp'] = pd.to_datetime(pollution_df['timestamp'])
            pollution_df = pollution_df.sort_values('timestamp')

            return weather_df, pollution_df

        except Exception as e:
            self.logger.error(f"Error in data loading: {e}")
            raise

    def check_for_missing_or_inf(self, df: pd.DataFrame, context: str) -> None:
        """Log NaN and Inf counts for debugging."""
        nan_counts = df.isna().sum()
        inf_counts = (df == float('inf')).sum()
        neg_inf_counts = (df == float('-inf')).sum()
        
        self.logger.info(f"[{context}] NaN counts per column:\n{nan_counts}")
        self.logger.info(f"[{context}] +Inf counts per column:\n{inf_counts}")
        self.logger.info(f"[{context}] -Inf counts per column:\n{neg_inf_counts}")

    
if __name__ == "__main__":
    try:
        preprocessor = EnhancedDataPreprocessor()
        
        # Load data
        weather_df, pollution_df = preprocessor.load_data(
            weather_folder="E:\\Semester 8\\MLOps\\Project_Task_1\\data\\weather",
            pollution_folder="E:\\Semester 8\\MLOps\\Project_Task_1\\data\\pollution"
        )
        
        # Preprocess data
        processed_df = preprocessor.preprocess(weather_df, pollution_df)
        
        print("Processed data shape:", processed_df.shape)
        print("\nMissing values after processing:")
        print(processed_df.isnull().sum())
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
