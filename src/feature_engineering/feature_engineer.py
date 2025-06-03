import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import yaml
from datetime import datetime
import calendar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize FeatureEngineer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.scalers = {}
        self.feature_history = []
        
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('feature_engineering', {})
    
    def scale(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        method: str = 'standard',
        **kwargs
    ) -> pd.DataFrame:
        """
        Scale numerical features using various methods.
        
        Args:
            df: Input DataFrame
            columns: Column(s) to scale
            method: Scaling method ('standard', 'robust', or 'minmax')
            **kwargs: Additional parameters for scalers
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if method == 'standard':
                scaler = StandardScaler(**kwargs)
            elif method == 'robust':
                scaler = RobustScaler(**kwargs)
            elif method == 'minmax':
                scaler = MinMaxScaler(**kwargs)
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Fit and transform
            df[f"{col}_scaled"] = scaler.fit_transform(
                df[col].values.reshape(-1, 1)
            ).flatten()
            
            # Store scaler for future use
            self.scalers[col] = scaler
            
            # Log operation
            self._log_feature_operation(
                'scaling',
                col,
                f"{col}_scaled",
                method=method,
                **kwargs
            )
        
        return df
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        method: str = 'onehot',
        max_categories: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            columns: Column(s) to encode
            method: Encoding method ('onehot' or 'label')
            max_categories: Maximum number of categories to encode
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if method == 'onehot':
                # Get value counts and optionally limit categories
                value_counts = df[col].value_counts()
                if max_categories:
                    categories = value_counts.nlargest(max_categories).index
                    other_mask = ~df[col].isin(categories)
                    if other_mask.any():
                        df.loc[other_mask, col] = 'Other'
                
                # Create dummy variables
                dummies = pd.get_dummies(
                    df[col],
                    prefix=col,
                    dummy_na=True
                )
                df = pd.concat([df, dummies], axis=1)
                
                # Log operation
                self._log_feature_operation(
                    'encoding',
                    col,
                    list(dummies.columns),
                    method='onehot',
                    n_categories=len(dummies.columns)
                )
                
            elif method == 'label':
                # Create a mapping of categories to integers
                categories = df[col].unique()
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                
                # Apply mapping
                df[f"{col}_encoded"] = df[col].map(mapping)
                
                # Log operation
                self._log_feature_operation(
                    'encoding',
                    col,
                    f"{col}_encoded",
                    method='label',
                    n_categories=len(categories)
                )
            
            else:
                raise ValueError(f"Unknown encoding method: {method}")
        
        return df
    
    def add_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add time-based features from timestamp column.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            features: List of features to add
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        if features is None:
            features = self.config.get('time_features', [
                'hour_sin_cos',
                'day_of_week',
                'month',
                'is_weekend',
                'is_holiday'
            ])
        
        for feature in features:
            if feature == 'hour_sin_cos':
                # Cyclical encoding of hour
                hours = df[timestamp_col].dt.hour
                df['hour_sin'] = np.sin(2 * np.pi * hours/24)
                df['hour_cos'] = np.cos(2 * np.pi * hours/24)
                
            elif feature == 'day_of_week':
                df['day_of_week'] = df[timestamp_col].dt.dayofweek
                
            elif feature == 'month':
                df['month'] = df[timestamp_col].dt.month
                
            elif feature == 'is_weekend':
                df['is_weekend'] = df[timestamp_col].dt.dayofweek.isin([5, 6])
                
            elif feature == 'is_holiday':
                # Simple holiday detection (weekends)
                df['is_holiday'] = df[timestamp_col].dt.dayofweek.isin([5, 6])
                
            self._log_feature_operation(
                'time_feature',
                timestamp_col,
                feature,
                method=feature
            )
        
        return df
    
    def window_agg(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        window_sizes: Union[int, List[int]],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Add rolling window aggregation features.
        
        Args:
            df: Input DataFrame
            columns: Column(s) to aggregate
            window_sizes: Window size(s) in number of periods
            functions: Aggregation functions to apply
            
        Returns:
            DataFrame with additional window features
        """
        df = df.copy()
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes]
        
        for col in columns:
            for window in window_sizes:
                rolling = df[col].rolling(window=window)
                
                for func in functions:
                    if func == 'mean':
                        df[f"{col}_roll{window}_mean"] = rolling.mean()
                    elif func == 'std':
                        df[f"{col}_roll{window}_std"] = rolling.std()
                    elif func == 'min':
                        df[f"{col}_roll{window}_min"] = rolling.min()
                    elif func == 'max':
                        df[f"{col}_roll{window}_max"] = rolling.max()
                    elif func == 'range':
                        df[f"{col}_roll{window}_range"] = (
                            rolling.max() - rolling.min()
                        )
                    
                    self._log_feature_operation(
                        'window_agg',
                        col,
                        f"{col}_roll{window}_{func}",
                        window_size=window,
                        function=func
                    )
        
        return df
    
    def _log_feature_operation(
        self,
        operation: str,
        input_col: str,
        output_col: Union[str, List[str]],
        **kwargs
    ):
        """Log feature engineering operation."""
        log_entry = {
            'operation': operation,
            'input_column': input_col,
            'output_column': output_col,
            'parameters': kwargs,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.feature_history.append(log_entry)
        logger.info(
            f"Applied {operation} to {input_col} -> "
            f"{output_col} with params {kwargs}"
        )
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all feature engineering operations."""
        return pd.DataFrame(self.feature_history)
    
    def save_scalers(
        self,
        output_dir: Union[str, Path] = "../models/scalers"
    ) -> Dict[str, Path]:
        """
        Save fitted scalers for future use.
        
        Args:
            output_dir: Directory to save scalers
            
        Returns:
            Dictionary mapping column names to scaler paths
        """
        from joblib import dump
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        for col, scaler in self.scalers.items():
            path = output_dir / f"{col}_scaler.joblib"
            dump(scaler, path)
            saved_paths[col] = path
            
        return saved_paths

if __name__ == "__main__":
    # Example usage
    try:
        from ..data_ingest.data_loader import load_csv
        
        # Load sample data
        df = load_csv("../data/raw/sample.csv")
        
        # Initialize engineer with config
        engineer = FeatureEngineer("../config/preprocessing.yaml")
        
        # Apply various transformations
        df = engineer.scale(df, ['voltage', 'current'], method='robust')
        df = engineer.add_time_features(df, 'timestamp')
        df = engineer.window_agg(
            df,
            'voltage',
            window_sizes=[6, 12, 24],
            functions=['mean', 'std']
        )
        
        # Get feature engineering summary
        summary = engineer.get_feature_summary()
        print("\nFeature Engineering Summary:")
        print(summary)
        
    except Exception as e:
        logger.error(f"Failed to run feature engineering example: {str(e)}") 