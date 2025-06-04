import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict
from pathlib import Path
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingHandler:
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize MissingHandler with configuration.
        
        Args:
            config: Configuration dictionary for missing value handling
        """
        self.config = config or {}
        self.imputers = {}
        self.imputation_history = []
    
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('missing_values', {})
    
    def apply_config_strategy(
        self,
        df: pd.DataFrame,
        column: str,
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply missing value handling strategy based on configuration or specified method.
        
        Args:
            df: Input DataFrame
            column: Column to handle missing values for
            method: Optional method override
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        if method is None:
            method = self.config.get('default_method', 'interpolate')
        
        logger.info(f"Applying {method} to handle missing values in {column}")
        
        if method == 'interpolate':
            return self._interpolate(df, column)
        elif method == 'forward_fill':
            return self._forward_fill(df, column)
        elif method == 'backward_fill':
            return self._backward_fill(df, column)
        elif method == 'model_impute':
            return self._model_impute(df, column)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _interpolate(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Interpolate missing values."""
        df = df.copy()
        df[column] = df[column].interpolate(method='linear')
        return df
    
    def _forward_fill(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Forward fill missing values."""
        df = df.copy()
        df[column] = df[column].fillna(method='ffill')
        return df
    
    def _backward_fill(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Backward fill missing values."""
        df = df.copy()
        df[column] = df[column].fillna(method='bfill')
        return df
    
    def _model_impute(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Impute missing values using a machine learning model."""
        df = df.copy()
        
        # Get numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != column]
        
        if not feature_cols:
            logger.warning(f"No numeric features available for {column}, using simple imputation")
            return self._interpolate(df, column)
        
        # Split data into known and unknown values
        known_mask = df[column].notna()
        unknown_mask = df[column].isna()
        
        if not any(known_mask):
            logger.warning(f"No known values for {column}, using simple imputation")
            return self._interpolate(df, column)
        
        # Prepare data
        X_known = df.loc[known_mask, feature_cols]
        y_known = df.loc[known_mask, column]
        X_unknown = df.loc[unknown_mask, feature_cols]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_known, y_known)
        
        # Predict missing values
        if any(unknown_mask):
            predictions = model.predict(X_unknown)
            df.loc[unknown_mask, column] = predictions
        
        return df
    
    def get_missing_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Dictionary containing missing value statistics
        """
        missing_counts = df.isna().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        stats = {
            'total_rows': len(df),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'total_missing_cells': missing_counts.sum(),
            'missing_cell_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100
        }
        
        return stats
    
    def _log_imputation(
        self,
        column: str,
        method: str,
        missing_count: int,
        **kwargs
    ):
        """Log imputation operation details."""
        log_entry = {
            'column': column,
            'method': method,
            'missing_count': missing_count,
            'parameters': kwargs
        }
        self.imputation_history.append(log_entry)
        logger.info(
            f"Imputed {missing_count} missing values in {column} "
            f"using {method} method"
        )
    
    def get_imputation_summary(self) -> pd.DataFrame:
        """Get summary of all imputation operations."""
        return pd.DataFrame(self.imputation_history)

if __name__ == "__main__":
    # Example usage
    try:
        from ..data_ingest.data_loader import load_csv
        
        # Load sample data
        df = load_csv("../data/raw/sample.csv")
        
        # Initialize handler with config
        handler = MissingHandler("../config/preprocessing.yaml")
        
        # Apply different imputation strategies
        df = handler.apply_config_strategy(df, 'categorical_col')
        df = handler.apply_config_strategy(df, 'voltage')
        df = handler.apply_config_strategy(df, 'current')
        
        # Get imputation summary
        summary = handler.get_imputation_summary()
        print("\nImputation Summary:")
        print(summary)
        
    except Exception as e:
        logger.error(f"Failed to run imputation example: {str(e)}") 