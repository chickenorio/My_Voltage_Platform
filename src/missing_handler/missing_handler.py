import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict
from pathlib import Path
import logging
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingHandler:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize MissingHandler with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.imputation_history = []
    
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('missing_values', {})
    
    def fill_constant(
        self,
        df: pd.DataFrame,
        col: str,
        value: Union[int, float, str]
    ) -> pd.DataFrame:
        """Fill missing values with a constant value."""
        df = df.copy()
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(value)
            self._log_imputation(col, 'constant', missing_count, value)
        return df
    
    def fill_forward(
        self,
        df: pd.DataFrame,
        col: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fill missing values using forward fill strategy."""
        df = df.copy()
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(method='ffill', limit=limit)
            self._log_imputation(col, 'forward_fill', missing_count, limit=limit)
        return df
    
    def fill_backward(
        self,
        df: pd.DataFrame,
        col: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fill missing values using backward fill strategy."""
        df = df.copy()
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(method='bfill', limit=limit)
            self._log_imputation(col, 'backward_fill', missing_count, limit=limit)
        return df
    
    def interpolate(
        self,
        df: pd.DataFrame,
        col: str,
        method: str = 'time',
        limit: Optional[int] = None,
        order: int = 2
    ) -> pd.DataFrame:
        """
        Interpolate missing values using various methods.
        
        Args:
            df: Input DataFrame
            col: Column to interpolate
            method: Interpolation method ('linear', 'time', 'polynomial', etc.)
            limit: Maximum number of consecutive NaNs to fill
            order: Order of polynomial interpolation
        """
        df = df.copy()
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if method == 'polynomial':
                df[col] = df[col].interpolate(
                    method=method,
                    order=order,
                    limit=limit
                )
            else:
                df[col] = df[col].interpolate(
                    method=method,
                    limit=limit
                )
            self._log_imputation(
                col, f'interpolate_{method}',
                missing_count,
                limit=limit,
                order=order
            )
        return df
    
    def model_impute(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        method: str = 'knn',
        **kwargs
    ) -> pd.DataFrame:
        """
        Impute missing values using ML models.
        
        Args:
            df: Input DataFrame
            target_col: Column to impute
            feature_cols: Columns to use as features
            method: Imputation method ('knn' or 'rf')
            **kwargs: Additional parameters for the imputer
        """
        df = df.copy()
        missing_count = df[target_col].isna().sum()
        
        if missing_count == 0:
            return df
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Prepare feature matrix
        X = df[feature_cols]
        
        if method == 'knn':
            imputer = KNNImputer(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'uniform')
            )
            # Impute features first if needed
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Now impute target
            df[target_col] = imputer.fit_transform(
                df[[target_col]].join(X_imputed)
            )[:, 0]
            
        elif method == 'rf':
            rf = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                random_state=kwargs.get('random_state', 42)
            )
            
            # Train on non-missing data
            mask = ~df[target_col].isna()
            rf.fit(X[mask], df[target_col][mask])
            
            # Predict missing values
            missing_mask = df[target_col].isna()
            df.loc[missing_mask, target_col] = rf.predict(X[missing_mask])
        
        self._log_imputation(
            target_col,
            f'model_{method}',
            missing_count,
            **kwargs
        )
        return df
    
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
    
    def apply_config_strategy(
        self,
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """
        Apply the imputation strategy specified in the config for a column.
        
        Args:
            df: Input DataFrame
            column: Column to impute
        """
        if column not in self.config:
            logger.warning(f"No imputation strategy found for column: {column}")
            return df
        
        strategy = self.config[column]
        method = strategy['strategy']
        
        if method == 'interpolate':
            return self.interpolate(
                df,
                column,
                method=strategy.get('method', 'time'),
                limit=strategy.get('limit')
            )
        elif method == 'model_impute':
            return self.model_impute(
                df,
                column,
                method=strategy.get('method', 'knn'),
                **strategy.get('parameters', {})
            )
        elif method == 'forward_fill':
            return self.fill_forward(
                df,
                column,
                limit=strategy.get('limit')
            )
        elif method == 'backward_fill':
            return self.fill_backward(
                df,
                column,
                limit=strategy.get('limit')
            )
        else:
            logger.warning(f"Unknown imputation strategy: {method}")
            return df

if __name__ == "__main__":
    # Example usage
    try:
        from ..data_ingest.data_loader import load_csv
        
        # Load sample data
        df = load_csv("../data/raw/sample.csv")
        
        # Initialize handler with config
        handler = MissingHandler("../config/preprocessing.yaml")
        
        # Apply different imputation strategies
        df = handler.fill_constant(df, 'categorical_col', 'UNKNOWN')
        df = handler.interpolate(df, 'voltage', method='time')
        df = handler.model_impute(df, 'current', method='knn')
        
        # Get imputation summary
        summary = handler.get_imputation_summary()
        print("\nImputation Summary:")
        print(summary)
        
    except Exception as e:
        logger.error(f"Failed to run imputation example: {str(e)}") 