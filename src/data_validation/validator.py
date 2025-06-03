import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path
import logging
import yaml
from datetime import datetime
import great_expectations as ge
import pandera as pa
from pandera.typing import Series

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoltageDataSchema(pa.SchemaModel):
    """Pandera schema for voltage network data."""
    
    timestamp: Series[pd.Timestamp]
    voltage: Series[float] = pa.Field(ge=180, le=260)
    current: Series[float] = pa.Field(ge=0, le=500)
    power: Series[float] = pa.Field(ge=0)
    id_line: Series[str]
    
    class Config:
        coerce = True
        strict = True

class DataValidator:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DataValidator with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.validation_history = []
        
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('validation', {})
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: Optional[pa.SchemaModel] = None
    ) -> Tuple[bool, Dict]:
        """
        Validate DataFrame against Pandera schema.
        
        Args:
            df: DataFrame to validate
            schema: Optional custom schema (uses VoltageDataSchema by default)
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        if schema is None:
            schema = VoltageDataSchema
        
        try:
            schema.validate(df)
            results = {'is_valid': True, 'errors': None}
            logger.info("Schema validation passed")
        except pa.errors.SchemaError as e:
            results = {
                'is_valid': False,
                'errors': str(e),
                'failure_cases': e.failure_cases.to_dict()
            }
            logger.warning(f"Schema validation failed: {str(e)}")
        
        self._log_validation('schema', results)
        return results['is_valid'], results
    
    def validate_with_expectations(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Validate DataFrame using Great Expectations.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        ge_df = ge.from_pandas(df)
        results = {'expectations': []}
        all_passed = True
        
        # Load validation rules from config
        rules = self.config
        
        # Validate numeric ranges
        for col, specs in rules.items():
            if isinstance(specs, dict) and 'min' in specs:
                exp_result = ge_df.expect_column_values_to_be_between(
                    col,
                    min_value=specs['min'],
                    max_value=specs.get('max'),
                    mostly=1 - specs.get('missing_threshold', 0.1)
                )
                results['expectations'].append(exp_result)
                all_passed &= exp_result.success
        
        # Validate unique combinations
        for combo in rules.get('unique_combinations', []):
            exp_result = ge_df.expect_compound_columns_to_be_unique(combo)
            results['expectations'].append(exp_result)
            all_passed &= exp_result.success
        
        # Additional validations
        results['expectations'].extend([
            # Check for missing values
            ge_df.expect_column_values_to_not_be_null('timestamp'),
            
            # Check timestamp order
            ge_df.expect_column_values_to_be_increasing('timestamp'),
            
            # Check for duplicate timestamps per line
            ge_df.expect_compound_columns_to_be_unique(
                ['timestamp', 'id_line']
            ),
            
            # Check power = voltage * current (approximately)
            ge_df.expect_column_pair_values_to_be_equal(
                'power',
                'voltage_current_product',
                condition=lambda x, y: abs(x - y) <= 0.1 * x,
                ignore_row_if='either_value_is_missing'
            )
        ])
        
        for exp_result in results['expectations']:
            all_passed &= exp_result.success
        
        self._log_validation('great_expectations', {
            'is_valid': all_passed,
            'results': results
        })
        
        return all_passed, results
    
    def validate_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.1
    ) -> Tuple[bool, Dict]:
        """
        Validate missing value proportions.
        
        Args:
            df: DataFrame to validate
            threshold: Maximum allowed proportion of missing values
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        missing_props = df.isnull().mean()
        exceeding_cols = missing_props[missing_props > threshold]
        
        results = {
            'is_valid': len(exceeding_cols) == 0,
            'missing_proportions': missing_props.to_dict(),
            'exceeding_threshold': exceeding_cols.to_dict()
        }
        
        self._log_validation('missing_values', results)
        return results['is_valid'], results
    
    def validate_time_continuity(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        max_gap: pd.Timedelta = pd.Timedelta(hours=1)
    ) -> Tuple[bool, Dict]:
        """
        Validate time series continuity.
        
        Args:
            df: DataFrame to validate
            timestamp_col: Name of timestamp column
            max_gap: Maximum allowed gap between measurements
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        df = df.sort_values(timestamp_col)
        gaps = df[timestamp_col].diff()
        large_gaps = gaps[gaps > max_gap]
        
        results = {
            'is_valid': len(large_gaps) == 0,
            'max_gap': gaps.max(),
            'gap_locations': large_gaps.index.tolist()
        }
        
        self._log_validation('time_continuity', results)
        return results['is_valid'], results
    
    def _log_validation(
        self,
        validation_type: str,
        results: Dict
    ):
        """Log validation operation."""
        log_entry = {
            'validation_type': validation_type,
            'results': results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.validation_history.append(log_entry)
        
        if results.get('is_valid', False):
            logger.info(f"{validation_type} validation passed")
        else:
            logger.warning(
                f"{validation_type} validation failed: "
                f"{results.get('errors', '')}"
            )
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of all validation operations."""
        return pd.DataFrame(self.validation_history)
    
    def export_validation_report(
        self,
        output_dir: Union[str, Path] = "../data/processed/validation"
    ) -> Path:
        """
        Export validation results to a report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"validation_report_{timestamp}.html"
        
        # Convert validation history to HTML report
        summary = self.get_validation_summary()
        html_content = f"""
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Data Validation Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            {summary.to_html()}
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Validation report saved to {report_path}")
        return report_path

if __name__ == "__main__":
    # Example usage
    try:
        from ..data_ingest.data_loader import load_csv
        
        # Load sample data
        df = load_csv("../data/raw/sample.csv")
        
        # Initialize validator with config
        validator = DataValidator("../config/preprocessing.yaml")
        
        # Run various validations
        schema_valid, schema_results = validator.validate_schema(df)
        ge_valid, ge_results = validator.validate_with_expectations(df)
        missing_valid, missing_results = validator.validate_missing_values(df)
        time_valid, time_results = validator.validate_time_continuity(df)
        
        # Export validation report
        report_path = validator.export_validation_report()
        print(f"\nValidation report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to run validation example: {str(e)}") 