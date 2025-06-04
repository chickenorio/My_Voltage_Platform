import pandas as pd
import chardet
import csv
from pathlib import Path
from typing import Union, Optional
import logging
import yaml
import io
from streamlit.runtime.uploaded_file_manager import UploadedFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_encoding(file_path: Union[str, Path, UploadedFile]) -> str:
    """Detect the encoding of a file using chardet."""
    if isinstance(file_path, UploadedFile):
        raw_data = file_path.getvalue()
        result = chardet.detect(raw_data)
        return result['encoding']
    else:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']

def detect_delimiter(file_path: Union[str, Path, UploadedFile], encoding: str) -> str:
    """Detect the delimiter of a CSV file."""
    if isinstance(file_path, UploadedFile):
        content = file_path.getvalue().decode(encoding)
        first_line = content.split('\n')[0]
    else:
        with open(file_path, 'r', encoding=encoding) as file:
            first_line = file.readline()
    
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(first_line)
    return dialect.delimiter

def load_csv(
    path: Union[str, Path, UploadedFile],
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load a CSV file with automatic encoding and delimiter detection.
    
    Args:
        path: Path to the CSV file or UploadedFile object from Streamlit
        encoding: Optional encoding override
        delimiter: Optional delimiter override
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        # Handle Streamlit UploadedFile
        if isinstance(path, UploadedFile):
            # Detect encoding if not provided
            if encoding is None:
                encoding = detect_encoding(path)
                logger.info(f"Detected encoding: {encoding}")
            
            # Detect delimiter if not provided
            if delimiter is None:
                delimiter = detect_delimiter(path, encoding)
                logger.info(f"Detected delimiter: {delimiter}")
            
            # Read the file content
            content = path.getvalue().decode(encoding)
            df = pd.read_csv(io.StringIO(content), delimiter=delimiter, **kwargs)
            logger.info(f"Successfully loaded uploaded file with shape {df.shape}")
            return df
        
        # Handle regular file path
        else:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            # Detect encoding if not provided
            if encoding is None:
                encoding = detect_encoding(path)
                logger.info(f"Detected encoding: {encoding}")
            
            # Detect delimiter if not provided
            if delimiter is None:
                delimiter = detect_delimiter(path, encoding)
                logger.info(f"Detected delimiter: {delimiter}")
            
            df = pd.read_csv(path, encoding=encoding, delimiter=delimiter, **kwargs)
            logger.info(f"Successfully loaded {path} with shape {df.shape}")
            return df
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_processed_data(
    df: pd.DataFrame,
    filename: str,
    base_path: Union[str, Path] = "../data/processed"
) -> Path:
    """
    Save processed DataFrame with timestamp versioning.
    
    Args:
        df: DataFrame to save
        filename: Base filename without extension
        base_path: Directory to save the file in
    
    Returns:
        Path: Path to the saved file
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    full_path = base_path / f"{filename}_{timestamp}.csv"
    
    df.to_csv(full_path, index=False)
    logger.info(f"Saved processed data to {full_path}")
    return full_path

def load_config(config_path: Union[str, Path]) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Example usage
    try:
        df = load_csv("../data/raw/sample.csv")
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to load sample data: {str(e)}") 