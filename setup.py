import os
from pathlib import Path
import subprocess
import logging
import yaml
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create project directory structure."""
    directories = [
        "data/raw",
        "data/processed",
        "data/processed/reports",
        "data/processed/validation",
        "src/data_ingest",
        "src/profiling",
        "src/missing_handler",
        "src/anomaly_detector",
        "src/feature_engineering",
        "src/correlation",
        "src/data_validation",
        "src/ai_models/models",
        "src/visualization",
        "ui",
        "notebooks",
        "config",
        "models",
        "models/history",
        "models/scalers"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create __init__.py files
    src_dirs = [d for d in directories if d.startswith("src")]
    for directory in src_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch()
        logger.info(f"Created {init_file}")

def initialize_dvc():
    """Initialize DVC and configure for data versioning."""
    try:
        # Initialize DVC
        subprocess.run(["dvc", "init"], check=True)
        logger.info("Initialized DVC")
        
        # Add data directory to DVC
        subprocess.run(
            ["dvc", "add", "data/raw"],
            check=True
        )
        logger.info("Added data/raw to DVC")
        
        # Create .gitignore for data
        with open("data/raw/.gitignore", "w") as f:
            f.write("*\n")
        logger.info("Created .gitignore for data directory")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize DVC: {str(e)}")
        raise

def create_config_files():
    """Create configuration files."""
    # Model configuration
    models_config = {
        "set_transformer": {
            "hidden_dim": 128,
            "num_heads": 4,
            "num_layers": 3,
            "dropout": 0.1,
            "lr": 1e-4,
            "batch_size": 64,
            "epochs": 50,
            "device": "cuda"
        }
    }
    
    # Preprocessing configuration
    preprocessing_config = {
        "missing_values": {
            "voltage": {
                "strategy": "interpolate",
                "method": "time",
                "limit": 24  # hours
            },
            "current": {
                "strategy": "model_impute",
                "method": "knn",
                "n_neighbors": 5
            },
            "power": {
                "strategy": "forward_fill",
                "limit": 12  # hours
            }
        },
        "anomaly_detection": {
            "statistical": {
                "z_score_threshold": 3,
                "rolling_window": 24  # hours
            },
            "isolation_forest": {
                "contamination": 0.1,
                "random_state": 42
            },
            "lof": {
                "n_neighbors": 20,
                "contamination": 0.1
            }
        },
        "feature_engineering": {
            "scaling": {
                "voltage": "standard",
                "current": "robust",
                "power": "minmax"
            },
            "time_features": [
                "hour_sin_cos",
                "day_of_week",
                "month",
                "is_weekend",
                "is_holiday"
            ],
            "window_aggregations": {
                "sizes": [6, 12, 24],  # hours
                "functions": [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "range"
                ]
            }
        },
        "validation": {
            "voltage": {
                "min": 180,
                "max": 260
            },
            "current": {
                "min": 0,
                "max": 500
            },
            "power": {
                "min": 0
            },
            "unique_combinations": [
                ["timestamp", "id_line"]
            ]
        }
    }
    
    # Save configurations
    with open("config/models.yaml", "w") as f:
        yaml.dump(models_config, f, default_flow_style=False)
    logger.info("Created models.yaml")
    
    with open("config/preprocessing.yaml", "w") as f:
        yaml.dump(preprocessing_config, f, default_flow_style=False)
    logger.info("Created preprocessing.yaml")

def create_sample_data():
    """Create sample voltage network data."""
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    n_samples = 1000
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_samples,
        freq="H"
    )
    
    # Base voltage with daily pattern
    voltage = 230 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    # Add noise
    voltage += np.random.normal(0, 2, n_samples)
    
    # Current with similar pattern
    current = 100 + 50 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    current += np.random.normal(0, 5, n_samples)
    current = np.maximum(current, 0)  # Ensure non-negative
    
    # Power = voltage * current (approximately)
    power = voltage * current * 0.95  # 95% power factor
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "voltage": voltage,
        "current": current,
        "power": power,
        "id_line": "LINE_001"
    })
    
    # Add some missing values
    mask = np.random.random(n_samples) < 0.05
    df.loc[mask, ["voltage", "current", "power"]] = np.nan
    
    # Add some anomalies
    anomaly_mask = np.random.random(n_samples) < 0.02
    df.loc[anomaly_mask, "voltage"] *= 1.5
    
    # Save to CSV
    df.to_csv("data/raw/sample.csv", index=False)
    logger.info("Created sample.csv")

def main():
    """Initialize project structure and configuration."""
    try:
        # Create directory structure
        create_directory_structure()
        
        # Initialize DVC
        initialize_dvc()
        
        # Create configuration files
        create_config_files()
        
        # Create sample data
        create_sample_data()
        
        logger.info("Project initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Project initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 