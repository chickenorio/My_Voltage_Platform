import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime
import mlflow
from tqdm import tqdm
import optuna
from .models.base_model import BaseVoltageModel
from .models.set_transformer import SetTransformerModel
# Import other model classes as needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_datasets(
    df: pd.DataFrame,
    window_size: int,
    target_col: str = 'voltage',
    feature_cols: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create PyTorch datasets from DataFrame.
    
    Args:
        df: Input DataFrame
        window_size: Size of sliding window
        target_col: Name of target column
        feature_cols: List of feature column names
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Create sequences
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[feature_cols].iloc[i-window_size:i].values)
        y.append(df[target_col].iloc[i])
    
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).reshape(-1, 1)
    
    # Split data
    n = len(X)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return (
        TensorDataset(X_train, y_train),
        TensorDataset(X_val, y_val),
        TensorDataset(X_test, y_test)
    )

def train_model(
    model: BaseVoltageModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    experiment_name: str = "voltage_prediction"
) -> BaseVoltageModel:
    """
    Train a model with MLflow tracking.
    
    Args:
        model: Model instance to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        experiment_name: Name for MLflow experiment
        
    Returns:
        Trained model
    """
    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Setup training
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        best_val_loss = float('inf')
        best_model_path = None
        early_stop_count = 0
        
        # Training loop
        for epoch in range(config['epochs']):
            # Training phase
            model.train()
            train_losses = []
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
                for batch in pbar:
                    optimizer.zero_grad()
                    loss = model.training_step(batch)
                    loss.backward()
                    
                    # Optional gradient clipping
                    if 'clip_grad' in config:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config['clip_grad']
                        )
                    
                    optimizer.step()
                    train_losses.append(loss.item())
                    pbar.set_postfix({'loss': np.mean(train_losses)})
            
            # Validation phase
            model.eval()
            val_metrics = []
            
            with torch.no_grad():
                for batch in val_loader:
                    metrics = model.validation_step(batch)
                    val_metrics.append(metrics)
            
            # Average validation metrics
            avg_val_metrics = {
                k: np.mean([m[k] for m in val_metrics])
                for k in val_metrics[0].keys()
            }
            
            # Update learning rate
            scheduler.step(avg_val_metrics['val_loss'])
            
            # Log metrics
            mlflow.log_metrics(
                {
                    'train_loss': np.mean(train_losses),
                    **avg_val_metrics
                },
                step=epoch
            )
            
            # Save best model
            if avg_val_metrics['val_loss'] < best_val_loss:
                best_val_loss = avg_val_metrics['val_loss']
                best_model_path = Path(f"../models/{model.__class__.__name__}_best.pt")
                model.save_checkpoint(best_model_path, optimizer, epoch)
                mlflow.log_artifact(str(best_model_path))
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            # Early stopping
            if early_stop_count >= config.get('early_stop_patience', 10):
                logger.info("Early stopping triggered")
                break
            
            logger.info(
                f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, "
                f"val_loss={avg_val_metrics['val_loss']:.4f}"
            )
        
        # Load best model
        if best_model_path is not None:
            model.load_checkpoint(best_model_path)
        
        return model

def objective(
    trial: optuna.Trial,
    model_class: type,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_config: Dict[str, Any]
) -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        model_class: Model class to optimize
        train_loader: Training data loader
        val_loader: Validation data loader
        base_config: Base configuration to modify
        
    Returns:
        Validation loss
    """
    # Define hyperparameter search space
    config = base_config.copy()
    
    # Common hyperparameters
    config.update({
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    })
    
    # Model-specific hyperparameters
    if model_class == SetTransformerModel:
        config.update({
            'hidden_dim': trial.suggest_categorical(
                'hidden_dim',
                [64, 128, 256]
            ),
            'num_heads': trial.suggest_categorical(
                'num_heads',
                [4, 8]
            ),
            'num_layers': trial.suggest_int('num_layers', 2, 4)
        })
    
    # Initialize and train model
    model = model_class(config)
    model = train_model(
        model,
        train_loader,
        val_loader,
        config,
        experiment_name=f"hpo_{model_class.__name__}"
    )
    
    # Return best validation loss
    return min(m['val_loss'] for m in model.training_history)

def optimize_hyperparameters(
    model_class: type,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_config: Dict[str, Any],
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Optimize model hyperparameters using Optuna.
    
    Args:
        model_class: Model class to optimize
        train_loader: Training data loader
        val_loader: Validation data loader
        base_config: Base configuration
        n_trials: Number of optimization trials
        
    Returns:
        Best hyperparameters
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(
            trial,
            model_class,
            train_loader,
            val_loader,
            base_config
        ),
        n_trials=n_trials
    )
    
    logger.info(f"Best trial: {study.best_trial.params}")
    return {**base_config, **study.best_trial.params}

def run_all(
    models: Optional[List[str]] = None,
    data_path: Optional[Path] = None,
    config_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    Train multiple models and compare results.
    
    Args:
        models: List of model names to train
        data_path: Path to data file
        config_path: Path to config file
        
    Returns:
        Dictionary of model names and their best validation losses
    """
    # Load data
    if data_path is None:
        data_path = Path("../data/processed/voltage_data.csv")
    df = pd.read_csv(data_path)
    
    # Load config
    if config_path is None:
        config_path = Path("../config/models.yaml")
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    # Get available models
    model_map = {
        'set_transformer': SetTransformerModel,
        # Add other models here
    }
    
    if models is None:
        models = list(model_map.keys())
    
    # Create datasets
    train_ds, val_ds, test_ds = make_datasets(df, window_size=24)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        pin_memory=True
    )
    
    results = {}
    
    for model_name in models:
        if model_name not in model_map:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        logger.info(f"\nTraining {model_name}...")
        model_class = model_map[model_name]
        config = all_configs[model_name]
        
        # Train model
        model = model_class(config)
        model = train_model(
            model,
            train_loader,
            val_loader,
            config,
            experiment_name=model_name
        )
        
        # Evaluate on test set
        model.eval()
        test_metrics = []
        with torch.no_grad():
            for batch in test_loader:
                metrics = model.validation_step(batch)
                test_metrics.append(metrics)
        
        avg_test_metrics = {
            k: np.mean([m[k] for m in test_metrics])
            for k in test_metrics[0].keys()
        }
        
        results[model_name] = avg_test_metrics
        logger.info(f"{model_name} test metrics: {avg_test_metrics}")
    
    return results

if __name__ == "__main__":
    import pandas as pd
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Train voltage prediction models")
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to train"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to data file"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Perform hyperparameter optimization"
    )
    
    args = parser.parse_args()
    
    try:
        models = args.models.split(",") if args.models else None
        data_path = Path(args.data) if args.data else None
        config_path = Path(args.config) if args.config else None
        
        if args.optimize:
            # Example hyperparameter optimization
            df = pd.read_csv(data_path or "../data/processed/voltage_data.csv")
            train_ds, val_ds, _ = make_datasets(df, window_size=24)
            
            train_loader = DataLoader(
                train_ds,
                batch_size=32,
                shuffle=True,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=32,
                pin_memory=True
            )
            
            with open(config_path or "../config/models.yaml", 'r') as f:
                base_config = yaml.safe_load(f)['set_transformer']
            
            best_params = optimize_hyperparameters(
                SetTransformerModel,
                train_loader,
                val_loader,
                base_config
            )
            
            logger.info(f"Best hyperparameters: {best_params}")
        else:
            # Train models
            results = run_all(models, data_path, config_path)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = Path(f"../models/results_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"\nResults saved to {results_path}")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}") 