import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseVoltageModel(nn.Module):
    """Base class for all voltage network models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        self.training_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        raise NotImplementedError(
            "Forward pass must be implemented by subclass"
        )
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Loss value
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        
        # Log metrics
        metrics = self.compute_metrics(y_pred, y)
        self._log_training_step(loss.item(), metrics)
        
        return loss
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform single validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of validation metrics
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            y_pred = self(x)
            loss = self.compute_loss(y_pred, y)
            metrics = self.compute_metrics(y_pred, y)
            metrics['val_loss'] = loss.item()
        
        return metrics
    
    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            y_pred: Model predictions
            y_true: Ground truth values
            
        Returns:
            Loss value
        """
        return nn.MSELoss()(y_pred, y_true)
    
    def compute_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_pred: Model predictions
            y_true: Ground truth values
            
        Returns:
            Dictionary of metric names and values
        """
        with torch.no_grad():
            # Mean Absolute Error
            mae = nn.L1Loss()(y_pred, y_true).item()
            
            # Mean Squared Error
            mse = nn.MSELoss()(y_pred, y_true).item()
            
            # Root Mean Squared Error
            rmse = torch.sqrt(torch.tensor(mse)).item()
            
            # Mean Absolute Percentage Error
            mape = torch.mean(
                torch.abs((y_true - y_pred) / y_true)
            ).item() * 100
            
            # R-squared
            ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
            ss_res = torch.sum((y_true - y_pred) ** 2)
            r2 = (1 - ss_res / ss_tot).item()
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            }
    
    def _log_training_step(
        self,
        loss: float,
        metrics: Dict[str, float]
    ):
        """Log training step information."""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'loss': loss,
            **metrics
        }
        self.training_history.append(log_entry)
    
    def save_checkpoint(
        self,
        filepath: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            optimizer: Optional optimizer to save state
            epoch: Optional epoch number
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(
        self,
        filepath: Path,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Optional[int]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            optimizer: Optional optimizer to load state
            
        Returns:
            Optional epoch number
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.training_history = checkpoint.get('training_history', [])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint.get('epoch')
    
    def export_training_history(
        self,
        output_dir: Path = Path("../models/history")
    ) -> Path:
        """
        Export training history to JSON file.
        
        Args:
            output_dir: Directory to save history
            
        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"training_history_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Exported training history to {filepath}")
        return filepath
    
    @classmethod
    def from_config(
        cls,
        config_path: Path
    ) -> 'BaseVoltageModel':
        """
        Create model instance from configuration file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Returns:
            Instantiated model
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get count of trainable and non-trainable parameters."""
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        non_trainable = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        return {
            'trainable': trainable,
            'non_trainable': non_trainable,
            'total': trainable + non_trainable
        } 