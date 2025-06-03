import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from .base_model import BaseVoltageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Multi-head attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query tensor [batch_size, seq_len_q, dim]
            k: Key tensor [batch_size, seq_len_k, dim]
            v: Value tensor [batch_size, seq_len_v, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len_q, dim]
        """
        batch_size = q.size(0)
        
        # Project and reshape
        q = self.q_proj(q).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(k).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(v).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.dim
        )
        out = self.out_proj(out)
        
        return out

class SetAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Set Attention Block for processing sets of vectors.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiheadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Set Attention Block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x)
        x = x + residual
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

class SetTransformerModel(BaseVoltageModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Set Transformer model for voltage network analysis.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Model parameters
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Input embedding
        self.input_proj = nn.Linear(
            config.get('input_dim', 5),  # Default: voltage, current, power, etc.
            self.hidden_dim
        )
        
        # Set Attention Blocks
        self.blocks = nn.ModuleList([
            SetAttentionBlock(
                self.hidden_dim,
                self.num_heads,
                self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)  # Predict voltage
        )
        
        self.to(self.device)
        logger.info(
            f"Initialized SetTransformerModel with "
            f"{self.get_parameter_count()['total']} parameters"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Set Transformer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, 1]
        """
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Apply Set Attention Blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to output
        x = self.output_proj(x)
        
        return x
    
    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with optional voltage-specific weighting.
        
        Args:
            y_pred: Model predictions
            y_true: Ground truth values
            
        Returns:
            Loss value
        """
        # Base MSE loss
        mse_loss = nn.MSELoss()(y_pred, y_true)
        
        # Optional: Add penalty for predictions outside valid voltage range
        voltage_penalty = torch.mean(
            torch.relu(180 - y_pred) + torch.relu(y_pred - 260)
        )
        
        return mse_loss + 0.1 * voltage_penalty

if __name__ == "__main__":
    # Example usage
    try:
        # Create dummy data
        batch_size, seq_len, input_dim = 32, 24, 5
        x = torch.randn(batch_size, seq_len, input_dim)
        y = torch.randn(batch_size, seq_len, 1)
        
        # Initialize model
        config = {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1,
            'input_dim': input_dim,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        model = SetTransformerModel(config)
        print(f"\nModel Architecture:")
        print(model)
        
        # Test forward pass
        y_pred = model(x)
        print(f"\nOutput shape: {y_pred.shape}")
        
        # Test loss computation
        loss = model.compute_loss(y_pred, y)
        print(f"Loss value: {loss.item():.4f}")
        
    except Exception as e:
        logger.error(f"Failed to run SetTransformerModel example: {str(e)}") 