

import torch
from torch_geometric.nn import GATConv
from torch.nn import Dropout
from torch.nn import LayerNorm


class DeepAttnBlock(torch.nn.Module):
    def __init__(
        self, 
        num_levels: int, 
        hidden_channels: int, 
        num_heads: int = 1, 
        dropout_prob: float = 0.5
    ):
        super().__init__()
        
        if num_levels < 1:
            raise ValueError(f"At least one attention layers is required, got {num_levels}")
        
        if hidden_channels <= 0:
            raise ValueError(f"Hidden channels must be greater than 0, got {hidden_channels}") 
        
        self.num_levels = num_levels
        self.attention_levels: list[GATConv] = []
        self.dropout_levels: list[Dropout] = []
        self.norm_levels: list[LayerNorm] = []
        self.activation_func = torch.nn.ReLU()
        
        # Create layers
        for _ in range(num_levels):
            self.attention_levels.append(GATConv(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                heads=num_heads, 
                dropout=dropout_prob, 
            ))
            self.dropout_levels.append(Dropout(dropout_prob))
            # self.lin_layer.append(Linear(hidden_channels, hidden_channels))
            self.norm_levels.append(LayerNorm(hidden_channels))
        
        
    def forward(self, x, edge_index):
        for i in range(self.num_levels):
            # Apply GAT message passing layer
            x_att = self.attention_levels[i](x, edge_index)
            
            # Add & Norm
            x_norm = self.norm_levels[i](x + x_att)
            
            # Dropout
            x = self.dropout_levels[i](x_norm)
            
            # Use activation function for non-linearity
            x = self.activation_func(x)
            
        return x