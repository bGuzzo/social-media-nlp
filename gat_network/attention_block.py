

import torch
# import torch_geometric
from torch_geometric.nn import GATConv
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import LayerNorm


class DeepAttnBlock(torch.nn.Module):
    def __init__(
        self, 
        num_levels: int, 
        num_heads: int, 
        hidden_channels: int, 
        dropout_prob: float = 0.5
    ):
        super().__init__()
        
        if num_levels < 1:
            raise ValueError(f"At least one attention layers is required, got {num_levels}")
        
        self.attention_levels: list[GATConv] = []
        self.dropout_levels: list[Dropout] = []
        # self.lin_layer: list[Linear] = []
        self.norm_levels: list[LayerNorm] = []
        
        self.num_levels = num_levels
        
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
            # Apply GAT layer
            x_att = self.attention_levels[i](x, edge_index)
            
            # Add & Norm
            x_norm = self.norm_levels[i](x + x_att)
            
            # Dropout
            x = self.dropout_levels[i](x_norm)
        
        return x