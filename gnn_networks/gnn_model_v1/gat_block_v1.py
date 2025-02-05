

import torch
from torch_geometric.nn import GATConv
from torch.nn import Dropout

class DeepGATBlockV1(torch.nn.Module):
    def __init__(
        self, 
        num_levels: int, 
        hidden_channels: int, 
        num_heads: int = 4, 
        dropout_prob: float = 0.5,
        activation_func: torch.nn.Module = torch.nn.ReLU(),
        norm_func_class: torch.nn.Module = torch.nn.LayerNorm
    ):
        super().__init__()
        
        if num_levels < 1:
            raise ValueError(f"At least one attention layers is required, got {num_levels}")
        
        if hidden_channels <= 0:
            raise ValueError(f"Hidden channels must be greater than 0, got {hidden_channels}") 
        
        # Module params
        self.num_levels = num_levels
        self.activation_func = activation_func
        
        # Graph attention submodule
        self.attention_levels: list[GATConv] = []
        self.att_dropout_levels: list[Dropout] = []
        self.att_norm_levels: list[torch.nn.Module] = []
        
        # Feed-Forward sub module
        self.lin_layer_1: list[torch.nn.Linear] = []
        self.lin_layer_2: list[torch.nn.Linear] = []
        self.lin_dropout_levels: list[Dropout] = []
        self.lin_norm_levels: list[torch.nn.Module] = []
        
        # Overall level output dropout (helps on neep archietcture)
        self.out_dropout = Dropout(dropout_prob)
        
        # Create layers
        for _ in range(num_levels):
            
            # Build graph attention submodule instances
            self.attention_levels.append(GATConv(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                heads=num_heads, 
                dropout=dropout_prob, 
                concat=False
            ))
            self.att_dropout_levels.append(Dropout(dropout_prob))
            self.att_norm_levels.append(norm_func_class(hidden_channels))
            
            # Build Feed-Forward submodule instances
            self.lin_layer_1.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lin_layer_2.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lin_dropout_levels.append(Dropout(dropout_prob))
            self.lin_norm_levels.append(norm_func_class(hidden_channels))
        
    def forward(self, x, edge_index):
        for i in range(self.num_levels):
            # Apply GAT message passing layer & dropout
            x_att = self.attention_levels[i](x, edge_index)
            x_att = self.att_dropout_levels[i](x_att)
            
            # Add & Norm GAT sub-layer
            x = self.att_norm_levels[i](x + x_att)
            
            # Apply double Feed-Forward linear trasformation with activation function
            x_ff1 = self.lin_layer_1[i](x)
            x_ff1 = self.activation_func(x_ff1)
            x_ff2 = self.lin_layer_2[i](x_ff1)
            x_ff2 = self.lin_dropout_levels[i](x_ff2)
            
            # Add & Norm Feed-Forward sub-layer
            x = self.lin_norm_levels[i](x + x_ff2)
        
        x = self.out_dropout(x)
        return x