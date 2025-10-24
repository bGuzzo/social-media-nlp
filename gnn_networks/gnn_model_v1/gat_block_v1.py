"""
This script defines the `DeepGATBlockV1`, a fundamental building block for the `GatModelV1`.
This module encapsulates a deep, multi-level architecture inspired by the encoder layer of the
original Transformer model, but adapted for the graph domain. It is designed to process
graph-structured data and learn rich node representations through a series of attention
and feed-forward operations.

The `DeepGATBlockV1` consists of a stack of identical layers, each comprising two main sub-modules:

1.  **Multi-Head Graph Attention Sub-module**:
    - This sub-module uses a `GATConv` layer to perform multi-head graph attention, allowing
      the model to jointly attend to information from different representation subspaces.
    - A residual connection is employed, adding the output of the attention layer to its input,
      which helps in preventing the vanishing gradient problem in deep networks. This is followed
      by layer normalization.

2.  **Feed-Forward Network Sub-module**:
    - This sub-module is a standard two-layer feed-forward network with a ReLU activation
      function in between. It provides a non-linear transformation to the output of the
      attention sub-module.
    - Similar to the attention sub-module, it also includes a residual connection and layer
      normalization.

By stacking these layers, the `DeepGATBlockV1` can capture complex and long-range dependencies
within the graph. The final output of the block is passed through a dropout layer for
regularization.

This module is a key component of the `GatModelV1` and serves as the baseline for the more
advanced `DeepGATBlockV2`.
"""

import torch
from torch_geometric.nn import GATConv
from torch.nn import Dropout, Module, ModuleList, Linear

class DeepGATBlockV1(Module):
    """
    A deep Graph Attention Network (GAT) block, composed of multiple levels of GAT and feed-forward layers.

    Args:
        num_levels (int): The number of levels (i.e., stacked GAT and FFN blocks).
        hidden_channels (int): The number of hidden channels.
        num_heads (int, optional): The number of attention heads. Defaults to 4.
        dropout_prob (float, optional): The dropout probability. Defaults to 0.5.
        activation_func (Module, optional): The activation function to use. Defaults to torch.nn.ReLU().
        norm_func_class (Module, optional): The normalization function class to use. Defaults to torch.nn.LayerNorm.
    """
    def __init__(
        self, 
        num_levels: int, 
        hidden_channels: int, 
        num_heads: int = 4, 
        dropout_prob: float = 0.5,
        activation_func: Module = torch.nn.ReLU(),
        norm_func_class: Module = torch.nn.LayerNorm
    ):
        super().__init__()
        
        if num_levels < 1:
            raise ValueError(f"At least one attention layer is required, got {num_levels}")
        if hidden_channels <= 0:
            raise ValueError(f"Hidden channels must be greater than 0, got {hidden_channels}") 
        
        self.num_levels = num_levels
        self.activation_func = activation_func
        
        self.attention_levels = ModuleList()
        self.att_dropout_levels = ModuleList()
        self.att_norm_levels = ModuleList()
        
        self.lin_layer_1 = ModuleList()
        self.lin_layer_2 = ModuleList()
        self.lin_dropout_levels = ModuleList()
        self.lin_norm_levels = ModuleList()
        
        self.out_dropout = Dropout(dropout_prob)
        
        for _ in range(num_levels):
            self.attention_levels.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout_prob, concat=False))
            self.att_dropout_levels.append(Dropout(dropout_prob))
            self.att_norm_levels.append(norm_func_class(hidden_channels))
            
            self.lin_layer_1.append(Linear(hidden_channels, hidden_channels))
            self.lin_layer_2.append(Linear(hidden_channels, hidden_channels))
            self.lin_dropout_levels.append(Dropout(dropout_prob))
            self.lin_norm_levels.append(norm_func_class(hidden_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the DeepGATBlockV1.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge index of the graph.

        Returns:
            torch.Tensor: The output node features after processing through the block.
        """
        for i in range(self.num_levels):
            # Graph Attention sub-layer with residual connection and normalization
            x_att = self.attention_levels[i](x, edge_index)
            x_att = self.att_dropout_levels[i](x_att)
            x = self.att_norm_levels[i](x + x_att)
            
            # Feed-Forward sub-layer with residual connection and normalization
            x_ff1 = self.lin_layer_1[i](x)
            x_ff1 = self.activation_func(x_ff1)
            x_ff2 = self.lin_layer_2[i](x_ff1)
            x_ff2 = self.lin_dropout_levels[i](x_ff2)
            x = self.lin_norm_levels[i](x + x_ff2)
        
        x = self.out_dropout(x)
        return x