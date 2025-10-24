"""
This script defines the `DeepGATBlockV2`, a modular and enhanced building block for a deep
Graph Attention Network (GAT). This module is central to the architecture of `GatModelV2`
and represents a more flexible and powerful version of its predecessor, `DeepGATBlockV1`.

The design of `DeepGATBlockV2` is heavily inspired by the encoder layer of the original
Transformer model, adapting its principles to the graph domain. It consists of a stack of
identical layers, each of which is composed of two main sub-modules:

1.  **Multi-Head Graph Attention Sub-module**:
    - This sub-module employs a multi-head graph attention mechanism (`GATConv`) to learn
      the relative importance of neighboring nodes and aggregate their features.
    - It is followed by a dropout layer and a residual connection, which adds the output of
      the attention layer to its input. The result is then normalized.

2.  **Feed-Forward Network Sub-module**:
    - This sub-module consists of a two-layer feed-forward network that applies a non-linear
      transformation to the output of the attention sub-module.
    - It also includes a residual connection and a normalization layer.

The key improvements in `DeepGATBlockV2` are its modularity and flexibility. It is designed
to accept custom activation and normalization functions as arguments, allowing for easy
experimentation with different architectural configurations. In the context of `GatModelV2`,
this block is instantiated with the `SwiGLU` activation function and `RMSNorm` for
normalization, which are more advanced components than those used in the first version.

By stacking these blocks, it is possible to create a deep GAT model that can capture
complex and long-range dependencies within the graph structure, which is crucial for
achieving high performance on tasks like link prediction.
"""

import torch
from torch_geometric.nn import GATConv
from torch.nn import Dropout, Module, ModuleList, Linear

class DeepGATBlockV2(Module):
    """
    A deep Graph Attention Network (GAT) block, composed of multiple levels of GAT and feed-forward layers.

    Args:
        num_levels (int): The number of levels (i.e., stacked GAT and FFN blocks).
        hidden_channels (int): The number of hidden channels.
        num_heads (int, optional): The number of attention heads. Defaults to 4.
        dropout_prob (float, optional): The dropout probability. Defaults to 0.5.
        activation_func (Module, optional): The activation function to use. Defaults to torch.nn.ReLU().
        norm_func (Module, optional): The normalization function to use. Defaults to torch.nn.RMSNorm.
    """
    def __init__(
        self, 
        num_levels: int, 
        hidden_channels: int, 
        num_heads: int = 4, 
        dropout_prob: float = 0.5,
        activation_func: Module = torch.nn.ReLU(),
        norm_func: Module = torch.nn.RMSNorm
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
            self.att_norm_levels.append(norm_func(hidden_channels))
            
            self.lin_layer_1.append(Linear(hidden_channels, hidden_channels))
            self.lin_layer_2.append(Linear(hidden_channels, hidden_channels))
            self.lin_dropout_levels.append(Dropout(dropout_prob))
            self.lin_norm_levels.append(norm_func(hidden_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the DeepGATBlockV2.

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