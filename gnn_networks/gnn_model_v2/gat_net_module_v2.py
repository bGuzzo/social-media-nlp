"""
This script defines the second version of the Graph Attention Network (GAT) model, `GatModelV2`,
which is designed for the task of link prediction in graph-structured data. This model represents
an architectural evolution from `GatModelV1`, incorporating more advanced components to potentially
enhance performance and training stability.

The key architectural improvements in `GatModelV2` are:

1.  **SwiGLU Activation Function**:
    - Instead of the standard ReLU activation used in the first version, `GatModelV2` employs
      the SwiGLU (Swish-Gated Linear Unit) activation function within its feed-forward network
      blocks. SwiGLU is a more sophisticated activation function that has been shown to
      yield performance improvements in various deep learning models, particularly those
      based on the Transformer architecture.

2.  **Root Mean Square Normalization (RMSNorm)**:
    - This model utilizes RMSNorm as its normalization layer, as opposed to the Layer Normalization
      used in `GatModelV1`. RMSNorm is a simpler and often more efficient normalization technique
      that can contribute to faster convergence and better performance.

The overall architecture of `GatModelV2` follows a similar structure to its predecessor:
- An input linear layer for feature projection.
- A `DeepGATBlockV2` that contains the core multi-head graph attention mechanism and the
  feed-forward layers with SwiGLU activation.
- An output linear layer to produce the final node embeddings.
- A decoder that uses a dot product to compute the logits for edge existence.

This model is a testament to the iterative process of neural network design, where architectural
refinements are made to push the boundaries of performance and efficiency.
"""

import torch
from torch.nn import Linear
from gnn_networks.gnn_model_v2.gat_block_v2 import DeepGATBlockV2
from gnn_networks.gnn_model_v2.swiglu_func_v2 import SwiGLU

# Default model parameters
NUM_HIDDEN_CHANNELS = 64
NUM_OUTPUT_CHANNELS = 32
NUM_ATTENTION_LAYER = 1
NUM_ATTENTION_HEAD = 1
DROPOUT_PROB = 0.5

class GatModelV2(torch.nn.Module):
    """
    A Graph Attention Network (GAT) model for link prediction, incorporating SwiGLU and RMSNorm.

    Args:
        in_channels (int): The number of input features per node.
        hidden_channels (int, optional): The number of hidden channels. Defaults to NUM_HIDDEN_CHANNELS.
        out_channels (int, optional): The number of output features per node. Defaults to NUM_OUTPUT_CHANNELS.
        num_attention_layer (int, optional): The number of attention layers. Defaults to NUM_ATTENTION_LAYER.
        num_attention_head (int, optional): The number of attention heads. Defaults to NUM_ATTENTION_HEAD.
        dropout_prob (float, optional): The dropout probability. Defaults to DROPOUT_PROB.
        norm_func (torch.nn.Module, optional): The normalization function to use. Defaults to torch.nn.RMSNorm.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = NUM_HIDDEN_CHANNELS,
        out_channels: int = NUM_OUTPUT_CHANNELS,
        num_attention_layer: int = NUM_ATTENTION_LAYER,
        num_attention_head=NUM_ATTENTION_HEAD,
        dropout_prob: float = DROPOUT_PROB,
        norm_func: torch.nn.Module = torch.nn.RMSNorm
    ):
        super().__init__()
        
        if not (0 <= dropout_prob < 1):
            raise ValueError(f"Dropout rate must be between 0 and 1, got {dropout_prob}")
        if num_attention_layer < 1:
            raise ValueError(f"At least one attention layer is required, got {num_attention_layer}")
        if in_channels <= 0 or hidden_channels <= 0 or out_channels <= 0:
            raise ValueError("Number of channels must be positive")

        self.in_dropout = torch.nn.Dropout(dropout_prob)
        self.in_lin_layer = Linear(in_channels, hidden_channels)
        
        activation_func = SwiGLU(hidden_channels)
        
        self.deep_layers = DeepGATBlockV2(
            num_levels=num_attention_layer, 
            num_heads=num_attention_head,
            hidden_channels=hidden_channels, 
            dropout_prob=dropout_prob,
            activation_func=activation_func,
            norm_func=norm_func
        )
        
        self.out_lin_level = Linear(hidden_channels, out_channels)
        
        self.model_name = f"gat_model_v2_{type(activation_func).__name__}_{norm_func.__name__}_{hidden_channels}_{out_channels}_{num_attention_layer}x{num_attention_head}_d_{dropout_prob}"
    
    def forward(self, x: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model, which computes the logits for positive and negative edges.

        Args:
            x (torch.Tensor): The input node features.
            pos_edge_index (torch.Tensor): The edge index of positive edges.
            neg_edge_index (torch.Tensor): The edge index of negative edges.

        Returns:
            torch.Tensor: The predicted logits for the edges.
        """
        z = self.encode(x, pos_edge_index)
        return self.decode(z, pos_edge_index, neg_edge_index)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        The encoder part of the model, which generates node embeddings.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge index of the graph.

        Returns:
            torch.Tensor: The generated node embeddings.
        """
        x = self.in_lin_layer(x)
        x = self.in_dropout(x)
        x = self.deep_layers(x, edge_index)
        x = self.out_lin_level(x)
        return x

    def decode(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> torch.Tensor:
        """
        The decoder part of the model, which predicts edge existence from node embeddings.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The edge index of positive edges.
            neg_edge_index (torch.Tensor): The edge index of negative edges.

        Returns:
            torch.Tensor: The predicted logits for the edges.
        """
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def get_adj_prob_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the probabilistic adjacency matrix from node embeddings.

        Args:
            z (torch.Tensor): The node embeddings.

        Returns:
            torch.Tensor: The probabilistic adjacency matrix.
        """
        adj_score = z @ z.t()
        return torch.sigmoid(adj_score)

    def get_name(self) -> str:
        """Returns the name of the model."""
        return self.model_name