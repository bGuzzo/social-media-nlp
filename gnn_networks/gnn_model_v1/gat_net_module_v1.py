"""
This script defines the first version of the Graph Attention Network (GAT) model, `GatModelV1`,
which is tailored for the task of link prediction in graph-structured data. This model serves
as the baseline architecture in this project and is foundational to the more advanced `GatModelV2`.

The architecture of `GatModelV1` is composed of several key components:

1.  **Input Processing**: An initial linear layer projects the input node features into a
    higher-dimensional hidden space. This is followed by a dropout layer for regularization.

2.  **Deep GAT Block (`DeepGATBlockV1`)**: The core of the model is the `DeepGATBlockV1`,
    which contains a stack of graph attention layers. Each layer in this block is designed
    to capture the relationships between nodes by attending to their neighbors' features.
    This block is inspired by the Transformer architecture, with each layer consisting of a
    multi-head attention mechanism followed by a feed-forward network, with residual
    connections and layer normalization.

3.  **Output Projection**: An output linear layer maps the processed hidden features from the
    `DeepGATBlockV1` to the final embedding space.

4.  **Link Prediction Decoder**: The model employs a simple yet effective dot-product decoder.
    Given the final node embeddings, the decoder computes the dot product between the embeddings
    of two nodes to predict the logit (i.e., the unnormalized probability) of an edge existing
    between them.

This model provides a solid foundation for graph-based link prediction and serves as a benchmark
for evaluating the architectural improvements introduced in `GatModelV2`.
"""

import torch
from torch.nn import Linear, Module
from gnn_networks.gnn_model_v1.gat_block_v1 import DeepGATBlockV1

# Default model parameters
NUM_HIDDEN_CHANNELS = 64
NUM_OUTPUT_CHANNELS = 32
NUM_ATTENTION_LAYER = 1
NUM_ATTENTION_HEAD = 1
DROPOUT_PROB = 0.5

class GatModelV1(Module):
    """
    The first version of the Graph Attention Network (GAT) model for link prediction.

    Args:
        in_channels (int): The number of input features per node.
        hidden_channels (int, optional): The number of hidden channels. Defaults to NUM_HIDDEN_CHANNELS.
        out_channels (int, optional): The number of output features per node. Defaults to NUM_OUTPUT_CHANNELS.
        num_attention_layer (int, optional): The number of attention layers. Defaults to NUM_ATTENTION_LAYER.
        num_attention_head (int, optional): The number of attention heads. Defaults to NUM_ATTENTION_HEAD.
        dropout_prob (float, optional): The dropout probability. Defaults to DROPOUT_PROB.
        activation_func (Module, optional): The activation function to use. Defaults to torch.nn.ReLU().
        norm_func_class (Module, optional): The normalization function class to use. Defaults to torch.nn.LayerNorm.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = NUM_HIDDEN_CHANNELS,
        out_channels: int = NUM_OUTPUT_CHANNELS,
        num_attention_layer: int = NUM_ATTENTION_LAYER,
        num_attention_head=NUM_ATTENTION_HEAD,
        dropout_prob: float = DROPOUT_PROB,
        activation_func: Module = torch.nn.ReLU(),
        norm_func_class: Module = torch.nn.LayerNorm
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
        
        self.deep_layers = DeepGATBlockV1(
            num_levels=num_attention_layer, 
            num_heads=num_attention_head,
            hidden_channels=hidden_channels, 
            dropout_prob=dropout_prob,
            activation_func=activation_func,
            norm_func_class=norm_func_class
        )
        
        self.out_lin_level = Linear(hidden_channels, out_channels)
        
        self.model_name = f"gat_model_v1_{type(activation_func).__name__}_{norm_func_class.__name__}_{hidden_channels}_{out_channels}_{num_attention_layer}x{num_attention_head}_d_{dropout_prob}"

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
    
    def forward(self, x: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Args:
            x (torch.Tensor): The input node features.
            pos_edge_index (torch.Tensor): The edge index of positive edges.
            neg_edge_index (torch.Tensor): The edge index of negative edges.

        Returns:
            torch.Tensor: The predicted logits for the edges.
        """
        z = self.encode(x, pos_edge_index)
        return self.decode(z, pos_edge_index, neg_edge_index)
    
    def get_name(self) -> str:
        """Returns the name of the model."""
        return self.model_name