import torch
from torch.nn import Linear
from gnn_model_v1.gat_block_v1 import DeepGATBlockV1


# Default model values
NUM_HIDDEN_CHANNELS = 64
NUM_OUTPUT_CHANNELS = 32
NUM_ATTENTION_LAYER = 1
NUM_ATTENTION_HEAD = 1
DROPOUT_PROB = 0.5

# Main class discussed in project report 
class GatModelV1(torch.nn.Module):
    
    """
    This GNN model uses a DeepGATBlock to perform graph attention network operations.
    It consists of an input dropout layer, an input linear layer, a DeepGATBlock for multiple attention layers, and an output linear layer.  
    The DeepGATBlock itself contains multiple levels of GATConv layers,
    dropout layers, and normalization layers (LayerNorm by default). 
    The model uses a simple dot-product decoder to predict link probabilities between nodes. 
    The model is designed for link prediction tasks.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = NUM_HIDDEN_CHANNELS,
        out_channels: int = NUM_OUTPUT_CHANNELS,
        num_attention_layer: int = NUM_ATTENTION_LAYER,
        num_attention_head=NUM_ATTENTION_HEAD,
        dropout_prob: float = DROPOUT_PROB,
        activation_func: torch.nn.Module = torch.nn.ReLU(),
        norm_func_class: torch.nn.Module = torch.nn.LayerNorm
    ):
        super().__init__()
        
        if dropout_prob >= 1 or dropout_prob < 0:
            raise ValueError(f"Dropout rate must be between 0 and 1, got {dropout_prob}")
        
        if num_attention_layer < 1:
            raise ValueError(f"At least one attention layers is required, got {num_attention_layer}")
        
        if in_channels <= 0:
            raise ValueError(f"Input channels must be greater than 0, got {in_channels}")
        
        if hidden_channels <= 0:
            raise ValueError(f"Hidden channels must be greater than 0, got {hidden_channels}")
        
        if out_channels <= 0:
            raise ValueError(f"Output channels must be greater than 0, got {out_channels}")
        
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

    # Encode input node feature and get processed node representation
    def encode(self, x, edge_index):
        x = self.in_lin_layer(x)
        x = self.in_dropout(x)
        x = self.deep_layers(x, edge_index)
        x = self.out_lin_level(x)
        return x

    # Decode the logits of the positive and negative edges
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # Compute logits by feature vectors multiplication
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    # Get probability adjacence matrix 
    def get_adj_prob_matrix(self, z):
        adj_score = z @ z.t() # Perferom a logit matrix self product
        return torch.sigmoid(adj_score)
    
    def forward(self, x, pos_edge_index, neg_edge_index):
        z = self.encode(x, pos_edge_index)
        return self.decode(z, pos_edge_index, neg_edge_index)
    
    def get_name(self) -> str:
        return self.model_name

