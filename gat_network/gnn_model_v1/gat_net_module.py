import torch
from torch.nn import Linear
from gnn_model_v1.attention_block import DeepAttnBlock


# Default model values
NUM_HIDDEN_CHANNELS = 64
NUM_OUTPUT_CHANNELS = 32
NUM_ATTENTION_LAYER = 1
NUM_ATTENTION_HEAD = 1
DROPOUT_PROB = 0.5


class GatModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = NUM_HIDDEN_CHANNELS,
        out_channels: int = NUM_OUTPUT_CHANNELS,
        num_attention_layer: int = NUM_ATTENTION_LAYER,
        num_attention_head=NUM_ATTENTION_HEAD,
        dropout_prob: float = DROPOUT_PROB,
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
        
        self.in_lin_layer = Linear(in_channels, hidden_channels)
        
        self.deep_layers = DeepAttnBlock(
            num_levels=num_attention_layer, 
            num_heads=num_attention_head,
            hidden_channels=hidden_channels, 
            dropout_prob=dropout_prob
        )
        
        self.out_lin_level = Linear(hidden_channels, out_channels)
        
    
    def forward(self, x, pos_edge_index, neg_edge_index):
        z = self.encode(x, pos_edge_index)
        return self.decode(z, pos_edge_index, neg_edge_index)

    def encode(self, x, edge_index):
        x = self.in_lin_layer(x)
        x = self.deep_layers(x, edge_index)
        x = self.out_lin_level(x)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        adj_score = z @ z.t()
        # Only for debug
        # print(f"Link prediction score matrix \n{torch.sigmoid(adj_score)}")
        return (adj_score > 0).nonzero(as_tuple=False).t()
