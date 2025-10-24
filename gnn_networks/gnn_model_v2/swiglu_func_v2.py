"""
This script defines the SwiGLU (Swish-Gated Linear Unit) activation function as a
custom PyTorch module. The SwiGLU activation function is a variant of the Gated
Linear Unit (GLU) that uses the Swish activation function as its gating mechanism.
It has been shown to improve the performance of Transformer-based models.

This implementation is designed to be a plug-and-play component in a neural network,
particularly within the feed-forward blocks of a Transformer-like architecture.
"""

import torch
from torch import nn

class SwiGLU(nn.Module):
    """
    Implements the SwiGLU activation function.

    This module takes a tensor as input, applies two separate linear transformations to it,
    and then combines them using a Swish-gated mechanism. The formula is:

    SwiGLU(x) = (x * W_1 + b_1) * sigmoid(x * W_1 + b_1) * (x * W_2 + b_2)

    where W_1, b_1, W_2, and b_2 are learnable parameters of the linear layers.

    Args:
        dimension (int): The input and output dimension of the linear layers.
    """
    def __init__(self, dimension: int):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the SwiGLU activation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the SwiGLU activation.
        """
        # Apply the first linear transformation and the Swish activation.
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        
        # Apply the second linear transformation and multiply with the Swish output.
        swiglu = swish * self.linear_2(x)

        return swiglu
