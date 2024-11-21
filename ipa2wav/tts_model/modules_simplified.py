"""Neural network building blocks for IPA2WAV synthesis.

Core neural network components:
1. Embedding: Token to vector conversion
2. LayerNorm: Normalization layer
3. Conv1d: 1D convolution with options
4. HighwayNetwork: Gated information flow
5. HighwayConv1d: Gated convolution
6. ConvTranspose1d: 1D transposed convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union


class Embedding(nn.Module):
    """Embeds tokens into dense vectors with optional zero padding."""
    def __init__(self, vocab_size: int, num_units: int, zero_pad: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            num_units, 
            padding_idx=0 if zero_pad else None
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.
        
        Args:
            inputs: Token IDs [batch_size, sequence_length]
            
        Returns:
            Embedded vectors [batch_size, sequence_length, num_units]
        """
        return self.embedding(inputs)


class LayerNorm(nn.Module):
    """Layer normalization with guaranteed unit standard deviation."""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, channels]
            
        Returns:
            Normalized tensor [batch_size, time_steps, channels]
        """
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize to zero mean and unit variance
        x = (inputs - mean) / (std + self.eps)
        
        # Scale and shift
        return self.gamma * x + self.beta


class Conv1d(nn.Module):
    """1D convolution with dropout and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: str = 'same',
        dilation: int = 1,
        dropout_rate: float = 0.0,
        activation_fn: Optional[Callable] = None
    ):
        super().__init__()
        
        # Store padding mode
        self.padding_mode = padding.lower()
        
        # Calculate padding size for 'same' mode
        if self.padding_mode == 'same':
            # For 'same' padding: output_size = input_size
            # pad = (kernel_size - 1) * dilation // 2
            total_pad = (kernel_size - 1) * dilation
            pad = total_pad // 2
        else:
            pad = 0
            
        # Core layers
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,  # Amount of padding (not mode)
            dilation=dilation,
            padding_mode='zeros'  # PyTorch's padding mode (zeros, reflect, replicate, circular)
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply convolution, dropout, and activation.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, channels]
            training: Whether in training mode (affects dropout)
            
        Returns:
            Output tensor [batch_size, time_steps, channels]
        """
        # Transpose for Conv1d (NCL -> NLC)
        x = inputs.transpose(1, 2)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply dropout if needed
        if self.dropout is not None and training:
            x = self.dropout(x)
            
        # Apply activation if needed
        if self.activation_fn is not None:
            x = self.activation_fn(x)
            
        # Transpose back (NLC -> NCL)
        return x.transpose(1, 2)


class HighwayNetwork(nn.Module):
    """Highway network for controlling information flow.
    
    Implements a gating mechanism that learns to balance the flow of 
    transformed and untransformed information through the network.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.H = nn.Linear(channels, channels)  # Transform gate
        self.T = nn.Linear(channels, channels)  # Carry gate
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply highway network transformation.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, channels]
            
        Returns:
            Transformed tensor [batch_size, time_steps, channels]
        """
        H = torch.relu(self.H(inputs))
        T = torch.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class HighwayConv1d(nn.Module):
    """Highway convolution combining gating with 1D convolution."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: str = 'same',
        dilation: int = 1,
        dropout_rate: float = 0.0,
        activation_fn: Optional[Callable] = None
    ):
        super().__init__()
        
        # Transform gate
        self.H = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        )
        
        # Carry gate
        self.T = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dropout_rate=dropout_rate,
            activation_fn=torch.sigmoid
        )
        
    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply highway convolution.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, in_channels]
            training: Whether in training mode (affects dropout)
            
        Returns:
            Output tensor [batch_size, time_steps, out_channels]
        """
        H = self.H(inputs, training)
        T = self.T(inputs, training)
        return H * T + inputs * (1.0 - T)


class ConvTranspose1d(nn.Module):
    """1D transposed convolution for upsampling."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: str = 'same',
        dropout_rate: float = 0.0,
        activation: Optional[Callable] = None
    ):
        super().__init__()
        
        # Calculate padding
        pad = (kernel_size - stride) // 2 if padding.lower() == 'same' else 0
            
        # Core layers
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation = activation
        
    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply transposed convolution for upsampling.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, in_channels]
            training: Whether in training mode (affects dropout)
            
        Returns:
            Output tensor [batch_size, time_steps*stride, out_channels]
        """
        # Transpose for ConvTranspose1d (NCL -> NLC)
        x = inputs.transpose(1, 2)
        
        # Apply transposed convolution
        x = self.conv_transpose(x)
        
        # Apply dropout if needed
        if self.dropout is not None and training:
            x = self.dropout(x)
            
        # Apply activation if needed
        if self.activation is not None:
            x = self.activation(x)
            
        # Transpose back (NLC -> NCL)
        return x.transpose(1, 2)
