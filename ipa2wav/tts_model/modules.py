# -*- coding: utf-8 -*-
"""Neural network building blocks for IPA2WAV synthesis.

This module provides the fundamental neural network components used throughout
the synthesis system, including embedding layers, normalization, convolutions,
and highway networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zero_pad=True):
        """Embeds a given tensor of ids into a dense vector representation.
        
        Creates and applies a trainable embedding lookup table to convert integer IDs
        into dense vectors. Optionally zero-pads the first row for padding tokens.
        
        Args:
            vocab_size: An int. Size of the vocabulary (number of unique tokens).
            num_units: An int. Dimension of the embedding vectors.
            zero_pad: A boolean. If True, all values in the first row (id 0)
                will be constant zeros, typically used for padding tokens.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_units, padding_idx=0 if zero_pad else None)
        
    def forward(self, inputs):
        """
        Args:
            inputs: A tensor with type int64 containing the ids to be looked up.
            
        Returns:
            A tensor of shape [batch_size, sequence_length, num_units] containing
            the embedded vectors.
        """
        return self.embedding(inputs)


class LayerNorm(nn.Module):
    def __init__(self, channels):
        """Applies layer normalization to the input tensor.
        
        Args:
            channels: Number of channels (features) in the input.
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, inputs):
        """
        Args:
            inputs: A tensor of shape [batch_size, time_steps, channels].
            
        Returns:
            A normalized tensor of the same shape.
        """
        return self.norm(inputs)


class HighwayNetwork(nn.Module):
    def __init__(self, channels):
        """Applies a highway network transformation to the inputs.
        
        Highway networks allow for easier training of deep networks by introducing
        gating units that control information flow. They help mitigate the vanishing
        gradient problem.
        
        Args:
            channels: Number of channels (features) in the input.
        """
        super().__init__()
        self.H = nn.Linear(channels, channels)
        self.T = nn.Linear(channels, channels)
        
    def forward(self, inputs):
        """
        Args:
            inputs: A 3D tensor of shape [batch_size, time_steps, channels].
            
        Returns:
            A 3D tensor of the same shape after applying the highway transformation.
        """
        H = F.relu(self.H(inputs))
        T = torch.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding='same', dilation=1, dropout_rate=0, use_bias=True,
                 activation_fn=None):
        """Applies 1D convolution to the input tensor with various options.
        
        A flexible 1D convolution implementation that supports:
        - Different padding modes
        - Dilation
        - Dropout
        - Activation functions
        - Bias terms
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Filter size (width).
            stride: Stride for convolution.
            padding: String. Either 'same' or 'valid'.
            dilation: Dilation rate for dilated convolution.
            dropout_rate: Float between 0 and 1. Dropout probability.
            use_bias: Boolean. Whether to add a bias term.
            activation_fn: Activation function to apply (None for linear).
        """
        super().__init__()
        
        # Calculate padding
        if padding.lower() == 'same':
            pad = (kernel_size - 1) * dilation // 2
        else:  # 'valid'
            pad = 0
            
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=pad, dilation=dilation,
                            bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        self.activation_fn = activation_fn
        
    def forward(self, inputs, training=True):
        """
        Args:
            inputs: A 3D tensor of shape [batch_size, time_steps, in_channels].
            training: Boolean. Whether in training mode (affects dropout).
            
        Returns:
            A 3D tensor of shape [batch_size, time_steps, out_channels].
        """
        # Convert [batch, time, channels] to [batch, channels, time]
        x = inputs.transpose(1, 2)
        
        x = self.conv(x)
        
        if self.dropout is not None and training:
            x = self.dropout(x)
            
        if self.activation_fn is not None:
            x = self.activation_fn(x)
            
        # Convert back to [batch, time, channels]
        return x.transpose(1, 2)


class HighwayConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding='same', dilation=1, dropout_rate=0, use_bias=True,
                 activation_fn=None):
        """Applies a highway convolution block.
        
        Combines highway networks with convolution, allowing the network to learn
        which information should pass through and which should be transformed.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Filter size (width).
            stride: Stride for convolution.
            padding: String. Either 'same' or 'valid'.
            dilation: Dilation rate for dilated convolution.
            dropout_rate: Float between 0 and 1. Dropout probability.
            use_bias: Boolean. Whether to add a bias term.
            activation_fn: Activation function to apply (None for linear).
        """
        super().__init__()
        
        self.H = Conv1d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, dropout_rate, use_bias,
                       activation_fn)
                       
        self.T = Conv1d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, dropout_rate, use_bias,
                       torch.sigmoid)
        
    def forward(self, inputs, training=True):
        """
        Args:
            inputs: A 3D tensor of shape [batch_size, time_steps, in_channels].
            training: Boolean. Whether in training mode (affects dropout).
            
        Returns:
            A 3D tensor of shape [batch_size, time_steps, out_channels].
        """
        # Manually pad the input tensor to maintain size
        pad = (self.H.conv.kernel_size[0] - 1) // 2
        inputs_padded = F.pad(inputs, (pad, pad), mode='reflect')

        H = self.H(inputs_padded, training)
        T = self.T(inputs_padded, training)
        return H * T + inputs * (1.0 - T)


class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding='same', dropout_rate=0, use_bias=True,
                 activation=None):
        """Applies transposed 1D convolution for upsampling.
        
        Implements the transposed convolution operation (also known as deconvolution)
        for upsampling sequences. Typically used in the SSRN network to increase
        temporal resolution.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Filter size (width).
            stride: Stride for upsampling.
            padding: String. Either 'same' or 'valid'.
            dropout_rate: Float between 0 and 1. Dropout probability.
            use_bias: Boolean. Whether to add a bias term.
            activation: Activation function to apply (None for linear).
        """
        super().__init__()
        
        # Calculate padding
        if padding.lower() == 'same':
            pad = (kernel_size - stride) // 2
        else:  # 'valid'
            pad = 0
            
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels,
                                               kernel_size, stride=stride,
                                               padding=pad, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        self.activation = activation
        
    def forward(self, inputs, training=True):
        """
        Args:
            inputs: A 3D tensor of shape [batch_size, time_steps, in_channels].
            training: Boolean. Whether in training mode (affects dropout).
            
        Returns:
            A 3D tensor of shape [batch_size, time_steps*stride, out_channels].
        """
        # Convert [batch, time, channels] to [batch, channels, time]
        x = inputs.transpose(1, 2)
        
        x = self.conv_transpose(x)
        
        if self.dropout is not None and training:
            x = self.dropout(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        # Convert back to [batch, time, channels]
        return x.transpose(1, 2)
