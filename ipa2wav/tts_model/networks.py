# -*- coding: utf-8 -*-
"""Neural network architectures for IPA2WAV synthesis.

This module defines the core network architectures:
1. Text2Mel: Converts IPA text to mel-spectrograms
2. SSRN: Converts mel-spectrograms to linear spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperparams import Hyperparams as hp
from .modules import Embedding, LayerNorm, Conv1d, HighwayConv1d, ConvTranspose1d


class TextEncoder(nn.Module):
    """Text Encoder network.
    
    Converts input text sequences into keys and values for attention mechanism.
    Architecture:
    1. Character embedding
    2. 1D convolution layers
    3. Highway convolution blocks
    """
    def __init__(self):
        super().__init__()
        
        # Embedding layer
        self.embed = Embedding(len(hp.vocab), hp.e)
        
        # Initial convolution layers
        self.prenet = nn.ModuleList([
            Conv1d(hp.e, 2*hp.d, kernel_size=1, dropout_rate=hp.dropout_rate,
                  activation_fn=torch.relu),
            Conv1d(2*hp.d, 2*hp.d, kernel_size=1, dropout_rate=hp.dropout_rate,
                  activation_fn=torch.relu)
        ])
        
        # Highway convolution blocks
        self.conv_bank = nn.ModuleList([
            HighwayConv1d(2*hp.d, 2*hp.d, kernel_size=k, dilation=1,
                         dropout_rate=hp.dropout_rate, activation_fn=torch.relu)
            for k in range(1, hp.K+1)
        ])
        
        # Projection layers
        self.proj1 = Conv1d(2*hp.d*hp.K, 2*hp.d, kernel_size=3,
                          dropout_rate=hp.dropout_rate, activation_fn=torch.relu)
        self.proj2 = Conv1d(2*hp.d, hp.d, kernel_size=3,
                          dropout_rate=hp.dropout_rate, activation_fn=None)
        
    def forward(self, L, training=True):
        """
        Args:
            L: Text inputs. Shape: [batch_size, sequence_length]
            training: Whether the network is in training mode. Affects dropout.
        
        Returns:
            tuple: A pair of tensors:
                - K: Keys for attention. Shape: [batch_size, seq_length, hidden_size]
                - V: Values for attention. Shape: [batch_size, seq_length, hidden_size]
        """
        # Embedding
        x = self.embed(L)
        
        # Pre-net
        for conv in self.prenet:
            x = conv(x, training)
        
        # Conv bank
        conv_outputs = []
        for conv in self.conv_bank:
            y = conv(x, training)
            conv_outputs.append(y)
        x = torch.cat(conv_outputs, dim=-1)
        
        # Projections
        x = self.proj1(x, training)
        x = self.proj2(x, training)
        
        return x, x  # K, V are the same in this implementation


class AudioEncoder(nn.Module):
    """Audio Encoder network.
    
    Processes mel-spectrograms into queries for attention mechanism.
    Uses a series of convolutional layers with highway connections.
    """
    def __init__(self):
        super().__init__()
        
        # Pre-net
        self.prenet = nn.ModuleList([
            Conv1d(hp.n_mels, hp.d, kernel_size=1, dropout_rate=hp.dropout_rate,
                  activation_fn=torch.relu),
            Conv1d(hp.d, hp.d, kernel_size=1, dropout_rate=hp.dropout_rate,
                  activation_fn=torch.relu)
        ])
        
        # CBHG module
        self.conv_bank = nn.ModuleList([
            Conv1d(hp.d, hp.d, kernel_size=k, activation_fn=torch.relu)
            for k in range(1, hp.K+1)
        ])
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        self.conv_projections = nn.ModuleList([
            Conv1d(hp.d*hp.K, hp.d, kernel_size=3, activation_fn=torch.relu),
            Conv1d(hp.d, hp.d, kernel_size=3, activation_fn=None)
        ])
        
        self.highway_net = nn.ModuleList([
            HighwayConv1d(hp.d, hp.d, kernel_size=3)
            for _ in range(4)
        ])
        
    def forward(self, S, training=True):
        """
        Args:
            S: Mel-spectrogram inputs. Shape: [batch_size, time_steps/r, n_mels]
            training: Whether the network is in training mode. Affects dropout.
        
        Returns:
            tf.Tensor: Queries for attention. Shape: [batch_size, time_steps/r, hidden_size]
        """
        # Pre-net
        for conv in self.prenet:
            S = conv(S, training)
        
        # Conv bank
        conv_outputs = []
        for conv in self.conv_bank:
            y = conv(S, training)
            conv_outputs.append(y)
        x = torch.cat(conv_outputs, dim=-1)
        
        # Max pooling
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.maxpool(x)
        x = x.transpose(1, 2)  # [B, T, C]
        
        # Conv projections
        for conv in self.conv_projections:
            x = conv(x, training)
        
        # Highway layers
        for highway in self.highway_net:
            x = highway(x, training)
        
        return x


class Attention(nn.Module):
    """Multi-head attention mechanism with optional monotonic constraint."""
    def __init__(self):
        super().__init__()
        self.attention_heads = hp.attention_heads
        
    def forward(self, Q, K, V, monotonic_attention=False, prev_max_attentions=None):
        """
        Args:
            Q: Queries from audio encoder. Shape: [batch_size, time_steps/r, hidden_size]
            K: Keys from text encoder. Shape: [batch_size, seq_length, hidden_size]
            V: Values from text encoder. Shape: [batch_size, seq_length, hidden_size]
            monotonic_attention: Whether to enforce monotonic attention (for inference)
            prev_max_attentions: Previous attention peaks for monotonic attention
            
        Returns:
            tuple: A tuple containing:
                - R: Context vectors concatenated with queries
                - alignments: Attention weights
                - max_attentions: Positions of maximum attention
        """
        # Split heads
        Q_split = torch.chunk(Q, self.attention_heads, dim=-1)
        K_split = torch.chunk(K, self.attention_heads, dim=-1)
        V_split = torch.chunk(V, self.attention_heads, dim=-1)
        
        # Compute attention for each head
        outputs = []
        alignments = []
        max_attentions = []
        
        for i in range(self.attention_heads):
            q, k, v = Q_split[i], K_split[i], V_split[i]
            
            # Attention scores
            attention = torch.bmm(q, k.transpose(1, 2))
            attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
            
            if monotonic_attention and prev_max_attentions is not None:
                # Create mask for monotonic attention
                mask = torch.arange(k.size(1)).unsqueeze(0).to(q.device)
                mask = mask <= (prev_max_attentions[i].unsqueeze(1) + hp.attention_win_size)
                attention = attention.masked_fill(~mask.unsqueeze(1), float('-inf'))
            
            # Attention weights
            attention_weights = F.softmax(attention, dim=-1)
            
            # Apply attention
            head_output = torch.bmm(attention_weights, v)
            outputs.append(head_output)
            alignments.append(attention_weights)
            
            # Maximum attention position
            max_attentions.append(torch.argmax(attention_weights, dim=-1))
        
        # Concatenate heads
        R = torch.cat(outputs, dim=-1)
        alignments = torch.stack(alignments, dim=1)  # [B, heads, T_q, T_k]
        max_attentions = torch.stack(max_attentions, dim=1)  # [B, heads, T_q]
        
        # Concatenate with original queries
        R = torch.cat([R, Q], dim=-1)
        
        return R, alignments, max_attentions


class AudioDecoder(nn.Module):
    """Audio Decoder network.
    
    Generates mel-spectrogram predictions from context vectors and queries.
    Uses a series of convolutional layers with highway connections.
    """
    def __init__(self):
        super().__init__()
        
        # Pre-net
        self.prenet = nn.ModuleList([
            Conv1d(hp.d*2, hp.d, kernel_size=1, dropout_rate=hp.dropout_rate,
                  activation_fn=torch.relu),
            Conv1d(hp.d, hp.d, kernel_size=1, dropout_rate=hp.dropout_rate,
                  activation_fn=torch.relu)
        ])
        
        # CBHG module
        self.conv_bank = nn.ModuleList([
            Conv1d(hp.d, hp.d, kernel_size=k, activation_fn=torch.relu)
            for k in range(1, hp.K+1)
        ])
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        self.conv_projections = nn.ModuleList([
            Conv1d(hp.d*hp.K, hp.d, kernel_size=3, activation_fn=torch.relu),
            Conv1d(hp.d, hp.d, kernel_size=3, activation_fn=None)
        ])
        
        self.highway_net = nn.ModuleList([
            HighwayConv1d(hp.d, hp.d, kernel_size=3)
            for _ in range(4)
        ])
        
        # Output projection
        self.out_proj = Conv1d(hp.d, hp.n_mels, kernel_size=1, activation_fn=None)
        
    def forward(self, R, training=True):
        """
        Args:
            R: Context and query vectors. Shape: [batch_size, time_steps/r, 2*hidden_size]
            training: Whether the network is in training mode. Affects dropout.
            
        Returns:
            tf.Tensor: Predicted mel-spectrograms. Shape: [batch_size, time_steps/r, n_mels]
        """
        # Pre-net
        for conv in self.prenet:
            R = conv(R, training)
        
        # Conv bank
        conv_outputs = []
        for conv in self.conv_bank:
            y = conv(R, training)
            conv_outputs.append(y)
        x = torch.cat(conv_outputs, dim=-1)
        
        # Max pooling
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.maxpool(x)
        x = x.transpose(1, 2)  # [B, T, C]
        
        # Conv projections
        for conv in self.conv_projections:
            x = conv(x, training)
        
        # Highway layers
        for highway in self.highway_net:
            x = highway(x, training)
        
        # Output projection
        mel_out = self.out_proj(x, training)
        
        return mel_out


class SSRN(nn.Module):
    """Spectrogram Super-resolution Network (SSRN).
    
    Converts mel-spectrograms to full linear-scale spectrograms by:
    1. Upsampling in time
    2. Expanding frequency resolution
    """
    def __init__(self):
        super().__init__()
        
        # Conv layers
        self.conv1 = Conv1d(hp.n_mels, hp.c, kernel_size=1, dropout_rate=hp.dropout_rate,
                          activation_fn=torch.relu)
        
        # Conv bank
        self.conv_bank = nn.ModuleList([
            Conv1d(hp.c, hp.c, kernel_size=k, activation_fn=torch.relu)
            for k in range(1, hp.K+1)
        ])
        
        # Max pooling
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # Conv projections
        self.conv_proj1 = Conv1d(hp.c*hp.K, hp.c, kernel_size=3,
                               activation_fn=torch.relu)
        self.conv_proj2 = Conv1d(hp.c, hp.c, kernel_size=3,
                               activation_fn=None)
        
        # Highway net
        self.highway_net = nn.ModuleList([
            HighwayConv1d(hp.c, hp.c, kernel_size=3)
            for _ in range(4)
        ])
        
        # Deconv layers
        self.deconv1 = ConvTranspose1d(hp.c, hp.c, kernel_size=2, stride=2,
                                     activation=torch.relu)
        self.deconv2 = ConvTranspose1d(hp.c, hp.c, kernel_size=2, stride=2,
                                     activation=torch.relu)
        
        # Output projection
        self.out_proj = Conv1d(hp.c, hp.n_fft//2+1, kernel_size=1,
                             activation_fn=None)
        
    def forward(self, Y, training=True):
        """
        Args:
            Y: Mel-spectrogram inputs. Shape: [batch_size, time_steps/r, n_mels]
            training: Whether the network is in training mode. Affects dropout.
            
        Returns:
            tf.Tensor: Predicted linear spectrograms. Shape: [batch_size, time_steps, 1+n_fft/2]
        """
        # Initial conv
        x = self.conv1(Y, training)
        
        # Conv bank
        conv_outputs = []
        for conv in self.conv_bank:
            y = conv(x, training)
            conv_outputs.append(y)
        x = torch.cat(conv_outputs, dim=-1)
        
        # Max pooling
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.maxpool(x)
        x = x.transpose(1, 2)  # [B, T, C]
        
        # Conv projections
        x = self.conv_proj1(x, training)
        x = self.conv_proj2(x, training)
        
        # Highway layers
        for highway in self.highway_net:
            x = highway(x, training)
        
        # Deconv layers
        x = self.deconv1(x, training)
        x = self.deconv2(x, training)
        
        # Output projection
        mag = self.out_proj(x, training)
        
        return mag
