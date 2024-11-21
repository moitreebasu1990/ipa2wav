"""Neural network architectures for IPA2WAV synthesis.

This module defines simplified versions of the core network architectures:
1. Text2Mel: Converts IPA text to mel-spectrograms
2. SSRN: Converts mel-spectrograms to linear spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .hyperparams import Hyperparams as hp
from .modules_simplified import (
    Embedding, LayerNorm, Conv1d, HighwayNetwork,
    HighwayConv1d, ConvTranspose1d
)


class TextEncoder(nn.Module):
    """Simplified Text Encoder network."""
    def __init__(self):
        super().__init__()
        
        # Get hyperparameter values
        vocab_size = len(hp.vocab)
        embed_size = hp.embed_size
        d = hp.d
        dropout_rate = hp.dropout_rate
        highway_layers = hp.highway_layers
        
        # Embedding layer
        self.embed = Embedding(vocab_size, embed_size)
        
        # Initial convolutions
        self.prenet = nn.ModuleList([
            Conv1d(embed_size, d, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu),
            Conv1d(d, d, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu)
        ])
        
        # Highway convolutions for complex feature extraction
        self.highway_convs = nn.ModuleList([
            HighwayConv1d(d, d, kernel_size=3, dropout_rate=dropout_rate,
                         activation_fn=torch.relu)
            for _ in range(highway_layers)
        ])
        
        # Final projections for K and V
        self.key_proj = Conv1d(d, embed_size, kernel_size=1, dropout_rate=dropout_rate)
        self.value_proj = Conv1d(d, embed_size, kernel_size=1, dropout_rate=dropout_rate)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_length]
            
        Returns:
            tuple: (K, V)
                - K: Key tensor [batch_size, seq_length, embed_size]
                - V: Value tensor [batch_size, seq_length, embed_size]
        """
        # Embedding
        x = self.embed(x)                    # [B, T, E]
        
        # Initial convolutions
        for conv in self.prenet:
            x = conv(x)                      # [B, T, D]
        
        # Highway convolutions
        for highway in self.highway_convs:
            x = highway(x)                   # [B, T, D]
        
        # Project to K and V
        K = self.key_proj(x)                # [B, T, E]
        V = self.value_proj(x)              # [B, T, E]
        
        return K, V


class AudioEncoder(nn.Module):
    """Simplified Audio Encoder network."""
    def __init__(self):
        super().__init__()
        
        # Get hyperparameter values
        n_mels = hp.n_mels
        d = hp.d
        embed_size = hp.embed_size
        dropout_rate = hp.dropout_rate
        highway_layers = hp.highway_layers
        
        # Initial convolutions
        self.prenet = nn.ModuleList([
            Conv1d(n_mels, d, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu),
            Conv1d(d, d, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu)
        ])
        
        # Highway network for complex feature extraction
        self.highway_net = nn.ModuleList([
            HighwayNetwork(d)
            for _ in range(highway_layers)
        ])
        
        # Final projection
        self.proj = Conv1d(d, embed_size, kernel_size=1, dropout_rate=dropout_rate)
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, time_steps, n_mels]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, time_steps, embed_size]
        """
        # Initial convolutions
        for conv in self.prenet:
            x = conv(x, training)           # [B, T/r, D]
        
        # Highway networks
        for highway in self.highway_net:
            x = highway(x)                  # [B, T/r, D]
        
        # Final projection
        x = self.proj(x, training)         # [B, T/r, E]
        
        return x


class Attention(nn.Module):
    """Multi-head attention with optional monotonic constraint."""
    def __init__(self):
        super().__init__()
        
        # Get hyperparameter values
        embed_size = hp.embed_size
        attention_heads = hp.attention_heads
        attention_dim = hp.attention_dim
        
        self.num_heads = attention_heads
        self.attention_dim = attention_dim
        
        # Projections for Q, K, V
        self.query_proj = nn.Linear(embed_size, attention_dim * attention_heads)
        self.key_proj = nn.Linear(embed_size, attention_dim * attention_heads)
        self.value_proj = nn.Linear(embed_size, attention_dim * attention_heads)
        self.output_proj = nn.Linear(attention_dim * attention_heads, embed_size)
        
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        monotonic_attention: bool = False,
        prev_max_attentions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply multi-head attention.
        
        Args:
            Q: Queries [batch_size, time_steps/r, embed_size]
            K: Keys [batch_size, seq_length, embed_size]
            V: Values [batch_size, seq_length, embed_size]
            monotonic_attention: Whether to enforce monotonic attention
            prev_max_attentions: Previous attention peaks for monotonic attention
            
        Returns:
            tuple:
                - Output tensor [batch_size, time_steps/r, embed_size]
                - Attention weights [batch_size, heads, time_steps/r, seq_length]
                - Maximum attention positions [batch_size]
        """
        batch_size = Q.size(0)
        
        # Project inputs
        q = self.query_proj(Q).view(batch_size, -1, self.num_heads, self.attention_dim)
        k = self.key_proj(K).view(batch_size, -1, self.num_heads, self.attention_dim)
        v = self.value_proj(V).view(batch_size, -1, self.num_heads, self.attention_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, T/r, D]
        k = k.transpose(1, 2)  # [B, H, T, D]
        v = v.transpose(1, 2)  # [B, H, T, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        
        if monotonic_attention and prev_max_attentions is not None:
            # Create attention mask for monotonic attention
            mask = torch.arange(scores.size(-1), device=scores.device)
            mask = mask.expand(scores.size(0), scores.size(1), scores.size(2), -1)
            prev_max = prev_max_attentions.view(scores.size(0), 1, 1, 1)
            win_size = hp.attention_win_size
            mask = (mask <= prev_max + win_size)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        
        # Compute context vectors
        context = torch.matmul(attn, v)      # [B, H, T/r, D]
        context = context.transpose(1, 2)     # [B, T/r, H, D]
        context = context.reshape(batch_size, -1, self.num_heads * self.attention_dim)
        
        # Project output
        output = self.output_proj(context)
        
        # Get maximum attention positions
        max_attentions = attn.max(dim=-1)[1].max(dim=1)[0]
        
        return output, attn, max_attentions


class AudioDecoder(nn.Module):
    """Simplified Audio Decoder network."""
    def __init__(self):
        super().__init__()
        
        # Get hyperparameter values
        embed_size = hp.embed_size
        d = hp.d
        n_mels = hp.n_mels
        dropout_rate = hp.dropout_rate
        highway_layers = hp.highway_layers
        
        # Initial convolutions
        self.prenet = nn.ModuleList([
            Conv1d(embed_size, d, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu),
            Conv1d(d, d, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu)
        ])
        
        # Highway convolutions
        self.highway_convs = nn.ModuleList([
            HighwayConv1d(d, d, kernel_size=3, dropout_rate=dropout_rate,
                         activation_fn=torch.relu)
            for _ in range(highway_layers)
        ])
        
        # Output projection
        self.proj = Conv1d(d, n_mels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, time_steps, embed_size]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, time_steps, n_mels]
        """
        # Initial convolutions
        for conv in self.prenet:
            x = conv(x, training)           # [B, T/r, D]
        
        # Highway convolutions
        for highway in self.highway_convs:
            x = highway(x, training)        # [B, T/r, D]
        
        # Output projection
        x = self.proj(x, training)         # [B, T/r, M]
        
        return x


class SSRN(nn.Module):
    """Simplified Spectrogram Super-resolution Network."""
    def __init__(self):
        super().__init__()
        
        # Get hyperparameter values
        n_mels = hp.n_mels
        c = hp.c
        n_fft = hp.n_fft
        dropout_rate = hp.dropout_rate
        highway_layers = hp.highway_layers
        
        # Initial convolutions
        self.prenet = nn.ModuleList([
            Conv1d(n_mels, c, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu),
            Conv1d(c, c, kernel_size=3, dropout_rate=dropout_rate,
                  activation_fn=torch.relu)
        ])
        
        # Highway convolutions
        self.highway_convs = nn.ModuleList([
            HighwayConv1d(c, c, kernel_size=3, dropout_rate=dropout_rate,
                         activation_fn=torch.relu)
            for _ in range(highway_layers)
        ])
        
        # Upsampling
        self.deconvs = nn.ModuleList([
            ConvTranspose1d(c, c, kernel_size=3, stride=2,
                          dropout_rate=dropout_rate, activation=torch.relu)
            for _ in range(2)
        ])
        
        # Output projection
        self.proj = Conv1d(c, n_fft//2+1, kernel_size=1)
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, time_steps, n_mels]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, time_steps*4, n_fft//2+1]
        """
        # Initial convolutions
        for conv in self.prenet:
            x = conv(x, training)           # [B, T, C]
        
        # Highway convolutions
        for highway in self.highway_convs:
            x = highway(x, training)        # [B, T, C]
        
        # Upsampling
        for deconv in self.deconvs:
            x = deconv(x, training)         # [B, T*4, C]
        
        # Output projection
        x = self.proj(x, training)         # [B, T*4, F]
        
        return x
