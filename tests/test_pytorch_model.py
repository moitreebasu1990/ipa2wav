"""Unit tests for PyTorch model components and training."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipa2wav.tts_model.modules_simplified import (
    Embedding, LayerNorm, HighwayNetwork, Conv1d, 
    HighwayConv1d, ConvTranspose1d
)
from ipa2wav.tts_model.networks_simplified import (
    TextEncoder, AudioEncoder, AudioDecoder, 
    Attention, SSRN
)
from ipa2wav.tts_model.train import Text2MelModel
from ipa2wav.tts_model.hyperparams import Hyperparams as hp


@pytest.fixture
def device():
    """Get PyTorch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestModules:
    """Test basic PyTorch neural network modules."""
    
    def test_embedding(self, device):
        """Test embedding layer."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10
        num_units = 8
        
        # Create layer
        embed = Embedding(vocab_size, num_units)
        embed.to(device)
        
        # Create input
        inputs = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], 
                            dtype=torch.long, device=device)
        
        # Forward pass
        outputs = embed(inputs)
        
        # Check output shape
        assert outputs.shape == (batch_size, seq_len, num_units)
        assert outputs.device == device
    
    def test_layer_norm(self, device):
        """Test layer normalization."""
        batch_size = 2
        seq_len = 4
        channels = 8
        
        # Create layer
        norm = LayerNorm(channels)
        norm.to(device)
        
        # Create input
        inputs = torch.randn(batch_size, seq_len, channels, device=device)
        
        # Forward pass
        outputs = norm(inputs)
        
        # Check output shape and normalization
        assert outputs.shape == inputs.shape
        assert outputs.device == device
        assert torch.allclose(outputs.mean(dim=-1), torch.zeros_like(outputs.mean(dim=-1)), atol=1e-6)
        assert torch.allclose(outputs.std(dim=-1), torch.ones_like(outputs.std(dim=-1)), atol=0.1)
    
    def test_highway_network(self, device):
        """Test highway network."""
        batch_size = 2
        seq_len = 4
        channels = 8
        
        # Create layer
        highway = HighwayNetwork(channels)
        highway.to(device)
        
        # Create input
        inputs = torch.randn(batch_size, seq_len, channels, device=device)
        
        # Forward pass
        outputs = highway(inputs)
        
        # Check output shape
        assert outputs.shape == inputs.shape
        assert outputs.device == device
    
    def test_conv1d(self, device):
        """Test 1D convolution."""
        batch_size = 2
        seq_len = 16
        in_channels = 8
        out_channels = 16
        kernel_size = 3
        
        # Test both padding modes
        padding_modes = ['same', 'valid']
        
        for padding in padding_modes:
            # Create layer
            conv = Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            conv.to(device)
            
            # Create input
            inputs = torch.randn(batch_size, seq_len, in_channels, device=device)
            
            # Forward pass
            outputs = conv(inputs)
            
            # Check output shape
            if padding == 'same':
                expected_length = seq_len  # Same padding preserves length
            else:  # 'valid'
                expected_length = seq_len - kernel_size + 1
                
            assert outputs.shape == (batch_size, expected_length, out_channels), \
                f"Wrong output shape for padding={padding}. Expected {(batch_size, expected_length, out_channels)}, got {outputs.shape}"
            assert outputs.device == device, \
                f"Wrong device for padding={padding}. Expected {device}, got {outputs.device}"
