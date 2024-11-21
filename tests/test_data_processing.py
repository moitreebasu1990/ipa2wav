"""Unit tests for data processing functionality."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipa2wav.data_processing.data_load import load_vocab, text_normalize, load_data
from ipa2wav.tts_model.hyperparams import Hyperparams as hp


class TestDataLoading:
    """Test data loading and preprocessing functions."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "həˈləʊ ˈwɜːld"  # "hello world" in IPA
    
    def test_load_vocab(self):
        """Test vocabulary loading."""
        char2idx, idx2char = load_vocab()
        
        # Check type and content
        assert isinstance(char2idx, dict)
        assert isinstance(idx2char, dict)
        assert len(char2idx) == len(idx2char)
        assert all(isinstance(k, str) for k in char2idx.keys())
        assert all(isinstance(v, int) for v in char2idx.values())
        
        # Check bidirectional mapping
        for char, idx in char2idx.items():
            assert idx2char[idx] == char
    
    def test_text_normalize(self, sample_text):
        """Test text normalization."""
        normalized = text_normalize(sample_text)
        
        # Check type
        assert isinstance(normalized, str)
        
        # Check content
        assert all(c in hp.vocab for c in normalized)
        assert len(normalized) > 0
        
        # Check specific transformations
        assert normalized.islower()
    
    def test_load_data_train(self, tmp_path):
        """Test data loading in training mode."""
        # Create temporary data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        hp.data_dir = str(data_dir)
        
        # Create sample transcript
        transcript = data_dir / "transcript.txt"
        transcript.write_text("sample.wav|həˈləʊ ˈwɜːld\n")
        
        # Create sample audio file
        sample_audio = data_dir / "sample.wav"
        sample_audio.touch()
        
        # Test loading
        fpaths, texts = load_data(mode="train")
        
        # Check types and content
        assert isinstance(fpaths, list)
        assert isinstance(texts, list)
        assert len(fpaths) == len(texts)
        assert all(isinstance(text, np.ndarray) for text in texts)
    
    def test_load_data_synthesize(self):
        """Test data loading in synthesis mode."""
        fpaths, texts = load_data(mode="synthesize")
        
        # Check empty lists for synthesis mode
        assert isinstance(fpaths, list)
        assert isinstance(texts, list)
        assert len(fpaths) == 0
        assert len(texts) == 0
    
    def test_invalid_mode(self):
        """Test error handling for invalid mode."""
        with pytest.raises(ValueError):
            load_data(mode="invalid")
