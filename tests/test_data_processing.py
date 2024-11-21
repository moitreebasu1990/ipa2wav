"""Unit tests for data processing functionality."""

import os
import sys
import pytest
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing.data_load import load_vocab, text_normalize, load_data
from src.tts_model.hyperparams import Hyperparams as hp

class TestDataLoading:
    """Test data loading and preprocessing functions."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "həˈləʊ ˈwɜːld"  # "hello world" in IPA
    
    @pytest.fixture
    def vocab_path(self, tmp_path):
        """Create a temporary vocabulary file."""
        vocab_file = tmp_path / "vocab.txt"
        vocab_content = "PE sxʃuɒhpjgm̃wŋaɛɪðnzʊbvlɑətirʒɜækʌθɔfId"
        vocab_file.write_text(vocab_content)
        return str(vocab_file)
    
    def test_text_normalize(self, sample_text):
        """Test text normalization function."""
        normalized = text_normalize(sample_text)
        assert isinstance(normalized, str)
        assert normalized.strip() != ""
        assert all(c in hp.vocab for c in normalized)
    
    def test_load_vocab(self, vocab_path):
        """Test vocabulary loading."""
        char2idx, idx2char = load_vocab()
        
        # Check mappings are dictionaries
        assert isinstance(char2idx, dict)
        assert isinstance(idx2char, dict)
        
        # Check mappings are inverses
        for char, idx in char2idx.items():
            assert idx2char[idx] == char
        
        # Check special tokens
        assert 'P' in char2idx  # Padding token
        assert 'E' in char2idx  # End of string token
    
    def test_load_data(self, tmp_path):
        """Test data loading functionality."""
        # Create sample data file
        data_file = tmp_path / "test.txt"
        data_file.write_text("test1|həˈləʊ\ntest2|ˈwɜːld")
        
        # Test loading
        texts, text_lengths = load_data(mode="train", data_path=str(data_file))
        
        # Check outputs
        assert isinstance(texts, list)
        assert isinstance(text_lengths, list)
        assert len(texts) == len(text_lengths)
        assert all(isinstance(t, np.ndarray) for t in texts)
        assert all(isinstance(l, int) for l in text_lengths)
    
    def test_invalid_text_normalize(self):
        """Test text normalization with invalid input."""
        with pytest.raises(ValueError):
            text_normalize("")
        
        with pytest.raises(ValueError):
            text_normalize(None)
    
    def test_batch_generation(self):
        """Test batch generation functionality."""
        # Create sample data
        texts = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        text_lengths = [3, 3]
        
        # Create batch
        batch = tf.data.Dataset.from_tensor_slices((texts, text_lengths))
        batch = batch.batch(2)
        
        # Check batch properties
        for text_batch, length_batch in batch:
            assert text_batch.shape[0] <= 2  # Batch size
            assert length_batch.shape[0] <= 2
            assert len(text_batch.shape) == 2  # 2D tensor [batch_size, seq_len]
    
    def test_text_length_consistency(self):
        """Test consistency between text and length values."""
        texts = [np.array([1, 2, 3]), np.array([4, 5, 6, 7])]
        text_lengths = [3, 4]
        
        # Verify lengths match actual sequence lengths
        for text, length in zip(texts, text_lengths):
            assert len(text) == length
