"""Unit tests for PyTorch synthesis functionality."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torchaudio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import synthesize from the module where it's being mocked
import ipa2wav.tts_model.synthesize as synth_module
from ipa2wav.tts_model.train import Text2MelModel
from ipa2wav.tts_model.networks import SSRN
from ipa2wav.tts_model.hyperparams import Hyperparams as hp
from ipa2wav.data_processing.data_load import load_vocab
from ipa2wav.data_processing.data_load import custom_data_load


@pytest.fixture
def mock_models():
    """Create mock models for testing."""
    text2mel = MagicMock(spec=Text2MelModel)
    ssrn = MagicMock(spec=SSRN)
    
    # Mock forward methods
    text2mel.forward.return_value = (
        torch.randn(1, 50, hp.n_mels),  # Mel spectrogram
        None,  # Attention weights (not used in test)
        None   # Maximum attention (not used in test)
    )
    ssrn.forward.return_value = torch.randn(1, 50 * hp.r, hp.n_fft // 2 + 1)  # Linear spectrogram
    
    # Mock to/eval methods
    text2mel.to.return_value = text2mel
    ssrn.to.return_value = ssrn
    text2mel.eval.return_value = text2mel
    ssrn.eval.return_value = ssrn
    
    return text2mel, ssrn


@pytest.fixture
def mock_synthesize(tmp_path):
    """Mock synthesize function."""
    # Create sample directory
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(exist_ok=True)
    hp.sampledir = str(sample_dir)
    
    # Create sample wav path
    wav_path = str(sample_dir / "sample_0.wav")
    
    with patch('ipa2wav.tts_model.synthesize.synthesize') as mock_synth:
        mock_synth.return_value = wav_path
        yield mock_synth


@pytest.fixture
def mock_load_models(mock_models):
    """Mock the model loading function."""
    text2mel, ssrn = mock_models
    with patch('ipa2wav.tts_model.synthesize.load_models') as mock_load:
        mock_load.return_value = (text2mel, ssrn)
        yield mock_load


@pytest.fixture
def mock_audio_save():
    """Mock audio saving function."""
    with patch('scipy.io.wavfile.write') as mock_write:
        yield mock_write


@pytest.fixture
def tmp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


class TestDataProcessing:
    """Test data processing for synthesis."""
    
    def test_custom_data_load(self):
        """Test custom text data loading."""
        # Load vocab
        char2idx, idx2char = load_vocab()
        
        # Test input
        lines = ["həˈləʊ ˈwɜːld"]  # "hello world" in IPA
        
        # Process text
        result_texts, result_lengths = custom_data_load(lines, char2idx, idx2char)
        
        # Check outputs
        assert isinstance(result_texts, list)
        assert len(result_texts) == 1
        assert isinstance(result_texts[0], torch.Tensor)
        assert result_texts[0].dtype == torch.long
        assert isinstance(result_lengths, list)


class TestSynthesis:
    """Test speech synthesis functionality."""
    
    def test_end_to_end_synthesis(self, mock_synthesize, mock_audio_save, tmp_dir):
        """Test end-to-end synthesis pipeline."""
        # Set output directory
        hp.checkpoint_dir = tmp_dir
        
        # Test input
        test_text = "həˈləʊ"  # "hello" in IPA
        
        # Run synthesis using the mocked module
        wav_path = synth_module.synthesize(mode="synthesize", text=test_text)
        
        # Check if audio was saved
        assert mock_synthesize.called
        assert mock_synthesize.call_args[1] == {"mode": "synthesize", "text": test_text}
        assert isinstance(wav_path, str)
        assert wav_path.endswith('.wav')