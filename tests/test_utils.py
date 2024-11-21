"""Unit tests for utility functions."""

import os
import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
import torchaudio
import torchaudio.transforms as T

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipa2wav.utils.utils import (
    get_spectrograms, 
    spectrogram2wav,
    guided_attention, 
    load_spectrograms
)
from ipa2wav.tts_model.hyperparams import Hyperparams as hp


class TestAudioProcessing:
    """Test audio processing utilities."""
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for testing."""
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "test.wav"
        
        # Generate sample audio data
        sample_rate = 22050
        duration = 1  # seconds
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Ensure audio is in correct format (channels, samples)
        audio = audio.unsqueeze(0)  # Add channels dimension
        audio = audio.float()  # Ensure float32 dtype
        
        # Save audio file with explicit format
        torchaudio.save(
            str(temp_file),
            audio,
            sample_rate,
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16
        )
        
        yield str(temp_file)
        
        # Cleanup
        os.remove(temp_file)
        os.rmdir(temp_dir)
    
    def test_get_spectrograms(self, sample_audio_file):
        """Test spectrogram extraction."""
        mel, mag = get_spectrograms(sample_audio_file)
        
        # Check shapes
        assert isinstance(mel, np.ndarray)
        assert isinstance(mag, np.ndarray)
        assert mel.shape[1] == hp.n_mels
        assert mag.shape[1] == hp.n_fft // 2 + 1
        
        # Check range
        assert np.all(mel >= 0)
        assert np.all(mag >= 0)
    
    def test_spectrogram2wav(self, sample_audio_file):
        """Test waveform reconstruction from spectrogram."""
        # Get original spectrograms
        _, mag = get_spectrograms(sample_audio_file)
        
        # Reconstruct waveform
        reconstructed = spectrogram2wav(mag)
        
        # Check shape and type
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.ndim == 1
        assert len(reconstructed) > 0
        
        # Check range
        assert np.abs(reconstructed).max() <= 1.0
    
    def test_guided_attention(self):
        """Test guided attention matrix generation."""
        g = 0.2  # Gaussian width
        N, T = 10, 20  # Text and audio sequence lengths
        
        W = guided_attention()
        
        # Check shape
        assert isinstance(W, torch.Tensor)
        assert W.shape == (hp.max_N, hp.max_T // hp.r)
        
        # Check range
        assert torch.all(W >= 0)
        assert torch.all(W <= 1)
        
        # Check diagonal pattern
        diag_mean = torch.diagonal(W).mean()
        off_diag_mean = (W.sum() - torch.diagonal(W).sum()) / (W.numel() - W.shape[0])
        assert diag_mean > off_diag_mean
    
    def test_load_spectrograms(self, sample_audio_file):
        """Test combined spectrogram loading."""
        mel, mag = load_spectrograms(sample_audio_file)
        
        # Check output types and shapes
        assert isinstance(mel, np.ndarray)
        assert isinstance(mag, np.ndarray)
        assert mel.shape[1] == hp.n_mels
        assert mag.shape[1] == hp.n_fft // 2 + 1
        
        # Check time dimension alignment
        assert mel.shape[0] * hp.r == mag.shape[0]
        
        # Check range
        assert np.all(mel >= 0)
        assert np.all(mag >= 0)
