"""Unit tests for utility functions."""

import os
import sys
import pytest
import numpy as np
import librosa
import soundfile as sf
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.utils import (get_spectrograms, spectrogram2wav, griffin_lim,
                           invert_spectrogram, guided_attention)
from src.tts_model.hyperparams import Hyperparams as hp


class TestAudioProcessing:
    """Test audio processing utilities."""
    
    @pytest.fixture
    def sample_audio(self):
        """Create a sample audio signal."""
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(hp.sr * duration))
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        return signal
    
    @pytest.fixture
    def audio_file(self, sample_audio, tmp_path):
        """Create a temporary audio file."""
        file_path = tmp_path / "test.wav"
        sf.write(file_path, sample_audio, hp.sr)
        return str(file_path)
    
    def test_get_spectrograms(self, audio_file):
        """Test spectrogram extraction."""
        mel, mag = get_spectrograms(audio_file)
        
        # Check output types
        assert isinstance(mel, np.ndarray)
        assert isinstance(mag, np.ndarray)
        
        # Check shapes
        assert len(mel.shape) == 2  # [time, n_mels]
        assert len(mag.shape) == 2  # [time, 1+n_fft//2]
        assert mel.shape[1] == hp.n_mels
        assert mag.shape[1] == hp.n_fft // 2 + 1
        
        # Check value ranges
        assert np.all(mel >= 0)
        assert np.all(mag >= 0)
        assert np.all(mel <= 1)
        assert np.all(mag <= 1)
    
    def test_spectrogram2wav(self, sample_audio):
        """Test waveform reconstruction from spectrogram."""
        # Get spectrogram
        S = librosa.stft(sample_audio, n_fft=hp.n_fft,
                        hop_length=hp.hop_length,
                        win_length=hp.win_length)
        mag = np.abs(S)
        
        # Reconstruct waveform
        wav = spectrogram2wav(mag)
        
        # Check output type and shape
        assert isinstance(wav, np.ndarray)
        assert wav.ndim == 1
        assert wav.dtype == np.float32
        
        # Check basic signal properties
        assert len(wav) > 0
        assert np.abs(wav).max() <= 1.0
    
    def test_griffin_lim(self):
        """Test Griffin-Lim algorithm."""
        # Create a simple spectrogram
        n_frames = 50
        n_freqs = hp.n_fft // 2 + 1
        mag = np.random.rand(n_freqs, n_frames)
        
        # Apply Griffin-Lim
        wav = griffin_lim(mag)
        
        # Check output
        assert isinstance(wav, np.ndarray)
        assert wav.ndim == 1
        assert len(wav) > 0
    
    def test_invert_spectrogram(self):
        """Test spectrogram inversion."""
        # Create a simple spectrogram
        n_frames = 50
        n_freqs = hp.n_fft // 2 + 1
        spec = np.random.complex64(np.random.rand(n_freqs, n_frames) + 
                                 1j * np.random.rand(n_freqs, n_frames))
        
        # Invert spectrogram
        wav = invert_spectrogram(spec)
        
        # Check output
        assert isinstance(wav, np.ndarray)
        assert wav.ndim == 1
        assert len(wav) > 0
    
    def test_guided_attention(self):
        """Test guided attention matrix generation."""
        W = guided_attention(g=0.2)
        
        # Check shape
        assert W.shape == (hp.max_N, hp.max_T)
        
        # Check value range
        assert np.all(W >= 0)
        assert np.all(W <= 1)
        
        # Check diagonal pattern
        for i in range(hp.max_N):
            for j in range(hp.max_T):
                if i/hp.max_N == j/hp.max_T:
                    assert W[i,j] < 0.5  # Lower values along diagonal
    
    def test_audio_reconstruction_pipeline(self, sample_audio):
        """Test full audio processing pipeline."""
        # Forward transform
        S = librosa.stft(sample_audio, n_fft=hp.n_fft,
                        hop_length=hp.hop_length,
                        win_length=hp.win_length)
        mag = np.abs(S)
        
        # Reconstruction
        wav_reconstructed = spectrogram2wav(mag)
        
        # Compare lengths
        expected_length = len(sample_audio)
        reconstructed_length = len(wav_reconstructed)
        length_diff = abs(expected_length - reconstructed_length)
        
        # Allow for small length differences due to framing
        assert length_diff <= hp.hop_length
        
        # Compare energy
        energy_original = np.sum(sample_audio**2)
        energy_reconstructed = np.sum(wav_reconstructed**2)
        energy_ratio = energy_reconstructed / energy_original
        
        # Energy should be roughly preserved
        assert 0.1 < energy_ratio < 10
    
    def test_invalid_audio_file(self):
        """Test handling of invalid audio file."""
        with pytest.raises(Exception):
            get_spectrograms("nonexistent_file.wav")
    
    def test_empty_spectrogram(self):
        """Test handling of empty spectrogram."""
        empty_mag = np.zeros((hp.n_fft // 2 + 1, 0))
        with pytest.raises(Exception):
            spectrogram2wav(empty_mag)
