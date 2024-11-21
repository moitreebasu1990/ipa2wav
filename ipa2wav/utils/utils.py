# -*- coding: utf-8 -*-
"""Utility functions for IPA2WAV synthesis.

This module provides various utility functions for audio processing,
spectrogram manipulation, and visualization. It includes implementations
of the Griffin-Lim algorithm and other signal processing tools.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy import signal

from ipa2wav.tts_model.hyperparams import Hyperparams as hp


def get_spectrograms(fpath):
    """Extract mel and linear spectrograms from an audio file.
    
    Processes an audio file through several steps:
    1. Load and trim audio
    2. Apply pre-emphasis
    3. Compute linear spectrogram
    4. Convert to mel-scale
    5. Apply power law compression
    
    Args:
        fpath (str): Path to the audio file
    
    Returns:
        tuple: Two numpy arrays:
            - mel: Mel-spectrogram of shape (T, n_mels)
            - mag: Linear spectrogram of shape (T, 1+n_fft/2)
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(fpath)
    if sample_rate != hp.sr:
        resampler = T.Resample(sample_rate, hp.sr)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Trim silence
    waveform = torch.nn.functional.pad(waveform, (1, 0))
    waveform = waveform[:, 1:] - hp.preemphasis * waveform[:, :-1]
    
    # Compute spectrograms
    stft = T.Spectrogram(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0  # magnitude spectrogram
    )
    mel_scale = T.MelScale(
        n_mels=hp.n_mels,
        sample_rate=hp.sr,
        n_stft=hp.n_fft // 2 + 1
    )
    
    # Linear spectrogram
    mag = stft(waveform)  # (1, 1+n_fft/2, T)
    mag = mag.squeeze(0).T  # (T, 1+n_fft/2)
    
    # Mel spectrogram
    mel = mel_scale(mag.T.unsqueeze(0))  # (1, n_mels, T)
    mel = mel.squeeze(0).T  # (T, n_mels)
    
    # Convert to numpy and apply power law compression
    mel = mel.numpy()
    mag = mag.numpy()
    mel = np.power(mel, hp.power)
    mag = np.power(mag, hp.power)
    
    return mel, mag


def spectrogram2wav(mag):
    """Convert linear spectrogram to waveform using Griffin-Lim algorithm.
    
    Args:
        mag (np.array): Linear spectrogram of shape (T, 1+n_fft/2)
    
    Returns:
        np.array: Reconstructed waveform
    """
    # Convert to torch tensor
    mag = torch.from_numpy(mag).T.unsqueeze(0)  # (1, 1+n_fft/2, T)
    
    # Initialize Griffin-Lim
    griffin_lim = T.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=hp.power,
        n_iter=hp.n_iter_griffin
    )
    
    # Reconstruct waveform
    waveform = griffin_lim(mag)
    
    # Apply inverse pre-emphasis
    waveform = torch.nn.functional.pad(waveform, (1, 0))
    waveform = waveform[:, 1:] + hp.preemphasis * waveform[:, :-1]
    
    return waveform.squeeze(0).numpy()


def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots attention alignment between text and audio.
    
    Creates a heatmap visualization of the attention weights showing how
    the model aligns text with audio features.
    
    Args:
        alignment (np.array): Attention weights of shape (encoder_steps, decoder_steps)
        gs (int): Global step (used for filename)
        dir (str): Output directory for the plot
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots()
    im = ax.imshow(alignment, aspect='auto', origin='lower')
    fig.colorbar(im)
    plt.xlabel('Decoder timestep')
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(dir / f'alignment_{gs}.png', format='png')
    plt.close()


def guided_attention(g=0.2):
    """Creates guided attention matrix.
    
    Implements the guided attention mechanism described in the paper,
    which encourages monotonic attention alignment.
    
    Args:
        g (float): Gaussian width parameter
    
    Returns:
        torch.Tensor: Guided attention matrix of shape (max_N, max_T/r)
    """
    # Create position tensors
    N = torch.arange(hp.max_N, dtype=torch.float32)
    T = torch.arange(hp.max_T // hp.r, dtype=torch.float32)
    
    # Normalize positions to [0, 1]
    N = N.unsqueeze(1) / hp.max_N  # Shape: (max_N, 1)
    T = T.unsqueeze(0) / (hp.max_T // hp.r)  # Shape: (1, max_T//r)
    
    # Compute attention weights using vectorized operations
    W = 1.0 - torch.exp(-((N - T) ** 2) / (2 * g ** 2))
    
    return W


def load_spectrograms(fpath):
    """Load mel and linear spectrograms from audio file.
    
    A wrapper around get_spectrograms that handles file loading and
    returns spectrograms in the format expected by the model.
    
    Args:
        fpath (str): Path to the audio file
    
    Returns:
        tuple: Two numpy arrays:
            - mel: Mel-spectrogram
            - mag: Linear spectrogram
    """
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    
    # Pad
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    
    # Reduction
    mel = mel[::hp.r, :]
    return mel, mag
