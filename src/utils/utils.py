# -*- coding: utf-8 -*-
"""Utility functions for IPA2WAV synthesis.

This module provides various utility functions for audio processing,
spectrogram manipulation, and visualization. It includes implementations
of the Griffin-Lim algorithm and other signal processing tools.
"""

from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal

from tts_model.hyperparams import Hyperparams as hp
import tensorflow as tf


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
    # Load and trim audio
    y, sr = librosa.load(fpath, sr=hp.sr)
    y, _ = librosa.effects.trim(y)
    
    # Pre-emphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
    
    # Linear spectrogram
    linear = librosa.stft(y=y,
                         n_fft=hp.n_fft,
                         hop_length=hp.hop_length,
                         win_length=hp.win_length)
    mag = np.abs(linear)  # (1+n_fft//2, T)
    
    # Mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)
    mel = np.dot(mel_basis, mag)  # (n_mels, T)
    
    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    
    # Normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    
    return mel, mag


def spectrogram2wav(mag):
    """Convert linear spectrogram to waveform using Griffin-Lim algorithm.
    
    Args:
        mag (np.array): Linear spectrogram of shape (T, 1+n_fft/2)
    
    Returns:
        np.array: Reconstructed waveform
    """
    # Transform to linear scale
    mag = np.clip(mag, 0, 1)
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    mag = np.power(10.0, mag * 0.05)
    
    # Transpose for STFT dimensionality
    mag = mag.T  # (1+n_fft//2, T)
    
    # Griffin-Lim Algorithm
    wav = griffin_lim(mag)
    
    # Inverse pre-emphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    
    # Trim silence
    wav, _ = librosa.effects.trim(wav)
    
    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    """Applies Griffin-Lim algorithm for phase reconstruction.
    
    Iteratively recovers phase information from magnitude spectrogram
    to reconstruct time-domain signal.
    
    Args:
        spectrogram (np.array): Linear spectrogram of shape (1+n_fft/2, T)
    
    Returns:
        np.array: Reconstructed time-domain signal
    """
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)
    return y


def invert_spectrogram(spectrogram):
    """Applies inverse STFT.
    
    Args:
        spectrogram (np.array): Complex spectrogram of shape (1+n_fft/2, T)
    
    Returns:
        np.array: Time-domain signal
    """
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length)


def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots attention alignment between text and audio.
    
    Creates a heatmap visualization of the attention weights showing how
    the model aligns text with audio features.
    
    Args:
        alignment (np.array): Attention weights of shape (encoder_steps, decoder_steps)
        gs (int): Global step (used for filename)
        dir (str): Output directory for the plot
    """
    if not os.path.exists(dir): os.makedirs(dir)
    
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)
    
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.xlabel('Decoder Steps')
    plt.ylabel('Encoder Steps')
    
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')
    plt.close(fig)


def guided_attention(g=0.2):
    """Creates guided attention matrix.
    
    Implements the guided attention mechanism described in the paper,
    which encourages monotonic attention alignment.
    
    Args:
        g (float): Gaussian width parameter
    
    Returns:
        np.array: Guided attention matrix of shape (max_N, max_T/r)
    """
    W = np.zeros((hp.max_N, hp.max_T))
    for n_pos in range(hp.max_N):
        for t_pos in range(hp.max_T):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos/hp.max_T - n_pos/hp.max_N)**2 / (2*g*g))
    return W


def learning_rate_decay(init_lr, global_step, warmup_steps=4000.0):
    """Implements learning rate decay with warmup.
    
    Uses the Noam learning rate schedule from the Transformer paper,
    which increases linearly for warmup_steps, then decays proportionally
    to the inverse square root of the step number.
    
    Args:
        init_lr (float): Initial learning rate
        global_step (tf.Tensor): Current training step
        warmup_steps (float): Number of warmup steps
    
    Returns:
        tf.Tensor: Current learning rate
    """
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)


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
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = len(mag)
    
    # Reduce length to be divisible by reduction factor
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    
    return fname, mel, mag
