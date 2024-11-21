# -*- coding: utf-8 -*-
"""
Hyperparameter configuration for the IPA2WAV synthesis model.
"""

class Hyperparams:
    """Hyperparameters for the IPA2WAV synthesis model.
    
    This class contains all configurable parameters used throughout the model:
    - Signal processing parameters (sampling rate, FFT settings, etc.)
    - Model architecture parameters (embedding size, hidden units, etc.)
    - Training parameters (batch size, learning rate, etc.)
    - Data parameters (vocabulary, maximum sequence lengths, etc.)
    
    Note:
        Some parameters are critical and should not be changed without careful consideration,
        particularly the reduction factor (r) which affects the model's architecture.
    """
    
    # Pipeline Settings
    prepro = True  # If True, preprocess data before training
    
    # Signal Processing Parameters
    sr = 22050  # Sampling rate in Hz
    n_fft = 2048  # Number of FFT points
    frame_shift = 0.0125  # Frame shift in seconds
    frame_length = 0.05  # Frame length in seconds
    hop_length = int(sr * frame_shift)  # Hop length in samples (276)
    win_length = int(sr * frame_length)  # Window length in samples (1102)
    n_mels = 80  # Number of Mel frequency bands
    power = 1.5  # Magnitude spectrogram power
    n_iter = 50  # Number of Griffin-Lim iterations
    preemphasis = 0.97  # Pre-emphasis coefficient
    max_db = 100  # Maximum decibel value
    ref_db = 20  # Reference decibel value
    
    # Model Architecture Parameters
    r = 4  # Reduction factor (DO NOT CHANGE)
    dropout_rate = 0.05  # Dropout rate for regularization
    e = 128  # Embedding dimension
    d = 256  # Hidden units in Text2Mel network
    c = 512  # Hidden units in SSRN network
    attention_win_size = 3  # Attention window size
    
    # Data Parameters
    data = "../data/LJSpeech-1.1"  # Path to training data
    test_data = 'harvard_sentences_IPA.txt'  # Test dataset file
    vocab = "PE sxʃuɒhpjgm̃wŋaɛɪðnzʊbvlɑətirʒɜækʌθɔfId"  # Vocabulary (IPA symbols)
    max_N = 180  # Maximum number of input characters
    max_T = 210  # Maximum number of output mel frames
    
    # Training Parameters
    lr = 0.001  # Initial learning rate
    logdir = "logdir/LJ11"  # Directory for logging
    sampledir = 'samples_IPA_22050'  # Directory for output samples
    B = 32  # Batch size
    num_iterations = 2000000  # Total training iterations
