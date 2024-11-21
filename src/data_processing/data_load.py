# -*- coding: utf-8 -*-
"""Data loading and preprocessing module for IPA2WAV synthesis.

This module handles data loading, text normalization, and batch generation
for training and inference. It supports various input formats and provides
robust preprocessing for IPA text input.
"""

from __future__ import print_function

from tts_model.hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from src.utils.utils import *
import codecs
import re
import os
import unicodedata

def load_vocab():
    """Load character-to-index and index-to-character mappings.
    
    Creates bidirectional mappings between characters in the vocabulary and their
    corresponding indices. This is used for converting between text and numerical
    representations.
    
    Returns:
        tuple: A pair of dictionaries:
            - char2idx: Maps characters to their indices
            - idx2char: Maps indices to their characters
    """
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    """Normalize text input for consistent processing.
    
    Performs several normalization steps:
    1. Removes diacritics and combines characters
    2. Converts to lowercase
    3. Removes characters not in vocabulary
    4. Normalizes whitespace
    
    Args:
        text (str): Input text to normalize
    
    Returns:
        str: Normalized text containing only valid characters
    """
    # Strip accents and combine characters
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                  if unicodedata.category(char) != 'Mn')
    
    # Convert to lowercase and remove invalid characters
    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    """Load dataset for training or synthesis.
    
    Loads and processes the dataset based on the specified mode. For training,
    it loads the full dataset with audio file paths and corresponding text.
    For synthesis, it processes input text for generation.
    
    Args:
        mode (str): Either "train" or "synthesize"
    
    Returns:
        tuple: Different returns based on mode:
            - Training: (fpaths, text_lengths, texts)
                - fpaths: List of audio file paths
                - text_lengths: List of text lengths
                - texts: List of encoded text sequences
            - Synthesis: (texts, text_lengths)
                - texts: List of encoded text sequences
                - text_lengths: List of text lengths
    
    Raises:
        ValueError: If mode is neither "train" nor "synthesize"
    """
    # Load vocabulary mappings
    char2idx, idx2char = load_vocab()

    if mode == "train":
        if "LJ" in hp.data:
            # Parse LJSpeech dataset
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            
            for line in lines:
                fname, text = line.strip().split("|")
                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                fpaths.append(fpath)

                # Normalize and encode text
                text = text_normalize(text) + "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

            return fpaths, text_lengths, texts
        else:
            # Add support for other datasets here
            raise NotImplementedError("Only LJSpeech dataset is currently supported")
            
    elif mode == "synthesize":
        # Load synthesize text
        if os.path.isfile(hp.test_data):
            # Load text from file
            lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()
            texts = [text_normalize(line.strip()) + "E" for line in lines]
            texts = [np.array([char2idx[char] for char in text], np.int32) for text in texts]
            text_lengths = [len(text) for text in texts]
            return texts, text_lengths
        else:
            # Process single line of text
            texts = [text_normalize(hp.test_data) + "E"]
            texts = [np.array([char2idx[char] for char in text], np.int32) for text in texts]
            text_lengths = [len(text) for text in texts]
            return texts, text_lengths
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'train' or 'synthesize'")

def get_batch():
    """Create input queues and batches for training.
    
    Sets up TensorFlow queues for efficient batch generation during training.
    Implements shuffling and dynamic padding of sequences to the maximum
    length in each batch.
    
    Returns:
        tuple: Training batch tensors:
            - texts: Batch of input text sequences
            - mels: Batch of target mel spectrograms
            - mags: Batch of target magnitude spectrograms
            - fnames: Batch of file names (for debugging)
            - num_batch: Number of batches in the dataset
    """
    # Load data
    fpaths, text_lengths, texts = load_data()

    # Calculate number of batches
    num_batch = len(fpaths) // hp.B
    
    # Create input queues
    fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

    # Parse
    text = tf.decode_raw(text, tf.int32)  # (None,)
    mel = tf.py_func(load_spectrograms, [fpath], tf.float32)  # (None, n_mels)
    mag = tf.py_func(load_spectrograms, [fpath], tf.float32)  # (None, 1+n_fft//2)

    # Add shape information
    text.set_shape((None,))
    mel.set_shape((None, hp.n_mels))
    mag.set_shape((None, 1+hp.n_fft//2))

    # Create batches
    texts, mels, mags, fpaths = tf.train.batch([text, mel, mag, fpath],
                                              shapes=[(None,), (None, hp.n_mels), (None, 1+hp.n_fft//2), ()],
                                              num_threads=8,
                                              batch_size=hp.B,
                                              capacity=hp.B*4,
                                              dynamic_pad=True)

    return texts, mels, mags, fpaths, num_batch
