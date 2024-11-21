# -*- coding: utf-8 -*-
"""Data loading and preprocessing module for IPA2WAV synthesis.

This module handles data loading, text normalization, and batch generation
for training and inference. It supports various input formats and provides
robust preprocessing for IPA text input.
"""

import codecs
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

from ipa2wav.tts_model.hyperparams import Hyperparams as hp
from ipa2wav.utils.utils import get_spectrograms


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
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(f"[^{hp.vocab}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    if mode not in ["train", "synthesize"]:
        raise ValueError("Mode must be either 'train' or 'synthesize'")
        
    # Load vocabulary mappings
    char2idx, _ = load_vocab()
    
    if mode == "train":
        # Read training data
        fpaths, texts = [], []
        transcript = os.path.join(hp.data_dir, 'transcript.txt')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        for line in lines:
            fname, text = line.strip().split('|')
            fpath = os.path.join(hp.data_dir, fname)
            text = text_normalize(text) + hp.end_token  # Add end token
            fpaths.append(fpath)
            texts.append(np.array([char2idx[char] for char in text], np.int32))
        return fpaths, texts
    else:
        # Return empty lists for synthesis mode
        return [], []


def custom_data_load(lines, char2idx, idx2char):
    """Load and process custom input text for synthesis.
    
    Args:
        lines (list): List of input text lines
        char2idx (dict): Character to index mapping
        idx2char (dict): Index to character mapping
    
    Returns:
        tuple: (texts, text_lengths)
            - texts: List of torch.Tensor, each containing encoded text sequence
            - text_lengths: List of sequence lengths
    """
    texts = []
    text_lengths = []
    
    for text in lines:
        # Normalize text
        text = text_normalize(text)
        
        # Convert characters to indices
        indices = []
        for char in text:
            if char in char2idx:
                indices.append(char2idx[char])
        
        if indices:  # Only process if we have valid characters
            # Convert to tensor
            text_tensor = torch.tensor(indices, dtype=torch.long)
            texts.append(text_tensor)
            text_lengths.append(len(indices))
    
    return texts, text_lengths


def get_batch():
    """Create input queues and batches for training.
    
    Sets up PyTorch DataLoader for efficient batch generation during training.
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
    # Load training data
    fpaths, texts = load_data(mode="train")
    
    # Create dataset
    dataset = []
    for fpath, text in zip(fpaths, texts):
        # Load spectrograms
        mel, mag = get_spectrograms(fpath)
        dataset.append((torch.tensor(text), torch.tensor(mel), torch.tensor(mag), fpath))
    
    # Create data loader
    from torch.utils.data import DataLoader, Dataset
    
    class TTSDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    loader = DataLoader(
        TTSDataset(dataset),
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=lambda batch: (
            torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
            torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True),
            torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True),
            [x[3] for x in batch]
        )
    )
    
    return loader, len(dataset) // hp.batch_size
