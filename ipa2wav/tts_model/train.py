# -*- coding: utf-8 -*-
"""Training module for IPA2WAV synthesis model."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ipa2wav.data_processing.data_load import load_data, load_vocab
from ipa2wav.tts_model.hyperparams import Hyperparams as hp
from ipa2wav.tts_model.modules_simplified import *
from ipa2wav.tts_model.networks_simplified import TextEncoder, AudioEncoder, AudioDecoder, Attention, SSRN
from ipa2wav.utils.utils import plot_alignment, guided_attention, get_spectrograms


class Text2MelModel(nn.Module):
    """Text-to-Mel model combining encoder, attention, and decoder."""
    
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.attention = Attention()
        self.audio_decoder = AudioDecoder()
        
    def forward(self, L, S=None, monotonic_attention=False, prev_max_attentions=None):
        """Forward pass through the network.
        
        Args:
            L: Text input tensor [batch_size, text_length]
            S: Mel spectrogram tensor [batch_size, T/r, n_mels] (optional)
            monotonic_attention: Whether to use monotonic attention
            prev_max_attentions: Previous attention positions
            
        Returns:
            tuple: (Y, A, max_attentions)
                - Y: Generated mel spectrograms
                - A: Attention weights
                - max_attentions: Positions of maximum attention
        """
        # Text encoding
        K, V = self.text_encoder(L)
        
        if S is None:
            # Inference mode
            Y, A, max_attentions = self._inference(K, V, monotonic_attention, prev_max_attentions)
        else:
            # Training mode
            Y, A, max_attentions = self._training(K, V, S)
        
        return Y, A, max_attentions
    
    def _inference(self, K, V, monotonic_attention, prev_max_attentions):
        """Inference mode forward pass."""
        Y = torch.zeros(K.size(0), hp.max_T//hp.r, hp.n_mels).to(K.device)
        A = torch.zeros(K.size(0), hp.max_T//hp.r, K.size(1)).to(K.device)
        max_attentions = []
        
        for t in range(hp.max_T//hp.r):
            # Get current mel frame
            if t > 0:
                Y_t = Y[:, :t, :]
            else:
                Y_t = Y[:, 0:1, :]
            
            # Audio encoding
            Q = self.audio_encoder(Y_t)
            
            # Attention
            R, A_t, max_attention = self.attention(
                Q[:, -1:, :], K, V,
                monotonic_attention=monotonic_attention,
                prev_max_attentions=prev_max_attentions
            )
            
            # Audio decoding
            Y_t = self.audio_decoder(R)
            
            # Update tensors
            Y[:, t:t+1, :] = Y_t
            A[:, t:t+1, :] = A_t
            max_attentions.append(max_attention)
            
            # Stop if end of sequence detected
            if t > 0 and torch.all(torch.abs(Y[:, t, :] - Y[:, t-1, :]) < hp.stop_threshold):
                break
        
        return Y, A, max_attentions
    
    def _training(self, K, V, S):
        """Training mode forward pass."""
        # Audio encoding
        Q = self.audio_encoder(S)
        
        # Attention
        R, A, max_attentions = self.attention(Q, K, V)
        
        # Audio decoding
        Y = self.audio_decoder(R)
        
        return Y, A, max_attentions


class TTSDataset(Dataset):
    """Dataset for training Text-to-Speech models."""
    
    def __init__(self, model_type="text2mel"):
        """Initialize dataset.
        
        Args:
            model_type: Either "text2mel" or "ssrn"
        """
        self.model_type = model_type
        self.fpaths, self.texts = load_data("train")
        
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, idx):
        """Get a training sample.
        
        Returns:
            tuple: Different returns based on model_type:
                text2mel: (text, mel_spec)
                ssrn: (mel_spec, mag_spec)
        """
        fpath = self.fpaths[idx]
        text = torch.tensor(self.texts[idx])
        
        # Load spectrograms
        mel, mag = get_spectrograms(fpath)
        mel = torch.tensor(mel)
        mag = torch.tensor(mag)
        
        if self.model_type == "text2mel":
            return text, mel
        else:  # ssrn
            return mel, mag


def train_model(model_type="text2mel"):
    """Train either Text2Mel or SSRN model.
    
    Args:
        model_type (str): Either "text2mel" or "ssrn"
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    if model_type == "text2mel":
        model = Text2MelModel().to(device)
    else:
        model = SSRN().to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    
    # Create dataset and data loader
    train_dataset = TTSDataset(model_type)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: (
            nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
            nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True)
        )
    )
    
    # Training loop
    model.train()
    for epoch in range(hp.num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hp.num_epochs}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            if model_type == "text2mel":
                y_pred, alignments, _ = model(x, y)
                
                # Calculate losses
                mel_loss = nn.L1Loss()(y_pred, y)
                attention_loss = guided_attention(alignments)
                loss = mel_loss + hp.attention_weight * attention_loss
            else:
                y_pred = model(x)
                loss = nn.L1Loss()(y_pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Save checkpoint
            if batch_idx % hp.save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss.item()
                }
                torch.save(checkpoint, os.path.join(hp.logdir, f"{model_type}.pt"))
                
                if model_type == "text2mel":
                    # Save attention plot
                    plot_alignment(
                        alignments[0].detach().cpu().numpy(),
                        epoch * len(train_loader) + batch_idx
                    )


if __name__ == '__main__':
    # Get model type from command line (1: Text2Mel, 2: SSRN)
    model_num = int(sys.argv[1])
    model_type = "text2mel" if model_num == 1 else "ssrn"
    
    # Train model
    train_model(model_type)
