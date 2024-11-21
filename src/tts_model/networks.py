# -*- coding: utf-8 -*-
"""Neural network architectures for IPA2WAV synthesis.

This module defines the core network architectures:
1. Text2Mel: Converts IPA text to mel-spectrograms
2. SSRN: Converts mel-spectrograms to linear spectrograms
"""

from __future__ import print_function

from .hyperparams import Hyperparams as hp
from .modules import *
import tensorflow as tf

def TextEnc(L, training=True):
    """Text Encoder network.
    
    Converts input text sequences into keys and values for attention mechanism.
    Architecture:
    1. Character embedding
    2. 1D convolution layers
    3. Highway convolution blocks
    
    Args:
        L (tf.Tensor): Text inputs. Shape: [batch_size, sequence_length]
        training (bool): Whether the network is in training mode. Affects dropout.
    
    Returns:
        tuple: A pair of tensors:
            - K: Keys for attention. Shape: [batch_size, seq_length, hidden_size]
            - V: Values for attention. Shape: [batch_size, seq_length, hidden_size]
    """
    i = 1
    # Embedding layer
    tensor = embed(L,
                  vocab_size=len(hp.vocab),
                  num_units=hp.e,
                  scope="embed_{}".format(i)); i += 1
    
    # Initial convolution layers
    tensor = conv1d(tensor,
                   filters=2*hp.d,
                   size=1,
                   rate=1,
                   dropout_rate=hp.dropout_rate,
                   activation_fn=tf.nn.relu,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                   size=1,
                   rate=1,
                   dropout_rate=hp.dropout_rate,
                   training=training,
                   scope="C_{}".format(i)); i += 1

    # Highway convolution blocks
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                       size=3,
                       rate=3**j,
                       dropout_rate=hp.dropout_rate,
                       activation_fn=None,
                       training=training,
                       scope="HC_{}".format(i)); i += 1
    
    for _ in range(2):
        tensor = hc(tensor,
                   size=3,
                   rate=1,
                   dropout_rate=hp.dropout_rate,
                   activation_fn=None,
                   training=training,
                   scope="HC_{}".format(i)); i += 1

    K, V = tf.split(tensor, 2, -1)
    return K, V

def AudioEnc(S, training=True):
    """Audio Encoder network.
    
    Processes mel-spectrograms into queries for attention mechanism.
    Uses a series of convolutional layers with highway connections.
    
    Args:
        S (tf.Tensor): Mel-spectrogram inputs. Shape: [batch_size, time_steps/r, n_mels]
        training (bool): Whether the network is in training mode. Affects dropout.
    
    Returns:
        tf.Tensor: Queries for attention. Shape: [batch_size, time_steps/r, hidden_size]
    """
    i = 1
    tensor = conv1d(S,
                   filters=hp.d,
                   size=1,
                   rate=1,
                   padding="CAUSAL",
                   dropout_rate=hp.dropout_rate,
                   activation_fn=tf.nn.relu,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                   size=1,
                   rate=1,
                   padding="CAUSAL",
                   dropout_rate=hp.dropout_rate,
                   activation_fn=tf.nn.relu,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                   size=1,
                   rate=1,
                   padding="CAUSAL",
                   dropout_rate=hp.dropout_rate,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                       size=3,
                       rate=3**j,
                       padding="CAUSAL",
                       dropout_rate=hp.dropout_rate,
                       training=training,
                       scope="HC_{}".format(i)); i += 1
    
    return tensor

def Attention(Q, K, V, monotonic_attention=False, prev_max_attentions=None):
    """Multi-head attention mechanism with optional monotonic constraint.
    
    Computes attention between queries from the audio encoder and keys/values
    from the text encoder. Can enforce monotonic attention during inference.
    
    Args:
        Q (tf.Tensor): Queries from audio encoder. Shape: [batch_size, time_steps/r, hidden_size]
        K (tf.Tensor): Keys from text encoder. Shape: [batch_size, seq_length, hidden_size]
        V (tf.Tensor): Values from text encoder. Shape: [batch_size, seq_length, hidden_size]
        monotonic_attention (bool): Whether to enforce monotonic attention (for inference)
        prev_max_attentions (tf.Tensor, optional): Previous attention peaks for monotonic attention
    
    Returns:
        tuple: A tuple containing:
            - R: Context vectors concatenated with queries. Shape: [batch_size, time_steps/r, 2*hidden_size]
            - alignments: Attention weights. Shape: [batch_size, seq_length, time_steps/r]
            - max_attentions: Positions of maximum attention. Shape: [batch_size, time_steps/r]
    """
    # Attention
    A = tf.matmul(Q, K, transpose_b=True) * tf.rsqrt(tf.cast(hp.d, tf.float32))
    
    if monotonic_attention and prev_max_attentions is not None:
        # Enforce monotonic attention
        key_masks = tf.sequence_mask(prev_max_attentions, tf.shape(K)[1])
        reverse_masks = tf.sequence_mask(tf.shape(K)[1]-prev_max_attentions-1, tf.shape(K)[1])[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, tf.shape(Q)[1], 1])
        A = tf.where(masks, A, -2**32+1)  # Force attention to be monotonic
        
    # Softmax attention weights
    A = tf.nn.softmax(A)
    
    # Compute context vectors
    R = tf.matmul(A, V)
    
    # Concatenate with queries
    R = tf.concat((R, Q), -1)
    
    # Get maximum attention positions
    max_attentions = tf.argmax(A, -1)
    
    alignments = tf.transpose(A, [0, 2, 1]) # (B, N, T/r)
    
    return R, alignments, max_attentions

def AudioDec(R, training=True):
    """Audio Decoder network.
    
    Generates mel-spectrogram predictions from context vectors and queries.
    Uses a series of convolutional layers with highway connections.
    
    Args:
        R (tf.Tensor): Context and query vectors. Shape: [batch_size, time_steps/r, 2*hidden_size]
        training (bool): Whether the network is in training mode. Affects dropout.
    
    Returns:
        tf.Tensor: Predicted mel-spectrograms. Shape: [batch_size, time_steps/r, n_mels]
    """
    i = 1
    tensor = conv1d(R,
                   filters=hp.d,
                   size=1,
                   rate=1,
                   padding="CAUSAL",
                   dropout_rate=hp.dropout_rate,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    
    for j in range(4):
        tensor = hc(tensor,
                   size=3,
                   rate=3**j,
                   padding="CAUSAL",
                   dropout_rate=hp.dropout_rate,
                   training=training,
                   scope="HC_{}".format(i)); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                   size=3,
                   rate=1,
                   padding="CAUSAL",
                   dropout_rate=hp.dropout_rate,
                   training=training,
                   scope="HC_{}".format(i)); i += 1
    
    tensor = conv1d(tensor,
                   filters=hp.n_mels,
                   size=1,
                   rate=1,
                   padding="CAUSAL",
                   dropout_rate=hp.dropout_rate,
                   activation_fn=tf.nn.sigmoid,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    
    return tensor

def SSRN(Y, training=True):
    """Spectrogram Super-resolution Network (SSRN).
    
    Converts mel-spectrograms to full linear-scale spectrograms by:
    1. Upsampling in time
    2. Expanding frequency resolution
    
    Args:
        Y (tf.Tensor): Mel-spectrogram inputs. Shape: [batch_size, time_steps/r, n_mels]
        training (bool): Whether the network is in training mode. Affects dropout.
    
    Returns:
        tf.Tensor: Predicted linear spectrograms. Shape: [batch_size, time_steps, 1+n_fft/2]
    """
    i = 1
    tensor = conv1d(Y,
                   filters=hp.c,
                   size=1,
                   rate=1,
                   dropout_rate=hp.dropout_rate,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    
    # Upsampling layers
    for j in range(2):
        tensor = conv1d(tensor,
                       filters=hp.c,
                       size=1,
                       rate=1,
                       dropout_rate=hp.dropout_rate,
                       training=training,
                       scope="C_{}".format(i)); i += 1
        tensor = conv1d_transpose(tensor,
                                scope="D_{}".format(i),
                                dropout_rate=hp.dropout_rate,
                                training=training); i += 1

    # Additional convolution layers
    for j in range(2):
        tensor = conv1d(tensor,
                       filters=hp.c,
                       size=1,
                       rate=1,
                       dropout_rate=hp.dropout_rate,
                       training=training,
                       scope="C_{}".format(i)); i += 1
        tensor = conv1d(tensor,
                       filters=hp.c,
                       size=1,
                       rate=1,
                       dropout_rate=hp.dropout_rate,
                       training=training,
                       scope="C_{}".format(i)); i += 1
        tensor = hc(tensor,
                   size=3,
                   rate=3**j,
                   dropout_rate=hp.dropout_rate,
                   training=training,
                   scope="HC_{}".format(i)); i += 1

    # Final convolution to match spectrogram dimensions
    tensor = conv1d(tensor,
                   filters=hp.n_fft//2+1,
                   size=1,
                   rate=1,
                   dropout_rate=hp.dropout_rate,
                   activation_fn=tf.nn.sigmoid,
                   training=training,
                   scope="C_{}".format(i)); i += 1
    
    return tensor
