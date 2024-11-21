import pytest
import torch
from ipa2wav.tts_model.networks_simplified import TextEncoder, AudioEncoder, Attention, AudioDecoder, SSRN
from ipa2wav.tts_model.hyperparams import Hyperparams as hp


def test_text_encoder_forward():
    encoder = TextEncoder()
    L = torch.randint(0, len(hp.vocab), (2, 100))  # Batch of 2, sequence length 100
    output = encoder(L)
    assert output is not None
    # Add more assertions based on expected output shape and values


def test_audio_encoder_forward():
    encoder = AudioEncoder()
    S = torch.randn(2, 100 // hp.r, hp.n_mels)  # Batch of 2, time steps, n_mels
    output = encoder(S)
    assert output is not None
    # Add more assertions based on expected output shape and values


def test_attention_forward():
    attention = Attention()
    Q = torch.randn(2, 100 // hp.r, hp.d)
    K = torch.randn(2, 100, hp.d)
    V = torch.randn(2, 100, hp.d)
    R, alignments, max_attentions = attention(Q, K, V)
    assert R is not None
    assert alignments is not None
    assert max_attentions is not None
    # Add more assertions based on expected output shape and values


def test_audio_decoder_forward():
    decoder = AudioDecoder()
    R = torch.randn(2, 100 // hp.r, 2 * hp.d)
    output = decoder(R)
    assert output is not None
    # Add more assertions based on expected output shape and values


def test_ssrn_forward():
    ssrn = SSRN()
    Y = torch.randn(2, 100 // hp.r, hp.n_mels)
    output = ssrn(Y)
    assert output is not None
    # Add more assertions based on expected output shape and values
