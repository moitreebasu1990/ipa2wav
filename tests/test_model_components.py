"""Unit tests for model components and layers."""

import os
import sys
import pytest
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tts_model.modules import embed, normalize, highwaynet, conv1d, conv1d_transpose
from src.tts_model.networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
from src.tts_model.hyperparams import Hyperparams as hp

class TestModules:
    """Test basic neural network modules."""
    
    @pytest.fixture
    def session(self):
        """Create TensorFlow session for testing."""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        yield sess
        sess.close()
    
    def test_embedding(self, session):
        """Test embedding layer."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10
        num_units = 8
        
        # Create input
        inputs = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=tf.int32)
        
        # Apply embedding
        outputs = embed(inputs, vocab_size, num_units)
        
        # Initialize variables
        session.run(tf.global_variables_initializer())
        
        # Run embedding
        output_val = session.run(outputs)
        
        # Check output shape
        assert output_val.shape == (batch_size, seq_len, num_units)
    
    def test_normalize(self, session):
        """Test layer normalization."""
        inputs = tf.random_normal([2, 4, 8])
        outputs = normalize(inputs)
        
        session.run(tf.global_variables_initializer())
        output_val = session.run(outputs)
        
        # Check shape preservation
        assert output_val.shape == inputs.shape.as_list()
        
        # Check normalization properties
        mean = np.mean(output_val, axis=-1)
        std = np.std(output_val, axis=-1)
        assert np.allclose(mean, 0, atol=1e-6)
        assert np.allclose(std, 1, atol=1)
    
    def test_highway_net(self, session):
        """Test highway network."""
        inputs = tf.random_normal([2, 4, 8])
        outputs = highwaynet(inputs)
        
        session.run(tf.global_variables_initializer())
        output_val = session.run(outputs)
        
        # Check shape preservation
        assert output_val.shape == inputs.shape.as_list()
    
    def test_conv1d(self, session):
        """Test 1D convolution."""
        inputs = tf.random_normal([2, 8, 16])  # [batch_size, time_steps, channels]
        outputs = conv1d(inputs, filters=32, size=3)
        
        session.run(tf.global_variables_initializer())
        output_val = session.run(outputs)
        
        # Check output shape
        assert output_val.shape[0] == 2  # batch_size
        assert output_val.shape[2] == 32  # filters
    
    def test_conv1d_transpose(self, session):
        """Test transposed 1D convolution."""
        inputs = tf.random_normal([2, 8, 16])
        outputs = conv1d_transpose(inputs, filters=32)
        
        session.run(tf.global_variables_initializer())
        output_val = session.run(outputs)
        
        # Check output shape
        assert output_val.shape[0] == 2  # batch_size
        assert output_val.shape[1] == 16  # 2x time_steps
        assert output_val.shape[2] == 32  # filters


class TestNetworks:
    """Test main network components."""
    
    @pytest.fixture
    def session(self):
        """Create TensorFlow session for testing."""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        yield sess
        sess.close()
    
    def test_text_encoder(self, session):
        """Test text encoder network."""
        batch_size = 2
        seq_len = 10
        inputs = tf.random_uniform([batch_size, seq_len], maxval=len(hp.vocab), dtype=tf.int32)
        
        K, V = TextEnc(inputs, training=True)
        
        session.run(tf.global_variables_initializer())
        K_val, V_val = session.run([K, V])
        
        # Check output shapes
        assert K_val.shape[0] == batch_size
        assert K_val.shape[1] == seq_len
        assert K_val.shape[2] == hp.d  # Hidden dimension
        assert V_val.shape == K_val.shape
    
    def test_audio_encoder(self, session):
        """Test audio encoder network."""
        batch_size = 2
        time_steps = 16
        inputs = tf.random_normal([batch_size, time_steps, hp.n_mels])
        
        Q = AudioEnc(inputs, training=True)
        
        session.run(tf.global_variables_initializer())
        Q_val = session.run(Q)
        
        # Check output shape
        assert Q_val.shape[0] == batch_size
        assert Q_val.shape[2] == hp.d  # Hidden dimension
    
    def test_attention_mechanism(self, session):
        """Test attention mechanism."""
        batch_size = 2
        query_len = 8
        key_len = 10
        
        # Create inputs
        Q = tf.random_normal([batch_size, query_len, hp.d])
        K = tf.random_normal([batch_size, key_len, hp.d])
        V = tf.random_normal([batch_size, key_len, hp.d])
        
        # Apply attention
        R, alignments, max_attentions = Attention(Q, K, V, 
                                                mononotic_attention=False,
                                                prev_max_attentions=tf.zeros([batch_size]))
        
        session.run(tf.global_variables_initializer())
        R_val, align_val = session.run([R, alignments])
        
        # Check output shapes
        assert R_val.shape[0] == batch_size
        assert R_val.shape[1] == query_len
        assert R_val.shape[2] == hp.d * 2  # Concatenated context
        
        # Check alignment properties
        assert align_val.shape == (batch_size, key_len, query_len)
        assert np.allclose(np.sum(align_val, axis=1), 1.0)  # Attention weights sum to 1
    
    def test_ssrn(self, session):
        """Test SSRN (Spectrogram Super-resolution Network)."""
        batch_size = 2
        time_steps = 16
        inputs = tf.random_normal([batch_size, time_steps, hp.n_mels])
        
        Z = SSRN(inputs, training=True)
        
        session.run(tf.global_variables_initializer())
        Z_val = session.run(Z)
        
        # Check output shape
        expected_freq_bins = hp.n_fft // 2 + 1
        assert Z_val.shape[0] == batch_size
        assert Z_val.shape[2] == expected_freq_bins  # Frequency bins
