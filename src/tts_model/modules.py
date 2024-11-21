# -*- coding: utf-8 -*-
"""Neural network building blocks for IPA2WAV synthesis.

This module provides the fundamental neural network components used throughout
the synthesis system, including embedding layers, normalization, convolutions,
and highway networks.
"""

from __future__ import print_function, division

import tensorflow as tf


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    """Embeds a given tensor of ids into a dense vector representation.
    
    Creates and applies a trainable embedding lookup table to convert integer IDs
    into dense vectors. Optionally zero-pads the first row for padding tokens.
    
    Args:
        inputs: A `Tensor` with type `int32` or `int64` containing the ids
            to be looked up in the embedding table.
        vocab_size: An int. Size of the vocabulary (number of unique tokens).
        num_units: An int. Dimension of the embedding vectors.
        zero_pad: A boolean. If True, all values in the first row (id 0)
            will be constant zeros, typically used for padding tokens.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse existing weights.
    
    Returns:
        A `Tensor` of shape [batch_size, sequence_length, num_units] containing
        the embedded vectors.
    """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                     dtype=tf.float32, 
                                     shape=[vocab_size, num_units],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                    lookup_table[1:, :]), 0)
        
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
    return outputs


def normalize(inputs, scope="normalize", reuse=None):
    """Applies layer normalization to the input tensor.
    
    Normalizes the last dimension of the input tensor using layer normalization,
    which helps stabilize training by normalizing the activations of each layer.
    
    Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension is
            batch_size. Normalization is applied over the last dimension.
        scope: Optional scope for variable_scope.
        reuse: Boolean, whether to reuse existing weights.
    
    Returns:
        A tensor of the same shape as inputs, but normalized over the last dimension.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + 1e-5) ** (.5))
        outputs = gamma * normalized + beta
        
    return outputs


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    """Applies a highway network transformation to the inputs.
    
    Highway networks allow for easier training of deep networks by introducing
    gating units that control information flow. They help mitigate the vanishing
    gradient problem.
    
    Args:
        inputs: A 3D tensor of shape [batch_size, time_steps, channels].
        num_units: An int or None. Number of units in the highway layer.
            If None, uses the input size.
        scope: Optional scope for variable_scope.
        reuse: Boolean, whether to reuse existing weights.
    
    Returns:
        A 3D tensor of the same shape as inputs after applying the highway
        transformation.
    """
    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense_1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                          bias_initializer=tf.constant_initializer(-1.0),
                          name="dense_2")
        outputs = H * T + inputs * (1. - T)
        
    return outputs


def conv1d(inputs,
          filters=None,
          size=1,
          rate=1,
          padding="SAME",
          dropout_rate=0,
          use_bias=True,
          activation_fn=None,
          training=True,
          scope="conv1d",
          reuse=None):
    """Applies 1D convolution to the input tensor with various options.
    
    A flexible 1D convolution implementation that supports:
    - Different padding modes
    - Dilation
    - Dropout
    - Activation functions
    - Bias terms
    
    Args:
        inputs: A 3D tensor of shape [batch_size, time_steps, channels].
        filters: An int. Number of output filters.
        size: An int. Filter size (width).
        rate: An int. Dilation rate for dilated convolution.
        padding: String. Either 'SAME', 'VALID', or 'CAUSAL' (case-insensitive).
        dropout_rate: Float between 0 and 1. Dropout probability.
        use_bias: Boolean. Whether to add a bias term.
        activation_fn: Activation function to apply (None for linear).
        training: Boolean. Whether in training mode (affects dropout).
        scope: Optional scope for variable_scope.
        reuse: Boolean, whether to reuse existing weights.
    
    Returns:
        A 3D tensor of shape [batch_size, time_steps, filters] after convolution.
    """
    with tf.variable_scope(scope, reuse=reuse):
        if padding.lower() == "causal":
            # Causal padding for sequence modeling
            pad_len = (size - 1) * rate
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "VALID"
            
        if filters is None:
            filters = inputs.get_shape().as_list()[-1]
            
        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                 "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
                 "kernel_initializer": tf.contrib.layers.xavier_initializer(),
                 "activation": activation_fn}
        
        outputs = tf.layers.conv1d(**params)
        
        if dropout_rate > 0:
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
            
    return outputs


def hc(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       dropout_rate=0,
       use_bias=True,
       activation_fn=None,
       training=True,
       scope="hc",
       reuse=None):
    """Applies a highway convolution block.
    
    Combines highway networks with convolution, allowing the network to learn
    which information should pass through and which should be transformed.
    
    Args:
        inputs: A 3D tensor of shape [batch_size, time_steps, channels].
        filters: An int. Number of output filters.
        size: An int. Filter size (width).
        rate: An int. Dilation rate for dilated convolution.
        padding: String. Either 'SAME', 'VALID', or 'CAUSAL' (case-insensitive).
        dropout_rate: Float between 0 and 1. Dropout probability.
        use_bias: Boolean. Whether to add a bias term.
        activation_fn: Activation function to apply (None for linear).
        training: Boolean. Whether in training mode (affects dropout).
        scope: Optional scope for variable_scope.
        reuse: Boolean, whether to reuse existing weights.
    
    Returns:
        A 3D tensor of shape [batch_size, time_steps, filters] after highway convolution.
    """
    with tf.variable_scope(scope, reuse=reuse):
        H = conv1d(inputs,
                  filters=filters,
                  size=size,
                  rate=rate,
                  padding=padding,
                  dropout_rate=dropout_rate,
                  use_bias=use_bias,
                  activation_fn=activation_fn,
                  training=training,
                  scope="conv1d_1")
        T = conv1d(inputs,
                  filters=filters,
                  size=size,
                  rate=rate,
                  padding=padding,
                  dropout_rate=dropout_rate,
                  use_bias=use_bias,
                  activation_fn=tf.nn.sigmoid,
                  training=training,
                  scope="conv1d_2")
        outputs = H * T + inputs * (1. - T)
        
    return outputs


def conv1d_transpose(inputs,
                    filters=None,
                    size=3,
                    stride=2,
                    padding='same',
                    dropout_rate=0,
                    use_bias=True,
                    activation=None,
                    training=True,
                    scope="conv1d_transpose",
                    reuse=None):
    """Applies transposed 1D convolution for upsampling.
    
    Implements the transposed convolution operation (also known as deconvolution)
    for upsampling sequences. Typically used in the SSRN network to increase
    temporal resolution.
    
    Args:
        inputs: A 3D tensor of shape [batch_size, time_steps, channels].
        filters: An int. Number of output filters.
        size: An int. Filter size (width).
        stride: An int. Stride for upsampling.
        padding: String. Either 'SAME' or 'VALID' (case-insensitive).
        dropout_rate: Float between 0 and 1. Dropout probability.
        use_bias: Boolean. Whether to add a bias term.
        activation: Activation function to apply (None for linear).
        training: Boolean. Whether in training mode (affects dropout).
        scope: Optional scope for variable_scope.
        reuse: Boolean, whether to reuse existing weights.
    
    Returns:
        A 3D tensor of shape [batch_size, time_steps*stride, filters] after
        transposed convolution.
    """
    with tf.variable_scope(scope, reuse=reuse):
        if filters is None:
            filters = inputs.get_shape().as_list()[-1]
            
        # Compute shapes
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        channels = inputs.get_shape().as_list()[-1]
        
        # Reshape for conv2d_transpose
        inputs_2d = tf.expand_dims(inputs, 2)
        filters_shape = [size, 1, filters, channels]
        strides = [1, stride, 1, 1]
        
        # Create and apply transposed convolution
        weights = tf.get_variable("weights",
                                filters_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
        
        outputs_shape = [batch_size, length * stride, 1, filters]
        outputs = tf.nn.conv2d_transpose(inputs_2d,
                                       weights,
                                       outputs_shape,
                                       strides,
                                       padding=padding.upper())
        
        # Reshape back to 3D
        outputs = tf.squeeze(outputs, [2])
        
        if use_bias:
            bias = tf.get_variable("bias", [filters])
            outputs = tf.nn.bias_add(outputs, bias)
            
        if activation is not None:
            outputs = activation(outputs)
            
        if dropout_rate > 0:
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
            
    return outputs
