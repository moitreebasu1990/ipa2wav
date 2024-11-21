# -*- coding: utf-8 -*-
"""Training module for IPA2WAV synthesis model."""

from __future__ import print_function

from tqdm import tqdm
from data_processing.data_load import get_batch, load_vocab
from .hyperparams import Hyperparams as hp
from .modules import *
from .networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils.utils import *
import sys


class Graph:
    """TensorFlow computation graph for the TTS model.
    
    This class builds the computation graph for either Text2Mel or SSRN network.
    It handles both training and synthesis modes, setting up appropriate
    placeholders, building the network architecture, and defining loss functions.
    
    Attributes:
        char2idx (dict): Mapping from characters to indices
        idx2char (dict): Mapping from indices to characters
        L (tf.Tensor): Input text sequences
        mels (tf.Tensor): Mel-spectrograms
        mags (tf.Tensor): Magnitude spectrograms
        fnames (tf.Tensor): File names for debugging
        num_batch (int): Number of batches in dataset
        prev_max_attentions (tf.Tensor): Previous attention peaks
        gts (tf.Tensor): Guided attention targets
    """
    
    def __init__(self, num=1, mode="train"):
        """Initialize the computation graph.
        
        Args:
            num (int): Network selector (1 for Text2Mel, 2 for SSRN)
            mode (str): Either "train" or "synthesize"
        """
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()
        
        # Set training flag
        training = (mode == "train")
        
        # Build graph
        if mode == "train":
            # Training mode: Get batched data
            self.L, self.mels, self.mags, self.fnames, self.num_batch = get_batch()
            self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            self.gts = tf.convert_to_tensor(guided_attention())
        else:
            # Synthesis mode: Create placeholders
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))
        
        # Build appropriate network
        if num == 1 or (not training):
            self._build_text2mel(training)
        else:
            self._build_ssrn(training)
    
    def _build_text2mel(self, training):
        """Build Text2Mel network.
        
        Constructs the Text2Mel network that converts text to mel-spectrograms:
        1. Text encoder
        2. Audio encoder
        3. Attention mechanism
        4. Audio decoder
        
        Args:
            training (bool): Whether in training mode
        """
        with tf.variable_scope("Text2Mel"):
            # Prepare decoder inputs
            self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), 
                              self.mels[:, :-1, :]), 1)
            
            # Build network components
            self.K, self.V = TextEnc(self.L, training=training)
            self.Q = AudioEnc(self.S, training=training)
            self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                    mononotic_attention=(not training),
                                                                    prev_max_attentions=self.prev_max_attentions)
            self.Y = AudioDec(self.R, training=training)
            
            # Define loss
            if training:
                self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))
                self.loss_att = tf.reduce_mean(tf.abs(self.alignments * self.gts))
                self.loss = self.loss_mels + self.loss_att
                
                # Optimizer
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lr = learning_rate_decay(self.global_step)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                
                # Gradient clipping
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                             global_step=self.global_step)
                
                # Summaries
                tf.summary.scalar('loss_mels', self.loss_mels)
                tf.summary.scalar('loss_att', self.loss_att)
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('learning_rate', self.lr)
                self.merged = tf.summary.merge_all()
    
    def _build_ssrn(self, training):
        """Build SSRN (Spectrogram Super-resolution Network).
        
        Constructs the SSRN network that converts mel-spectrograms to
        linear-scale spectrograms through upsampling and refinement.
        
        Args:
            training (bool): Whether in training mode
        """
        with tf.variable_scope("SSRN"):
            self.Z = SSRN(self.mels, training=training)
            
            if training:
                self.loss = tf.reduce_mean(tf.abs(self.Z - self.mags))
                
                # Optimizer
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lr = learning_rate_decay(self.global_step)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                
                # Gradient clipping
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                             global_step=self.global_step)
                
                # Summaries
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('learning_rate', self.lr)
                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # Get network number from command line (1: Text2Mel, 2: SSRN)
    num = int(sys.argv[1])
    
    # Build graph
    g = Graph(num=num); print("Graph loaded")
    
    # Start training
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        
        # Restore from checkpoint if exists
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Text2Mel" if num==1 else "SSRN")
        saver = tf.train.Saver(var_list=var_list)
        
        ckpt = tf.train.latest_checkpoint(hp.logdir + "-{}".format(num))
        if ckpt:
            saver.restore(sess, ckpt)
            print("Restored checkpoint:", ckpt)
        else:
            print("Starting new training session")
        
        # Set up summary writer
        writer = tf.summary.FileWriter(hp.logdir + "-{}".format(num), sess.graph)
        
        # Training loop
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        try:
            for _ in tqdm(range(hp.num_iterations)):
                _, gs, merged = sess.run([g.train_op, g.global_step, g.merged])
                writer.add_summary(merged, global_step=gs)
                
                # Save checkpoint
                if gs % 1000 == 0:
                    saver.save(sess, hp.logdir + "-{}/model_gs_{}".format(num, gs))
                    
        except KeyboardInterrupt:
            print()
        finally:
            coord.request_stop()
            coord.join(threads)
            
        print("Done")
