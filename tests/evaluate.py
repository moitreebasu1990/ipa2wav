#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation script for IPA2WAV synthesis system.

This script provides comprehensive evaluation of the IPA2WAV model:
1. Speech Quality Metrics:
   - PESQ (Perceptual Evaluation of Speech Quality)
   - STOI (Short-Time Objective Intelligibility)
   - Mel Cepstral Distortion (MCD)
2. Synthesis Performance:
   - Real-time factor
   - Memory usage
3. Attention Analysis:
   - Alignment visualization
   - Attention entropy
4. Model Robustness:
   - Various input lengths
   - Different IPA patterns
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
import psutil
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tts_model.hyperparams import Hyperparams as hp
from tts_model.train import Graph
from tts_model.utils import *
from data_processing.text_to_phone_IPA import process_text


class Evaluator:
    """Evaluator class for IPA2WAV model assessment."""
    
    def __init__(self, checkpoint_path=None):
        """Initialize evaluator.
        
        Args:
            checkpoint_path (str, optional): Path to model checkpoint.
                If None, uses latest checkpoint from hp.logdir.
        """
        self.process = psutil.Process(os.getpid())
        self.results = {
            'speech_quality': {},
            'synthesis_performance': {},
            'attention_metrics': {},
            'robustness': {}
        }
        
        # Initialize model
        print("Initializing model...")
        self.graph = Graph(mode="synthesize")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        # Restore checkpoint
        if checkpoint_path is None:
            checkpoint_path = tf.train.latest_checkpoint(hp.logdir + "-1")
        
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(self.sess, checkpoint_path)
        print("Text2Mel model restored")
        
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(self.sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN model restored")
    
    def evaluate_speech_quality(self, test_files, reference_dir):
        """Evaluate speech quality using objective metrics.
        
        Args:
            test_files (list): List of test file paths
            reference_dir (str): Directory containing reference audio files
        """
        print("\nEvaluating speech quality...")
        pesq_scores = []
        stoi_scores = []
        mcd_scores = []
        
        for test_file in tqdm(test_files):
            # Generate speech
            text = open(test_file, 'r').read().strip()
            wav = self.synthesize(text)
            
            # Load reference audio
            ref_file = os.path.join(reference_dir, os.path.basename(test_file).replace('.txt', '.wav'))
            ref_wav, _ = librosa.load(ref_file, sr=hp.sr)
            
            # Compute metrics
            try:
                pesq_score = pesq(hp.sr, ref_wav, wav, 'wb')
                pesq_scores.append(pesq_score)
            except:
                print(f"PESQ computation failed for {test_file}")
            
            stoi_score = stoi(ref_wav, wav, hp.sr, extended=False)
            stoi_scores.append(stoi_score)
            
            # Compute MCD
            mcd = self._compute_mcd(ref_wav, wav)
            mcd_scores.append(mcd)
        
        self.results['speech_quality'] = {
            'pesq_mean': np.mean(pesq_scores),
            'pesq_std': np.std(pesq_scores),
            'stoi_mean': np.mean(stoi_scores),
            'stoi_std': np.std(stoi_scores),
            'mcd_mean': np.mean(mcd_scores),
            'mcd_std': np.std(mcd_scores)
        }
    
    def evaluate_synthesis_performance(self, test_texts):
        """Evaluate synthesis speed and resource usage.
        
        Args:
            test_texts (list): List of test texts
        """
        print("\nEvaluating synthesis performance...")
        rtf_values = []  # Real-time factor
        memory_usage = []
        
        for text in tqdm(test_texts):
            # Measure synthesis time
            start_time = time.time()
            _ = self.synthesize(text)
            end_time = time.time()
            
            # Calculate real-time factor
            audio_length = len(_) / hp.sr
            rtf = (end_time - start_time) / audio_length
            rtf_values.append(rtf)
            
            # Measure memory usage
            memory_usage.append(self.process.memory_info().rss / 1024 / 1024)  # MB
        
        self.results['synthesis_performance'] = {
            'rtf_mean': np.mean(rtf_values),
            'rtf_std': np.std(rtf_values),
            'memory_mean': np.mean(memory_usage),
            'memory_std': np.std(memory_usage)
        }
    
    def evaluate_attention(self, test_texts, output_dir):
        """Analyze attention alignment quality.
        
        Args:
            test_texts (list): List of test texts
            output_dir (str): Directory to save attention plots
        """
        print("\nEvaluating attention mechanism...")
        attention_entropy = []
        
        for i, text in enumerate(tqdm(test_texts)):
            # Get attention weights
            mels, alignments = self.synthesize(text, return_attention=True)
            
            # Compute attention entropy
            entropy_vals = [entropy(alignment) for alignment in alignments[0]]
            attention_entropy.append(np.mean(entropy_vals))
            
            # Plot and save alignment
            plt.figure(figsize=(12, 8))
            plt.imshow(alignments[0].T, aspect='auto')
            plt.xlabel('Encoder Steps')
            plt.ylabel('Decoder Steps')
            plt.title(f'Attention Alignment - Sample {i}')
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, f'attention_{i}.png'))
            plt.close()
        
        self.results['attention_metrics'] = {
            'entropy_mean': float(np.mean(attention_entropy)),
            'entropy_std': float(np.std(attention_entropy))
        }
    
    def evaluate_robustness(self, test_sets):
        """Evaluate model robustness to different inputs.
        
        Args:
            test_sets (dict): Dictionary of test sets with categories
                Example: {
                    'short': ['short text 1', 'short text 2'],
                    'long': ['long text 1', ...],
                    'complex': ['complex IPA pattern 1', ...]
                }
        """
        print("\nEvaluating model robustness...")
        robustness_results = {}
        
        for category, texts in test_sets.items():
            success_rate = 0
            rtf_values = []
            
            for text in tqdm(texts, desc=f"Testing {category}"):
                try:
                    start_time = time.time()
                    _ = self.synthesize(text)
                    end_time = time.time()
                    
                    rtf_values.append(end_time - start_time)
                    success_rate += 1
                except Exception as e:
                    print(f"Failed on {category} text: {text}")
                    print(f"Error: {str(e)}")
            
            robustness_results[category] = {
                'success_rate': success_rate / len(texts),
                'rtf_mean': np.mean(rtf_values) if rtf_values else 0,
                'rtf_std': np.std(rtf_values) if rtf_values else 0
            }
        
        self.results['robustness'] = robustness_results
    
    def synthesize(self, text, return_attention=False):
        """Synthesize speech from text.
        
        Args:
            text (str): Input text in IPA format
            return_attention (bool): Whether to return attention weights
        
        Returns:
            np.array: Generated waveform
            np.array (optional): Attention weights if return_attention=True
        """
        # Convert text to IPA if needed
        if not all(c in hp.vocab for c in text):
            text = process_text(text)
        
        # Prepare input
        L = np.array([hp.char2idx[c] for c in text], dtype=np.int32)
        L = np.expand_dims(L, 0)
        
        # Generate mel-spectrogram
        feed_dict = {
            self.graph.L: L,
            self.graph.mels: np.zeros([1, hp.max_T, hp.n_mels]),
            self.graph.prev_max_attentions: np.zeros(1)
        }
        
        if return_attention:
            mels, alignments = self.sess.run([self.graph.Y, self.graph.alignments], feed_dict)
            return mels, alignments
        else:
            mels = self.sess.run(self.graph.Y, feed_dict)
        
        # Generate waveform
        Z = self.sess.run(self.graph.Z, {self.graph.Y: mels})
        wav = spectrogram2wav(Z[0])
        return wav
    
    def _compute_mcd(self, ref_wav, syn_wav):
        """Compute Mel Cepstral Distortion between reference and synthesized speech.
        
        Args:
            ref_wav (np.array): Reference waveform
            syn_wav (np.array): Synthesized waveform
        
        Returns:
            float: MCD value
        """
        # Extract MFCCs
        ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=hp.sr, n_mfcc=13)
        syn_mfcc = librosa.feature.mfcc(y=syn_wav, sr=hp.sr, n_mfcc=13)
        
        # Dynamic Time Warping
        _, wp = librosa.sequence.dtw(ref_mfcc, syn_mfcc)
        
        # Compute MCD
        diff = ref_mfcc[:, wp[:, 0]] - syn_mfcc[:, wp[:, 1]]
        mcd = np.mean(np.sqrt(2 * np.sum(diff**2, axis=0)))
        return mcd
    
    def save_results(self, output_file):
        """Save evaluation results to JSON file.
        
        Args:
            output_file (str): Path to output JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    """Main evaluation function."""
    # Setup paths
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(eval_dir, 'test_data')
    output_dir = os.path.join(eval_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Load test data
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                 if f.endswith('.txt')]
    
    # Load test sets for robustness evaluation
    with open(os.path.join(test_dir, 'test_sets.json'), 'r') as f:
        test_sets = json.load(f)
    
    # Run evaluations
    evaluator.evaluate_speech_quality(test_files, os.path.join(test_dir, 'reference'))
    evaluator.evaluate_synthesis_performance([open(f).read().strip() 
                                           for f in test_files])
    evaluator.evaluate_attention([open(f).read().strip() 
                                for f in test_files], output_dir)
    evaluator.evaluate_robustness(test_sets)
    
    # Save results
    evaluator.save_results(os.path.join(output_dir, 'evaluation_results.json'))


if __name__ == '__main__':
    main()
