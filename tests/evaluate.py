#!/usr/bin/env python3
"""
Evaluation script for IPA2WAV synthesis system.
Tests speech quality, synthesis performance, attention patterns, and model robustness.
"""

import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import psutil
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipa2wav.model import Text2Mel, SSRN
from ipa2wav.utils import load_config, preprocess_text

class Evaluator:
    def __init__(self, test_sets_path="test_data/test_sets.json"):
        self.test_sets_path = test_sets_path
        self.test_sets = self._load_test_sets()
        self.results = {
            "speech_quality": {},
            "synthesis_performance": {},
            "attention_metrics": {},
            "robustness": {}
        }
        
        # Load models
        config = load_config()
        self.text2mel = Text2Mel(config)
        self.ssrn = SSRN(config)
        
    def _load_test_sets(self):
        """Load test sets configuration."""
        with open(self.test_sets_path) as f:
            return json.load(f)
    
    def _calculate_mcd(self, ref_mel, syn_mel):
        """Calculate Mel Cepstral Distortion."""
        diff = ref_mel - syn_mel
        return np.mean(np.sqrt(np.sum(diff * diff, axis=1)))
    
    def _calculate_attention_metrics(self, attention_weights):
        """Calculate attention entropy and alignment quality."""
        entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-6))
        diagonality = np.sum(np.diag(attention_weights))
        return {"entropy": float(entropy), "diagonality": float(diagonality)}
    
    def evaluate_speech_quality(self, ref_path, syn_path):
        """Evaluate PESQ, STOI, and MCD metrics."""
        # Load audio files
        ref_audio, sr = sf.read(ref_path)
        syn_audio, sr = sf.read(syn_path)
        
        # Ensure same length
        min_len = min(len(ref_audio), len(syn_audio))
        ref_audio = ref_audio[:min_len]
        syn_audio = syn_audio[:min_len]
        
        # Calculate metrics
        pesq_score = pesq(sr, ref_audio, syn_audio, 'wb')
        stoi_score = stoi(ref_audio, syn_audio, sr, extended=False)
        
        # Calculate MCD
        ref_mel = librosa.feature.melspectrogram(y=ref_audio, sr=sr)
        syn_mel = librosa.feature.melspectrogram(y=syn_audio, sr=sr)
        mcd = self._calculate_mcd(ref_mel, syn_mel)
        
        return {
            "pesq": float(pesq_score),
            "stoi": float(stoi_score),
            "mcd": float(mcd)
        }
    
    def evaluate_synthesis_performance(self, text_input):
        """Evaluate synthesis speed and resource usage."""
        start_time = time.time()
        process = psutil.Process()
        start_mem = process.memory_info().rss
        
        # Run synthesis
        mel = self.text2mel.synthesize(text_input)
        audio = self.ssrn.synthesize(mel)
        
        end_time = time.time()
        end_mem = process.memory_info().rss
        
        duration = librosa.get_duration(y=audio)
        rtf = (end_time - start_time) / duration
        memory_used = (end_mem - start_mem) / 1024 / 1024  # MB
        
        return {
            "rtf": float(rtf),
            "memory_mb": float(memory_used),
            "process_time": float(end_time - start_time)
        }
    
    def plot_attention(self, attention_weights, save_path):
        """Plot and save attention alignment."""
        plt.figure(figsize=(10, 10))
        plt.imshow(attention_weights, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title('Attention Alignment')
        plt.xlabel('Decoder Steps')
        plt.ylabel('Encoder Steps')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def run_evaluation(self):
        """Run complete evaluation suite."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        for test_set in self.test_sets:
            print(f"Evaluating test set: {test_set['name']}")
            
            # Load test data
            with open(f"test_data/text/{test_set['text_file']}") as f:
                text = f.read().strip()
            ref_path = f"test_data/reference/{test_set['reference_file']}"
            
            # Synthesize speech
            perf_metrics = self.evaluate_synthesis_performance(text)
            self.results["synthesis_performance"][test_set["name"]] = perf_metrics
            
            # Generate and save audio
            syn_path = results_dir / f"syn_{test_set['name']}.wav"
            mel = self.text2mel.synthesize(text)
            audio = self.ssrn.synthesize(mel)
            sf.write(syn_path, audio, 22050)
            
            # Evaluate speech quality
            quality_metrics = self.evaluate_speech_quality(ref_path, syn_path)
            self.results["speech_quality"][test_set["name"]] = quality_metrics
            
            # Evaluate attention
            attention = self.text2mel.get_attention_weights()
            att_metrics = self._calculate_attention_metrics(attention)
            self.results["attention_metrics"][test_set["name"]] = att_metrics
            
            # Plot attention
            self.plot_attention(attention, 
                              results_dir / f"attention_{test_set['name']}.png")
        
        # Save results
        with open(results_dir / "evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

def main():
    evaluator = Evaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
