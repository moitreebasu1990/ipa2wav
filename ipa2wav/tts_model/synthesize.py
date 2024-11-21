# -*- coding: utf-8 -*-
"""Speech synthesis module for IPA2WAV.

This module provides functionality for synthesizing speech from IPA text input.
It uses the trained Text2Mel and SSRN models to generate high-quality speech.
"""

import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from scipy.io.wavfile import write
from tqdm import tqdm

from ipa2wav.data_processing.data_load import load_data, load_vocab, text_normalize
from ipa2wav.tts_model.hyperparams import Hyperparams as hp
from ipa2wav.tts_model.networks import TextEncoder, AudioEncoder, AudioDecoder, Attention, SSRN
from ipa2wav.tts_model.train import Text2MelModel
from ipa2wav.utils.utils import *

# Initialize Flask application for server mode
app = Flask(__name__, static_url_path="/static")
app.config['JSON_AS_ASCII'] = False

# Global variables for models
global text2mel_model
global ssrn_model
global device

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="IPA2WAV synthesis script")

    parser.add_argument("-i", "--interactive",
                       dest="interactive", action="store_true",
                       help="Run model in interactive mode")

    parser.add_argument("-e", "--evaluate",
                       dest="evaluate", action="store_true",
                       help="Run model in evaluation mode")

    parser.add_argument("-s", "--serve",
                       dest="serve", action="store_true",
                       help="Run model in server mode")

    parser.add_argument("-H", "--host", default="0.0.0.0",
                       help="Host address for server mode")

    parser.add_argument("-p", "--port", default=5000,
                       help="Port number for server mode")

    return parser


def parse_cmd():
    """Parse command line arguments for different operation modes.
    
    Supports three modes of operation:
    1. Interactive mode: Take input from command line
    2. Evaluation mode: Process test dataset
    3. Server mode: Run as a web service
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.interactive or args.evaluate or args.serve):
        parser.error("Must specify one mode: --interactive, --evaluate, or --serve")
    
    return args


def custom_data_load(lines, char2idx, idx2char):
    """Load and process custom input text for synthesis.
    
    Args:
        lines (list): List of input text lines
        char2idx (dict): Character to index mapping
        idx2char (dict): Index to character mapping
    
    Returns:
        tuple: (texts, text_lengths)
            - texts: List of encoded text sequences
            - text_lengths: List of sequence lengths
    """
    texts = []
    text_lengths = []
    
    for text in lines:
        # Normalize text
        text = text_normalize(text)
        
        # Convert to sequence of indices
        sequence = [char2idx[char] for char in text]
        texts.append(sequence)
        text_lengths.append(len(sequence))
    
    # Pad sequences
    max_len = max(text_lengths)
    texts = [seq + [0] * (max_len - len(seq)) for seq in texts]
    
    return torch.LongTensor(texts), torch.LongTensor(text_lengths)


def load_models(checkpoint_dir=hp.checkpoint_dir):
    """Load trained Text2Mel and SSRN models.
    
    Args:
        checkpoint_dir (str): Directory containing model checkpoints
    
    Returns:
        tuple: (text2mel_model, ssrn_model)
    """
    global device
    
    # Load Text2Mel model
    text2mel_model = Text2MelModel().to(device)
    text2mel_checkpoint = torch.load(
        Path(checkpoint_dir) / "text2mel" / "latest.pt",
        map_location=device
    )
    text2mel_model.load_state_dict(text2mel_checkpoint['model_state_dict'])
    text2mel_model.eval()
    
    # Load SSRN model
    ssrn_model = SSRN().to(device)
    ssrn_checkpoint = torch.load(
        Path(checkpoint_dir) / "ssrn" / "latest.pt",
        map_location=device
    )
    ssrn_model.load_state_dict(ssrn_checkpoint['model_state_dict'])
    ssrn_model.eval()
    
    return text2mel_model, ssrn_model


def initialize_server():
    """Initialize server by loading models."""
    global text2mel_model, ssrn_model, device
    
    # Load models
    text2mel_model, ssrn_model = load_models()
    
    # Move models to device
    text2mel_model = text2mel_model.to(device)
    ssrn_model = ssrn_model.to(device)
    
    # Set models to eval mode
    text2mel_model.eval()
    ssrn_model.eval()


def synthesize(mode="evaluate", text=""):
    """Synthesize speech from input text or test dataset.
    
    Args:
        mode (str): Operation mode - "evaluate" or "synthesize"
        text (str): Input text for synthesis (used in synthesize mode)
    
    Returns:
        str: Path to the generated audio file
    """
    global text2mel_model, ssrn_model, device
    
    # Load vocabulary
    char2idx, idx2char = load_vocab()
    
    # Load or create input data
    if mode == "evaluate":
        texts, text_lengths = load_data(mode="synthesis")
    else:
        texts, text_lengths = custom_data_load([text], char2idx, idx2char)
    
    # Move data to device
    texts = texts.to(device)
    
    # Create output directory
    os.makedirs(hp.sampledir, exist_ok=True)
    
    # Generate audio for each input
    wav_paths = []
    with torch.no_grad():
        for i, text_sequence in enumerate(tqdm(texts)):
            # Add batch dimension
            text_sequence = text_sequence.unsqueeze(0)
            
            # Generate mel-spectrogram
            mel_outputs, _, _ = text2mel_model(text_sequence)
            
            # Generate linear spectrogram
            mag_outputs = ssrn_model(mel_outputs)
            
            # Convert to waveform
            wav = spectrogram2wav(mag_outputs[0].cpu().numpy())
            
            # Save waveform
            wav_path = os.path.join(hp.sampledir, f'sample_{i}.wav')
            write(wav_path, hp.sr, wav)
            wav_paths.append(wav_path)
    
    return wav_paths[0] if len(wav_paths) == 1 else wav_paths


@app.route('/synthesize', methods=['POST'])
def service():
    """Web service endpoint for speech synthesis.
    
    Args:
        text (str): Input text for synthesis
    
    Returns:
        dict: JSON response containing:
            - status (str): "success" or "error"
            - message (str): Status message
            - wav_path (str): Path to generated audio file
    """
    global text2mel_model, ssrn_model, device
    
    try:
        # Initialize server if models aren't loaded
        if text2mel_model is None or ssrn_model is None:
            initialize_server()
            
        text = request.json.get('text', '')
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'No text provided'
            })
        
        wav_path = synthesize(mode="synthesize", text=text)
        
        return jsonify({
            'status': 'success',
            'message': 'Speech synthesis completed',
            'wav_path': wav_path
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_cmd()
    
    if args.interactive:
        # Interactive mode
        while True:
            try:
                text = input("\nText to synthesize (Ctrl+C to exit): ")
                wav_path = synthesize(mode="synthesize", text=text)
                print(f"Generated audio saved to: {wav_path}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
    elif args.evaluate:
        # Evaluation mode
        wav_paths = synthesize(mode="evaluate")
        print(f"Generated {len(wav_paths)} audio files in {hp.sampledir}")
        
    else:
        # Server mode
        app.run(host=args.host, port=int(args.port))
