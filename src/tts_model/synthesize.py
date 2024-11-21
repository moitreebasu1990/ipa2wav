# -*- coding: utf-8 -*-
"""Speech synthesis module for IPA2WAV.

This module provides functionality for synthesizing speech from IPA text input.
It uses the trained Text2Mel and SSRN models to generate high-quality speech.
"""

from __future__ import print_function

from .hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from .train import Graph
from utils.utils import *
from data_processing.data_load import load_data, text_normalize, load_vocab
from scipy.io.wavfile import write
from tqdm import tqdm
import codecs
import re
from flask import Flask
from flask import jsonify
from flask import request

# Initialize Flask application for server mode
app = Flask(__name__, static_url_path="/static")
app.config['JSON_AS_ASCII'] = False
global g
global sess


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
                       
    parser.add_argument("-t", "--text",
                       help="Input text for synthesis")

    return parser


def parse_cmd():
    """Parse command line arguments for different operation modes.
    
    Supports three modes of operation:
    1. Interactive mode: Take input from command line
    2. Evaluation mode: Process test dataset
    3. Server mode: Run as a web service
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - interactive (bool): Whether to run in interactive mode
            - evaluate (bool): Whether to run in evaluation mode
            - serve (bool): Whether to run in server mode
            - host (str): Host address for server mode
            - port (int): Port number for server mode
            - text (str): Input text for synthesis
    """
    parser = get_parser()
    return parser.parse_args()


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
    texts = [text_normalize(line.strip()) + "E" for line in lines]
    texts = [np.array([char2idx[char] for char in text], np.int32) for text in texts]
    text_lengths = [len(text) for text in texts]
    return texts, text_lengths


def synthesize(mode="evaluate", text=""):
    """Synthesize speech from input text or test dataset.
    
    Args:
        mode (str): Operation mode - "evaluate" or "synthesize"
        text (str): Input text for synthesis (used in synthesize mode)
    
    Returns:
        str: Path to the generated audio file
    """
    # Load graph
    g = Graph(mode=mode)
    print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from checkpoint
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored")

        if mode == "synthesize":
            # Process input text
            char2idx, idx2char = load_vocab()
            texts = [text_normalize(text) + "E"]
            texts = [np.array([char2idx[char] for char in text], np.int32) for text in texts]
            
            # Generate speech
            wav_path = generate_wav(sess, g, texts[0])
            return wav_path
        else:
            # Process test dataset
            texts, text_lengths = load_data(mode="synthesize")
            for i, text in enumerate(texts):
                wav_path = generate_wav(sess, g, text, index=i)
            return wav_path


def service(text):
    """Web service endpoint for speech synthesis.
    
    Args:
        text (str): Input text for synthesis
    
    Returns:
        dict: JSON response containing:
            - status (str): "success" or "error"
            - message (str): Status message
            - wav_path (str): Path to generated audio file
    """
    try:
        wav_path = synthesize(mode="synthesize", text=text)
        return jsonify({
            "status": "success",
            "message": "Speech synthesized successfully",
            "wav_path": wav_path
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


def generate_wav(sess, g, L, path=hp.sampledir, index=0):
    """Generate waveform from encoded text sequence.
    
    The generation process involves:
    1. Generate mel-spectrogram using Text2Mel network
    2. Generate linear spectrogram using SSRN network
    3. Convert spectrogram to waveform using Griffin-Lim algorithm
    
    Args:
        sess (tf.Session): TensorFlow session
        g (Graph): Model graph
        L (np.array): Encoded text sequence
        path (str): Output directory for generated audio
        index (int): Index for output filename
    
    Returns:
        str: Path to the generated audio file
    """
    # Pad input sequence
    if L.shape[0] % hp.max_N != 0:
        padding = hp.max_N - (L.shape[0] % hp.max_N)
        L = np.pad(L, [[0, padding]], 'constant', constant_values=0)

    # Feed forward
    Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
    prev_max_attentions = np.zeros((len(L),), np.int32)

    for j in tqdm(range(hp.max_T)):
        _gs, _Y, _max_attentions, _alignments = \
            sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                    {g.L: np.expand_dims(L, 0),
                     g.mels: Y,
                     g.prev_max_attentions: prev_max_attentions})
        Y = _Y[0]
        prev_max_attentions = _max_attentions[0]

    # Generate wav file
    Z = sess.run(g.Z, {g.Y: np.expand_dims(Y, 0)})
    Z = Z[0]

    # Generate wav file
    if not os.path.exists(path): os.makedirs(path)
    wav = spectrogram2wav(Z)
    wav_path = os.path.join(path, 'sample_{}.wav'.format(index))
    write(wav_path, hp.sr, wav)
    return wav_path


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_cmd()
    
    if args.interactive:
        # Interactive mode
        while True:
            try:
                text = input("Enter text to synthesize (Ctrl+C to exit): ")
                wav_path = synthesize(mode="synthesize", text=text)
                print(f"Generated audio saved to: {wav_path}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
    elif args.evaluate:
        # Evaluation mode
        synthesize(mode="evaluate")
        
    elif args.serve:
        # Server mode
        g = Graph(mode="synthesize")
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        # Restore models
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        
        # Start server
        app.run(host=args.host, port=args.port)
        
    else:
        # Direct synthesis from command line argument
        if args.text:
            wav_path = synthesize(mode="synthesize", text=args.text)
            print(f"Generated audio saved to: {wav_path}")
        else:
            print("Please provide text using -t option or use interactive mode with -i")
