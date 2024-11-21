# IPA2WAV Synthesis Framework

> **⚠️ DISCLAIMER:** This implementation uses TensorFlow 1.8 and is now outdated. For an updated PyTorch implementation, please check the `pytorch_implementation` branch.

A deep learning-based text-to-speech synthesis system that converts International Phonetic Alphabet (IPA) symbols to speech. This implementation is inspired by the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969), but extends it to work with IPA symbols instead of raw text.

## Project Overview

This project implements a novel approach to speech synthesis by:
- Using IPA symbols as input instead of raw text
- Implementing a two-stage synthesis pipeline:
  1. Text2Mel: Converts IPA symbols to mel-spectrograms
  2. SSRN (Spectrogram Super-resolution Network): Converts mel-spectrograms to full spectrograms
- Incorporating guided attention mechanism for improved alignment
- Supporting multiple datasets and languages through IPA representation

## Project Structure

```
thesis_work/
├── data/                      # Data directory
│   ├── raw/                  # Original dataset files
│   ├── processed/            # Preprocessed data
│   ├── external/             # External data sources
│   ├── LJSpeech-1.1/        # LJSpeech dataset
│   ├── cmu_us_arctic/       # CMU Arctic dataset
│   └── text_to_phone_IPA.py # IPA conversion script
├── src/                      # Source code
│   ├── data_processing/     # Data processing scripts
│   ├── features/            # Feature extraction
│   ├── tts_model/          # TTS model implementation
│   ├── evaluation/         # Model evaluation scripts
│   ├── visualization/      # Visualization tools
│   └── utils/              # Utility functions
├── notebooks/               # Jupyter notebooks for analysis
├── results/                 # Output directory
│   ├── figures/             # Generated plots
│   ├── models/              # Saved model checkpoints
│   └── logs/                # Training logs
├── docs/                    # Documentation
│   ├── papers/             # Related research papers
│   └── thesis/             # Thesis documents
└── tests/                  # Unit tests
```

## Installation

IPA2WAV can be installed in several ways depending on your needs:

### 1. Development Installation (Recommended for Contributors)

Clone the repository and install in development mode with all dependencies:

```bash
# Clone the repository
git clone https://github.com/moitreebasu1990/ipa2wav.git
cd ipa2wav

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test]"  # Install with both dev and test dependencies
# OR
pip install -e ".[dev]"       # Install with only dev dependencies
# OR
pip install -e ".[test]"      # Install with only test dependencies
```

### 2. Regular Installation

Install the latest stable version with core dependencies only:

```bash
pip install git+https://github.com/moitreebasu1990/ipa2wav.git
```

### 3. Manual Installation

Install from source with specific dependency sets:

```bash
# Clone the repository
git clone https://github.com/moitreebasu1990/ipa2wav.git
cd ipa2wav

# Install core package
pip install .                 # Install core package only
# OR
pip install ".[dev]"         # Install with development tools
# OR
pip install ".[test]"        # Install with test dependencies
# OR
pip install ".[dev,test]"    # Install with all dependencies
```

### Available Dependency Sets

1. Core Dependencies (installed by default):
   - numpy>=1.19.0
   - tensorflow>=1.8.0
   - librosa>=0.8.0
   - tqdm>=4.50.0
   - matplotlib>=3.3.0
   - scipy>=1.5.0
   - soundfile>=0.10.0
   - tensorboard>=2.4.0

2. Development Dependencies (`[dev]`):
   - black (code formatting)
   - flake8 (code linting)
   - isort (import sorting)
   - sphinx (documentation)
   - sphinx-rtd-theme (documentation theme)

3. Test Dependencies (`[test]`):
   - pytest>=6.0.0
   - pytest-cov (test coverage)
   - pesq (speech quality metric)
   - pystoi (speech intelligibility)
   - psutil (system monitoring)

### Command-Line Tools

After installation, the following commands become available:

```bash
# Preprocess data
ipa2wav-preprocess --dataset [dataset_name]

# Train the model
ipa2wav-train --stage 1  # Train Text2Mel network
ipa2wav-train --stage 2  # Train SSRN network

# Synthesize speech
ipa2wav-synthesize --text "[IPA_TEXT]" --out_file "output.wav"
```

### Verifying Installation

To verify your installation:

```bash
# Check if package is installed
pip show ipa2wav

# Try importing the package
python -c "import ipa2wav; print(ipa2wav.__version__)"

# Run tests
pytest tests/
```

## Dataset Preparation

### Supported Datasets
1. [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) (Primary test dataset)
2. [CMU Arctic Data](http://festvox.org/cmu_arctic/packed/)
3. [Pavoque Data](https://github.com/marytts/pavoque-data)

### Data Preprocessing
1. Download the desired dataset to `data/raw/`
2. Run preprocessing script:
```bash
python src/data_processing/prepro.py --dataset [dataset_name]
```
3. Processed data will be saved in `data/processed/`

## Training

The training process is split into two parallel stages:

### Stage 1: Text2Mel Training
```bash
python src/tts_model/train.py --stage 1
```

### Stage 2: SSRN Training
```bash
python src/tts_model/train.py --stage 2
```

Training progress and model checkpoints will be saved in `results/`.

### Training Configuration
All hyperparameters are defined in `src/tts_model/hyperparams.py`:
- Batch size: 32 (default)
- Learning rate: 0.001 with decay
- Sampling rate: 22050 Hz
- Mel bands: 80
- Embedding dimension: 128
- Hidden units (Text2Mel): 256
- Hidden units (SSRN): 512

## Model Architecture

### Key Components
1. Text2Mel Network:
   - Encoder-decoder architecture
   - Guided attention mechanism
   - Layer normalization
   - Dropout for regularization

2. SSRN (Spectrogram Super-resolution Network):
   - Convolutional architecture
   - Upsampling layers
   - Residual connections

## Evaluation

The project includes comprehensive evaluation tools to assess various aspects of the model's performance:

### Speech Quality Metrics
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- MCD (Mel Cepstral Distortion)

### Synthesis Performance
- Real-time factor (RTF)
- Memory usage tracking
- Processing time analysis

### Attention Analysis
- Attention alignment visualization
- Attention entropy measurement
- Alignment quality assessment

### Model Robustness
- Testing with various input lengths
- Complex IPA pattern handling
- Stress pattern evaluation
- Number and punctuation handling

### Running Evaluation

1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

2. Prepare test data:
   - Place test text files in `tests/test_data/`
   - Place reference audio files in `tests/test_data/reference/`
   - Configure test sets in `tests/test_data/test_sets.json`

3. Run evaluation:
```bash
python tests/evaluate.py
```

4. View results:
   - Results are saved in `results/evaluation_results.json`
   - Attention plots are saved in `results/attention_*.png`

### Interpreting Results

#### Speech Quality
- PESQ: Range 1-5, higher is better
- STOI: Range 0-1, higher is better
- MCD: Lower is better (typical range: 4-8)

#### Synthesis Performance
- RTF < 1: Faster than real-time
- RTF > 1: Slower than real-time

#### Attention Metrics
- Lower entropy indicates more focused attention
- Diagonal alignment patterns are ideal

#### Robustness
- Success rate should be close to 1.0
- Consistent RTF across categories is desirable

## Inference

To synthesize speech from IPA text:
```bash
python src/tts_model/synthesize.py --text "[IPA_TEXT]" --out_file "output.wav"
```

## Results

Training logs and visualizations can be found in `results/`:
- Model checkpoints: `results/models/`
- Training logs: `results/logs/`
- Generated figures: `results/figures/`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the authors of the original DC-TTS paper
- The LJ Speech Dataset team
- Contributors to the TensorFlow and librosa libraries
