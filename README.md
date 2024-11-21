# IPA2WAV Synthesis Framework

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
ipa2wav/
├── data/                    # Data directory containing datasets and processing scripts
├── docs/                    # Documentation
├── ipa2wav/                # Main package directory
├── models/                 # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for analysis
├── samples_IPA_22050/     # Generated speech samples
├── tests/                 # Unit tests
├── poetry.lock           # Poetry lock file with exact dependency versions
└── pyproject.toml        # Poetry project configuration and dependencies
```

## Installation

IPA2WAV uses Poetry for dependency management. Here's how to get started:

### 1. Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/docs/#installation)

### 2. Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/ipa2wav.git
cd ipa2wav

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Available Dependencies

All dependencies are managed through `pyproject.toml`. The main dependencies include:
- Core ML libraries (TensorFlow, etc.)
- Audio processing (librosa, soundfile)
- Development tools (pytest, black, etc.)

### Verifying Installation

To verify your installation:

```bash
# Activate poetry environment if not already active
poetry shell

# Run tests
poetry run pytest
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
poetry run python src/data_processing/prepro.py --dataset [dataset_name]
```
3. Processed data will be saved in `data/processed/`

## Training

The training process is split into two parallel stages:

### Stage 1: Text2Mel Training
```bash
poetry run python src/tts_model/train.py --stage 1
```

### Stage 2: SSRN Training
```bash
poetry run python src/tts_model/train.py --stage 2
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

1. All dependencies are already included in Poetry:
```bash
# Make sure you're in the Poetry shell
poetry shell
```

2. Prepare test data:
   - Place test text files in `tests/test_data/`
   - Place reference audio files in `tests/test_data/reference/`
   - Configure test sets in `tests/test_data/test_sets.json`

3. Run evaluation:
```bash
poetry run python tests/evaluate.py
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
poetry run python -m ipa2wav.synthesize --text "[IPA_TEXT]" --out_file "output.wav"
```

## Results

Training logs and visualizations can be found in `results/`:
- Model checkpoints: `results/models/`
- Training logs: `results/logs/`
- Generated figures: `results/figures/`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Set up development environment:
```bash
# Install dependencies including development ones
poetry install
# Activate virtual environment
poetry shell
# Run tests to ensure everything works
poetry run pytest
```
4. Commit your changes
5. Push to the branch
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the authors of the original DC-TTS paper
- The LJ Speech Dataset team
- Contributors to the TensorFlow and librosa libraries
