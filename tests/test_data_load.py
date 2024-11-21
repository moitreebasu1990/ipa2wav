import pytest
from ipa2wav.data_processing.data_load import load_data, load_vocab, text_normalize


def test_load_data():
    # Mock or create a small dataset for testing
    data = load_data('../data/LJSpeech-1.1')
    assert data is not None
    # Add more assertions based on expected data format and content


def test_load_vocab():
    vocab = load_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) > 0


def test_text_normalize():
    text = "Hello, World!"
    normalized_text = text_normalize(text)
    assert isinstance(normalized_text, str)
    assert normalized_text == "hello world"
