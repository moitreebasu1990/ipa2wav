"""Configuration for pytest."""

import os
import sys
import pytest
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def tf_session():
    """Create a TensorFlow session for testing.
    
    This session is shared across all tests in a session to avoid
    creating multiple sessions.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    yield sess
    sess.close()

@pytest.fixture(scope="session")
def sample_vocab():
    """Create a sample vocabulary for testing."""
    return "PE sxʃuɒhpjgm̃wŋaɛɪðnzʊbvlɑətirʒɜækʌθɔfId"

@pytest.fixture(scope="session")
def sample_text():
    """Create sample IPA text for testing."""
    return "həˈləʊ ˈwɜːld"  # "hello world" in IPA

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")
