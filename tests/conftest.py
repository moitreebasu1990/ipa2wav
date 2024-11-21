"""Test configuration and shared fixtures."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from flask import Flask

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipa2wav.tts_model.synthesize import app as flask_app
from ipa2wav.tts_model.hyperparams import Hyperparams as hp


@pytest.fixture
def app():
    """Create Flask application for testing."""
    # Configure app for testing
    flask_app.config.update({
        'TESTING': True,
        'JSON_AS_ASCII': False
    })
    
    # Create temporary directory for audio outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        hp.sampledir = tmp_dir
        yield flask_app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create Flask CLI test runner."""
    return app.test_cli_runner()


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")
