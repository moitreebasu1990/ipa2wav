import pytest
import argparse
from unittest.mock import patch, MagicMock
from ipa2wav.tts_model.synthesize import get_parser, parse_cmd, custom_data_load, load_models, initialize_server, synthesize, service


def test_get_parser():
    parser = get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parse_cmd():
    with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(interactive=True)):
        args = parse_cmd()
        assert args.interactive is True


def test_custom_data_load():
    lines = ['hello world']
    char2idx = {c: i for i, c in enumerate('helo wrd')}
    idx2char = {i: c for c, i in char2idx.items()}
    texts, text_lengths = custom_data_load(lines, char2idx, idx2char)
    assert len(texts) == 1
    assert text_lengths[0] == len(lines[0])


def test_load_models():
    with patch('ipa2wav.tts_model.synthesize.Text2MelModel') as MockText2MelModel:
        with patch('ipa2wav.tts_model.synthesize.SSRN') as MockSSRN:
            MockText2MelModel.return_value = MagicMock()
            MockSSRN.return_value = MagicMock()
            text2mel_model, ssrn_model = load_models()
            assert text2mel_model is not None
            assert ssrn_model is not None


def test_initialize_server():
    with patch('ipa2wav.tts_model.synthesize.load_models', return_value=(MagicMock(), MagicMock())):
        initialize_server()
        # Add assertions for server initialization


def test_synthesize():
    with patch('ipa2wav.tts_model.synthesize.synthesize') as mock_synthesize:
        mock_synthesize.return_value = 'path/to/audio.wav'
        result = synthesize(mode='synthesize', text='hello')
        assert result == 'path/to/audio.wav'


def test_service():
    with patch('ipa2wav.tts_model.synthesize.synthesize', return_value='path/to/audio.wav'):
        with patch('ipa2wav.tts_model.synthesize.request') as mock_request:
            mock_request.json = {'text': 'hello'}
            response = service()
            assert response['status'] == 'success'
            assert 'path/to/audio.wav' in response['wav_path']
