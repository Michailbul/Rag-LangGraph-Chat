import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from langchain.schema import AIMessage

class MockSessionState(dict):
    def __getattr__(self, name):
        if name not in self:
            self[name] = None
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

@pytest.fixture(scope="module")
def main_module():
    with patch('main.app'), \
         patch('main.OpenAI'), \
         patch('main.SimpleWebPageReader'), \
         patch('main.SummaryIndex'):
        import main
        yield main

@pytest.fixture
def mock_streamlit():
    session_state = MockSessionState()
    mock_sidebar = MagicMock()
    mock_sidebar.__enter__ = MagicMock()
    mock_sidebar.__exit__ = MagicMock()
    with patch('streamlit.text_area') as mock_text_area, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.chat_input') as mock_chat_input, \
         patch('streamlit.session_state', session_state), \
         patch('streamlit.sidebar', return_value=mock_sidebar) as mock_sidebar_func, \
         patch('streamlit.expander') as mock_expander, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.title') as mock_title, \
         patch('streamlit.subheader') as mock_subheader:
        yield {
            'text_area': mock_text_area,
            'button': mock_button,
            'chat_input': mock_chat_input,
            'session_state': session_state,
            'sidebar': mock_sidebar,
            'expander': mock_expander,
            'warning': mock_warning,
            'write': mock_write,
            'title': mock_title,
            'subheader': mock_subheader
        }

def test_url_input_processing_single_url(mock_streamlit, main_module):
    mock_streamlit['text_area'].return_value = "https://example.com"
    mock_streamlit['button'].return_value = True
    mock_streamlit['chat_input'].return_value = None  # No user input
    
    main_module.main()
    assert mock_streamlit['session_state'].urls == ["https://example.com"]

def test_url_input_processing_multiple_urls(mock_streamlit, main_module):
    mock_streamlit['text_area'].return_value = "https://example.com\nhttps://test.com"
    mock_streamlit['button'].return_value = True
    mock_streamlit['chat_input'].return_value = None  # No user input
    
    main_module.main()
    assert mock_streamlit['session_state'].urls == ["https://example.com", "https://test.com"]

def test_url_input_processing_empty(mock_streamlit, main_module):
    mock_streamlit['text_area'].return_value = ""
    mock_streamlit['button'].return_value = True
    mock_streamlit['chat_input'].return_value = None  # No user input
    
    main_module.main()
    mock_streamlit['warning'].assert_called_once_with("Error, no links")

def test_url_input_processing_whitespace(mock_streamlit, main_module):
    mock_streamlit['text_area'].return_value = "   \n  \n  "
    mock_streamlit['button'].return_value = True
    mock_streamlit['chat_input'].return_value = None  # No user input
    
    main_module.main()
    mock_streamlit['warning'].assert_called_once_with("Error, no links")

def test_url_input_processing_invalid_url(mock_streamlit, main_module):
    mock_streamlit['text_area'].return_value = "not_a_valid_url"
    mock_streamlit['button'].return_value = True
    mock_streamlit['chat_input'].return_value = None  # No user input
    
    main_module.main()
    assert mock_streamlit['session_state'].urls == ["not_a_valid_url"]

