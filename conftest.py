import pytest
from unittest.mock import patch
import os

@pytest.fixture(autouse=True, scope="session")
def mock_env_vars():
    mock_env = {
        "OPENAI_API_KEY": "dummy_api_key",
        "LLAMA_PARSE": "dummy_llama_parse_key",
        "LANGCHAIN_API_KEY": "dummy_langchain_key",
        "LLAMA_CLOUD_API_KEY": "dummy_llama_cloud_key"
    }
    with patch.dict(os.environ, mock_env, clear=True):
        yield