import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from stt import STTClient

@pytest.fixture
def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock_client:
        yield mock_client

@pytest.mark.asyncio
async def test_transcribe_success(mock_httpx_client):
    # Setup mock response
    mock_post = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "Hello world"}
    mock_post.return_value = mock_response
    
    mock_client_instance = AsyncMock()
    mock_client_instance.post = mock_post
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_httpx_client.return_value = mock_client_instance

    # Initialize client
    client = STTClient(api_key="test_key", model="scribe_v1", language_code="en")
    
    # Fake PCM data (16-bit, 48kHz, mono) - 1 second
    pcm_data = b'\x00\x00' * 48000
    
    # Run transcribe
    text = await client.transcribe(pcm_data, user="test_user")
    
    # Assertions
    assert text == "Hello world"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["headers"]["xi-api-key"] == "test_key"
    assert kwargs["data"]["model_id"] == "scribe_v1"
    assert kwargs["data"]["language_code"] == "en"
    assert "file" in kwargs["files"]

@pytest.mark.asyncio
async def test_transcribe_empty_response(mock_httpx_client):
    # Setup mock response with empty text
    mock_post = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": ""}
    mock_post.return_value = mock_response
    
    mock_client_instance = AsyncMock()
    mock_client_instance.post = mock_post
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_httpx_client.return_value = mock_client_instance

    client = STTClient(api_key="test_key")
    pcm_data = b'\x00\x00' * 48000
    
    text = await client.transcribe(pcm_data)
    assert text is None

@pytest.mark.asyncio
async def test_transcribe_api_error(mock_httpx_client):
    from httpx import HTTPStatusError, Request, Response
    
    # Setup mock with error
    mock_post = AsyncMock()
    mock_response = MagicMock()
    
    request = Request("POST", "https://api.elevenlabs.io/v1/speech-to-text")
    response = Response(401, request=request)
    mock_response.raise_for_status.side_effect = HTTPStatusError("401 Unauthorized", request=request, response=response)
    mock_post.return_value = mock_response
    
    mock_client_instance = AsyncMock()
    mock_client_instance.post = mock_post
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_httpx_client.return_value = mock_client_instance

    client = STTClient(api_key="test_key")
    pcm_data = b'\x00\x00' * 48000
    
    text = await client.transcribe(pcm_data)
    assert text is None
