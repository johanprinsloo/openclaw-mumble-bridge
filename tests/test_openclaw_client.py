"""Tests for the OpenClaw Gateway client.

Tests the SSE streaming client that communicates with OpenClaw's
OpenAI-compatible Chat Completions API. Uses mocked HTTP responses.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from openclaw_client import OpenClawClient


class MockAsyncIterator:
    """Mock async iterator for SSE lines."""
    
    def __init__(self, lines):
        self.lines = iter(lines)
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        try:
            return next(self.lines)
        except StopIteration:
            raise StopAsyncIteration


class TestOpenClawClientInit:
    """Test client initialization."""

    def test_default_values(self):
        client = OpenClawClient()
        
        assert client.gateway_url == "http://127.0.0.1:18789"
        assert client.agent_id == "main"
        assert client.session_user == "mumble-room"
        assert client.timeout_seconds == 60

    def test_custom_values(self):
        client = OpenClawClient(
            gateway_url="http://example.com:8080",
            gateway_token="secret-token",
            agent_id="custom-agent",
            session_user="custom-session",
            timeout_seconds=120,
        )
        
        assert client.gateway_url == "http://example.com:8080"
        assert client.gateway_token == "secret-token"
        assert client.agent_id == "custom-agent"
        assert client.session_user == "custom-session"
        assert client.timeout_seconds == 120


class TestSendStreaming:
    """Test the streaming send method."""

    @pytest.mark.asyncio
    async def test_yields_tokens_from_sse(self):
        """Test that tokens are correctly extracted from SSE stream."""
        client = OpenClawClient(gateway_token="test-token")
        
        # Mock SSE response lines
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            tokens = []
            async for token in client.send_streaming("Hello", speaker="alice"):
                tokens.append(token)
        
        assert tokens == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_includes_speaker_in_message(self):
        """Test that speaker name is included in the message content."""
        client = OpenClawClient(gateway_token="test-token")
        
        sse_lines = ['data: [DONE]']
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        captured_payload = None
        
        def capture_stream(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return mock_response
        
        mock_client.stream = capture_stream
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            async for _ in client.send_streaming("What's the weather?", speaker="bob"):
                pass
        
        assert captured_payload is not None
        assert captured_payload["messages"][0]["content"] == "[bob]: What's the weather?"

    @pytest.mark.asyncio
    async def test_stops_on_cancellation(self):
        """Test that streaming stops when cancellation event is set."""
        client = OpenClawClient(gateway_token="test-token")
        
        # Many tokens, but we'll cancel after first
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Token1"}}]}',
            'data: {"choices":[{"delta":{"content":"Token2"}}]}',
            'data: {"choices":[{"delta":{"content":"Token3"}}]}',
            'data: [DONE]',
        ]
        
        cancel_event = asyncio.Event()
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            tokens = []
            async for token in client.send_streaming(
                "Hello", speaker="alice", cancellation_event=cancel_event
            ):
                tokens.append(token)
                cancel_event.set()  # Cancel after first token
        
        # Should have stopped after first token
        assert len(tokens) == 1
        assert tokens[0] == "Token1"

    @pytest.mark.asyncio
    async def test_handles_malformed_sse_data(self):
        """Test that malformed SSE data is handled gracefully."""
        client = OpenClawClient(gateway_token="test-token")
        
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Good"}}]}',
            'data: {not valid json}',  # Malformed
            'data: {"choices":[{"delta":{"content":"Data"}}]}',
            'data: [DONE]',
        ]
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            tokens = []
            async for token in client.send_streaming("Hello", speaker="alice"):
                tokens.append(token)
        
        # Should skip malformed line but continue
        assert tokens == ["Good", "Data"]

    @pytest.mark.asyncio
    async def test_ignores_non_data_lines(self):
        """Test that non-data SSE lines are ignored."""
        client = OpenClawClient(gateway_token="test-token")
        
        sse_lines = [
            ': comment line',
            '',  # Empty line
            'event: ping',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: [DONE]',
        ]
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            tokens = []
            async for token in client.send_streaming("Hello", speaker="alice"):
                tokens.append(token)
        
        assert tokens == ["Hello"]

    @pytest.mark.asyncio
    async def test_handles_empty_delta(self):
        """Test handling of SSE events with empty or missing delta content."""
        client = OpenClawClient(gateway_token="test-token")
        
        sse_lines = [
            'data: {"choices":[{"delta":{}}]}',  # No content
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',  # Role only
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":""}}]}',  # Empty content
            'data: [DONE]',
        ]
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            tokens = []
            async for token in client.send_streaming("Hello", speaker="alice"):
                tokens.append(token)
        
        # Only "Hello" should be yielded
        assert tokens == ["Hello"]


class TestSend:
    """Test the non-streaming send method."""

    @pytest.mark.asyncio
    async def test_collects_all_tokens(self):
        """Test that send() collects all tokens into a single response."""
        client = OpenClawClient(gateway_token="test-token")
        
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" "}}]}',
            'data: {"choices":[{"delta":{"content":"world"}}]}',
            'data: [DONE]',
        ]
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.send("Test message", speaker="alice")
        
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self):
        """Test that send() returns None if no tokens received."""
        client = OpenClawClient(gateway_token="test-token")
        
        sse_lines = ['data: [DONE]']
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.send("Test message", speaker="alice")
        
        assert result is None


class TestRequestFormat:
    """Test the format of requests sent to OpenClaw."""

    @pytest.mark.asyncio
    async def test_request_url(self):
        """Test that requests go to the correct URL."""
        client = OpenClawClient(
            gateway_url="http://mygateway:9000",
            gateway_token="test-token",
        )
        
        captured_url = None
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(['data: [DONE]'])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        def capture_stream(method, url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return mock_response
        
        mock_client.stream = capture_stream
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            async for _ in client.send_streaming("Test", speaker="alice"):
                pass
        
        assert captured_url == "http://mygateway:9000/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_auth_header(self):
        """Test that auth header is sent correctly."""
        client = OpenClawClient(gateway_token="my-secret-token")
        
        captured_headers = None
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(['data: [DONE]'])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        def capture_stream(method, url, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers")
            return mock_response
        
        mock_client.stream = capture_stream
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            async for _ in client.send_streaming("Test", speaker="alice"):
                pass
        
        assert captured_headers["Authorization"] == "Bearer my-secret-token"

    @pytest.mark.asyncio
    async def test_model_format(self):
        """Test that model is formatted as openclaw:<agent_id>."""
        client = OpenClawClient(
            gateway_token="test-token",
            agent_id="custom-agent",
        )
        
        captured_payload = None
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(['data: [DONE]'])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        def capture_stream(method, url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return mock_response
        
        mock_client.stream = capture_stream
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            async for _ in client.send_streaming("Test", speaker="alice"):
                pass
        
        assert captured_payload["model"] == "openclaw:custom-agent"

    @pytest.mark.asyncio
    async def test_stream_flag(self):
        """Test that stream: true is set."""
        client = OpenClawClient(gateway_token="test-token")
        
        captured_payload = None
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(['data: [DONE]'])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        def capture_stream(method, url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return mock_response
        
        mock_client.stream = capture_stream
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            async for _ in client.send_streaming("Test", speaker="alice"):
                pass
        
        assert captured_payload["stream"] is True

    @pytest.mark.asyncio
    async def test_session_user_field(self):
        """Test that user field is set for session routing."""
        client = OpenClawClient(
            gateway_token="test-token",
            session_user="my-mumble-room",
        )
        
        captured_payload = None
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: MockAsyncIterator(['data: [DONE]'])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        def capture_stream(method, url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return mock_response
        
        mock_client.stream = capture_stream
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            async for _ in client.send_streaming("Test", speaker="alice"):
                pass
        
        assert captured_payload["user"] == "my-mumble-room"
