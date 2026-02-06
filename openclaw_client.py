"""OpenClaw Gateway client via OpenAI-compatible Chat Completions API.

Sends messages to the OpenClaw Gateway's /v1/chat/completions endpoint
and consumes the SSE stream token-by-token. Supports cancellation for
barge-in by closing the HTTP connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenClawClient:
    """Client for OpenClaw Gateway's OpenAI-compatible Chat Completions API.

    Attributes:
        gateway_url: Base URL of the Gateway (e.g., "http://127.0.0.1:18789").
        gateway_token: Authentication token.
        agent_id: Target agent (default "main").
        session_user: Fixed user string for shared session key.
        timeout_seconds: Request timeout.
    """

    gateway_url: str = "http://127.0.0.1:18789"
    gateway_token: str = ""
    agent_id: str = "main"
    session_user: str = "mumble-room"
    timeout_seconds: int = 60

    async def send_streaming(
        self,
        text: str,
        speaker: str = "unknown",
        cancellation_event: asyncio.Event | None = None,
    ) -> AsyncIterator[str]:
        """Send a message and yield response tokens as they arrive via SSE.

        Args:
            text: The transcribed user message.
            speaker: Mumble username for attribution in the shared session.
            cancellation_event: If set, stop reading the stream.

        Yields:
            Text tokens (content deltas) from the assistant response.
        """
        import httpx

        message_content = f"[{speaker}]: {text}"
        url = f"{self.gateway_url}/v1/chat/completions"

        payload = {
            "model": f"openclaw:{self.agent_id}",
            "stream": True,
            "user": self.session_user,
            "messages": [{"role": "user", "content": message_content}],
        }

        headers = {
            "Authorization": f"Bearer {self.gateway_token}",
            "Content-Type": "application/json",
        }

        logger.debug("Sending to OpenClaw: %s", message_content[:100])

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream(
                    "POST", url, json=payload, headers=headers
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        # Check cancellation between lines
                        if cancellation_event and cancellation_event.is_set():
                            logger.info("SSE stream cancelled (barge-in)")
                            return

                        if not line.startswith("data: "):
                            continue

                        data = line[6:]  # Strip "data: " prefix

                        if data == "[DONE]":
                            logger.debug("SSE stream complete")
                            return

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            logger.warning("Malformed SSE data: %s", data[:200])

        except httpx.HTTPStatusError as e:
            logger.error(
                "OpenClaw API error: %s %s", e.response.status_code, e.response.text
            )
        except httpx.RequestError as e:
            logger.error("OpenClaw API request failed: %s", e)

    async def send(
        self,
        text: str,
        speaker: str = "unknown",
    ) -> str | None:
        """Send a message and return the complete response (non-streaming).

        Convenience method for testing. Collects all tokens from the stream.

        Args:
            text: The transcribed user message.
            speaker: Mumble username.

        Returns:
            Complete response text, or None on failure.
        """
        tokens = []
        async for token in self.send_streaming(text, speaker):
            tokens.append(token)
        result = "".join(tokens)
        return result if result else None
