"""Sentence splitter for streaming TTS.

Accumulates tokens from the OpenClaw SSE stream and emits complete sentences
suitable for individual TTS API calls. Balances latency (emit early) against
TTS quality (don't emit tiny fragments).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Sentence-ending punctuation followed by space, newline, or end-of-input
_SENTENCE_END_RE = re.compile(r'[.!?:;](?:\s|$)')

# Minimum characters before we'll emit (avoid tiny TTS calls)
DEFAULT_MIN_LENGTH = 20

# Maximum buffer before forced emit (safety valve for run-on text)
DEFAULT_MAX_LENGTH = 500


@dataclass
class SentenceSplitter:
    """Accumulates streamed text tokens and yields complete sentences.

    Usage:
        splitter = SentenceSplitter()
        for token in sse_stream:
            for sentence in splitter.feed(token):
                tts_queue.put(sentence)
        # When stream ends:
        for sentence in splitter.flush():
            tts_queue.put(sentence)
    """

    min_length: int = DEFAULT_MIN_LENGTH
    max_length: int = DEFAULT_MAX_LENGTH
    _buffer: str = field(default="", init=False, repr=False)

    def feed(self, token: str) -> list[str]:
        """Feed a token into the splitter.

        Args:
            token: A text chunk from the SSE stream (typically a few characters).

        Returns:
            List of complete sentences ready for TTS (may be empty).
        """
        self._buffer += token
        return self._try_emit()

    def flush(self) -> list[str]:
        """Flush any remaining buffered text.

        Call this when the SSE stream ends (receives [DONE]).

        Returns:
            List containing the remaining text, or empty if buffer is empty.
        """
        text = self._buffer.strip()
        self._buffer = ""
        if text:
            return [text]
        return []

    def reset(self):
        """Clear the buffer. Used on barge-in to discard partial responses."""
        self._buffer = ""

    @property
    def buffered_length(self) -> int:
        """Current buffer length in characters."""
        return len(self._buffer)

    def _try_emit(self) -> list[str]:
        """Try to extract complete sentences from the buffer."""
        sentences: list[str] = []

        while True:
            # Force emit if buffer exceeds max length
            if len(self._buffer) >= self.max_length:
                # Try to find a good break point
                break_pos = self._find_break_point(self._buffer[: self.max_length])
                if break_pos > 0:
                    sentence = self._buffer[:break_pos].strip()
                    self._buffer = self._buffer[break_pos:]
                else:
                    # No good break point; force split at max_length
                    sentence = self._buffer[: self.max_length].strip()
                    self._buffer = self._buffer[self.max_length :]
                if sentence:
                    sentences.append(sentence)
                continue

            # Look for sentence boundaries only if we have enough text
            if len(self._buffer) < self.min_length:
                break

            # Find sentence-ending punctuation
            break_pos = self._find_sentence_end(self._buffer)
            if break_pos > 0 and break_pos >= self.min_length:
                sentence = self._buffer[:break_pos].strip()
                self._buffer = self._buffer[break_pos:]
                if sentence:
                    sentences.append(sentence)
                continue

            # Check for newline boundaries (paragraph breaks)
            newline_pos = self._find_newline_break(self._buffer)
            if newline_pos > 0 and newline_pos >= self.min_length:
                sentence = self._buffer[:newline_pos].strip()
                self._buffer = self._buffer[newline_pos:]
                if sentence:
                    sentences.append(sentence)
                continue

            # Not enough text or no boundary found yet
            break

        return sentences

    def _find_sentence_end(self, text: str) -> int:
        """Find the position after the first sentence-ending punctuation.

        Returns the character index to split at, or 0 if no boundary found.
        """
        for match in _SENTENCE_END_RE.finditer(text):
            candidate = match.end()
            if candidate >= self.min_length:
                return candidate
        return 0

    def _find_newline_break(self, text: str) -> int:
        """Find position of the last newline boundary suitable for a break."""
        # Look for double newline (paragraph break) or single newline
        last_pos = 0
        for i, ch in enumerate(text):
            if ch == "\n" and i >= self.min_length:
                last_pos = i + 1
        return last_pos

    def _find_break_point(self, text: str) -> int:
        """Find the best break point within text, preferring sentence ends > newlines > spaces."""
        # Prefer sentence end
        pos = self._find_sentence_end(text)
        if pos > 0:
            return pos

        # Then newline
        pos = self._find_newline_break(text)
        if pos > 0:
            return pos

        # Then last space
        last_space = text.rfind(" ")
        if last_space > 0:
            return last_space + 1

        return 0
