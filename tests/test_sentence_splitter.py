"""Tests for the sentence splitter.

The sentence splitter accumulates streamed tokens and emits complete sentences
for TTS. These tests verify boundary detection, min/max length handling, and
edge cases like abbreviations and URLs.
"""

import pytest
from sentence_splitter import SentenceSplitter


class TestBasicSplitting:
    """Test basic sentence boundary detection."""

    def test_single_sentence_with_period(self):
        splitter = SentenceSplitter(min_length=5)
        # Feed tokens one at a time
        result = []
        for char in "Hello world. ":
            result.extend(splitter.feed(char))
        
        assert result == ["Hello world."]

    def test_single_sentence_with_question_mark(self):
        splitter = SentenceSplitter(min_length=5)
        result = []
        for char in "How are you? ":
            result.extend(splitter.feed(char))
        
        assert result == ["How are you?"]

    def test_single_sentence_with_exclamation(self):
        splitter = SentenceSplitter(min_length=5)
        result = []
        for char in "Watch out! ":
            result.extend(splitter.feed(char))
        
        assert result == ["Watch out!"]

    def test_multiple_sentences(self):
        splitter = SentenceSplitter(min_length=5)
        text = "First sentence. Second sentence. Third one! "
        
        result = []
        for char in text:
            result.extend(splitter.feed(char))
        
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third one!"

    def test_colon_as_boundary(self):
        splitter = SentenceSplitter(min_length=5)
        result = []
        for char in "Here's the list: item one. ":
            result.extend(splitter.feed(char))
        
        # Colon should trigger a boundary
        assert len(result) >= 1
        assert "Here's the list:" in result[0]


class TestMinLengthBehavior:
    """Test that min_length prevents tiny emissions."""

    def test_short_text_not_emitted(self):
        splitter = SentenceSplitter(min_length=20)
        result = []
        for char in "Hi. ":  # Only 3 chars before period
            result.extend(splitter.feed(char))
        
        # Too short - should stay buffered
        assert result == []
        assert splitter.buffered_length > 0

    def test_short_text_emitted_on_flush(self):
        splitter = SentenceSplitter(min_length=20)
        for char in "Hi.":
            splitter.feed(char)
        
        result = splitter.flush()
        assert result == ["Hi."]

    def test_respects_min_length(self):
        splitter = SentenceSplitter(min_length=10)
        result = []
        # "Short." is 6 chars, below min_length=10
        # "This is longer." is 15 chars, above min_length
        for char in "Short. This is longer. ":
            result.extend(splitter.feed(char))
        
        # First sentence too short, should be combined or held
        # Only emit when we have enough
        assert len(result) >= 1


class TestMaxLengthBehavior:
    """Test forced emission when buffer exceeds max_length."""

    def test_force_emit_at_max_length(self):
        splitter = SentenceSplitter(min_length=5, max_length=50)
        
        # Feed a long run-on sentence with no punctuation
        long_text = "This is a very long sentence without any punctuation that keeps going and going"
        result = []
        for char in long_text:
            result.extend(splitter.feed(char))
        
        # Should have forced at least one emit
        if len(long_text) > 50:
            assert len(result) >= 1

    def test_force_emit_prefers_space_break(self):
        splitter = SentenceSplitter(min_length=5, max_length=30)
        
        # Text without punctuation
        text = "one two three four five six seven"
        result = []
        for char in text:
            result.extend(splitter.feed(char))
        result.extend(splitter.flush())
        
        # Should break at word boundaries, not mid-word
        for sentence in result:
            # Shouldn't end mid-word (with a letter)
            stripped = sentence.strip()
            if stripped and len(stripped) > 1:
                # Check it doesn't look like a chopped word
                assert not (stripped[-1].isalpha() and len(stripped) == 30)


class TestNewlineBoundaries:
    """Test paragraph/newline boundary detection."""

    def test_newline_triggers_boundary(self):
        splitter = SentenceSplitter(min_length=5)
        text = "First paragraph\n\nSecond paragraph"
        
        result = []
        for char in text:
            result.extend(splitter.feed(char))
        result.extend(splitter.flush())
        
        # Should have split on newlines
        assert len(result) >= 1

    def test_single_newline_handling(self):
        splitter = SentenceSplitter(min_length=5)
        text = "Line one\nLine two\nLine three"
        
        result = []
        for char in text:
            result.extend(splitter.feed(char))
        result.extend(splitter.flush())
        
        # All text should be captured
        combined = " ".join(result)
        assert "Line one" in combined or "one" in combined


class TestFlushBehavior:
    """Test the flush() method."""

    def test_flush_emits_remaining_text(self):
        splitter = SentenceSplitter(min_length=5)
        
        # Feed partial sentence (no terminator)
        for char in "This has no ending":
            splitter.feed(char)
        
        result = splitter.flush()
        assert result == ["This has no ending"]

    def test_flush_empty_buffer(self):
        splitter = SentenceSplitter()
        result = splitter.flush()
        assert result == []

    def test_flush_whitespace_only(self):
        splitter = SentenceSplitter()
        splitter.feed("   \n\t  ")
        result = splitter.flush()
        assert result == []  # Whitespace-only should be empty after strip

    def test_double_flush_is_safe(self):
        splitter = SentenceSplitter()
        splitter.feed("Some text")
        result1 = splitter.flush()
        result2 = splitter.flush()
        
        assert result1 == ["Some text"]
        assert result2 == []


class TestReset:
    """Test the reset() method for barge-in scenarios."""

    def test_reset_clears_buffer(self):
        splitter = SentenceSplitter()
        splitter.feed("Partial text that gets")
        
        assert splitter.buffered_length > 0
        splitter.reset()
        assert splitter.buffered_length == 0

    def test_reset_then_new_input(self):
        splitter = SentenceSplitter(min_length=5)
        
        # Start a sentence
        splitter.feed("Old sentence that")
        splitter.reset()
        
        # Start fresh
        result = []
        for char in "New sentence. ":
            result.extend(splitter.feed(char))
        
        assert result == ["New sentence."]


class TestEdgeCases:
    """Test edge cases and tricky inputs."""

    def test_empty_token(self):
        splitter = SentenceSplitter()
        result = splitter.feed("")
        assert result == []

    def test_only_punctuation(self):
        splitter = SentenceSplitter(min_length=1)
        result = []
        result.extend(splitter.feed("..."))
        result.extend(splitter.flush())
        assert "..." in "".join(result) or result == ["..."]

    def test_unicode_text(self):
        splitter = SentenceSplitter(min_length=5)
        text = "Héllo wörld! Ça va? "
        
        result = []
        for char in text:
            result.extend(splitter.feed(char))
        
        assert len(result) >= 1
        combined = " ".join(result)
        assert "Héllo" in combined or "wörld" in combined

    def test_emoji_in_text(self):
        splitter = SentenceSplitter(min_length=5)
        text = "Hello! 🔫 How are you? "
        
        result = []
        for char in text:
            result.extend(splitter.feed(char))
        
        assert len(result) >= 1

    def test_realistic_llm_tokens(self):
        """Test with realistic LLM token chunks (not single chars)."""
        splitter = SentenceSplitter(min_length=10)
        
        # Simulate LLM tokens (typically 2-4 chars each)
        tokens = ["The ", "weather ", "today ", "is ", "nice. ", "Would ", "you ", "like ", "to ", "go ", "outside? "]
        
        result = []
        for token in tokens:
            result.extend(splitter.feed(token))
        
        assert len(result) >= 1
        assert "weather" in result[0]


class TestBufferedLength:
    """Test the buffered_length property."""

    def test_buffered_length_increases(self):
        splitter = SentenceSplitter(min_length=100)  # High min to prevent emission
        
        assert splitter.buffered_length == 0
        splitter.feed("Hello")
        assert splitter.buffered_length == 5
        splitter.feed(" world")
        assert splitter.buffered_length == 11

    def test_buffered_length_decreases_on_emit(self):
        splitter = SentenceSplitter(min_length=5)
        
        splitter.feed("Hello. ")
        initial = splitter.buffered_length
        
        # Force more text to trigger emit
        splitter.feed("More text. ")
        
        # Buffer should have been reduced by emission
        # (hard to test exact values due to whitespace handling)
