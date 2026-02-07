"""Tests for the barge-in controller state machine.

The barge-in controller manages the lifecycle of voice interactions:
IDLE → LISTENING → PROCESSING → SPEAKING → IDLE

Key behavior: when a user starts PTT while in SPEAKING state, it triggers
a barge-in which cancels playback and transitions back to LISTENING.
"""

import pytest
import asyncio
from barge_in import BargeInController, BridgeState, InvalidTransitionError


class TestInitialState:
    """Test initial controller state."""

    def test_starts_in_idle(self):
        controller = BargeInController()
        assert controller.state == BridgeState.IDLE

    def test_cancellation_event_starts_unset(self):
        controller = BargeInController()
        assert not controller.is_cancelled
        assert not controller.cancellation_event.is_set()

    def test_barge_in_count_starts_at_zero(self):
        controller = BargeInController()
        assert controller.barge_in_count == 0


class TestNormalFlow:
    """Test the normal IDLE → LISTENING → PROCESSING → SPEAKING → IDLE flow."""

    def test_ptt_start_from_idle(self):
        controller = BargeInController()
        
        barged = controller.on_ptt_start("user1")
        
        assert controller.state == BridgeState.LISTENING
        assert not barged  # No barge-in from IDLE

    def test_ptt_end_transitions_to_processing(self):
        controller = BargeInController()
        controller.on_ptt_start("user1")
        
        controller.on_ptt_end("user1")
        
        assert controller.state == BridgeState.PROCESSING

    def test_response_start_transitions_to_speaking(self):
        controller = BargeInController()
        controller.on_ptt_start("user1")
        controller.on_ptt_end("user1")
        
        controller.on_response_start()
        
        assert controller.state == BridgeState.SPEAKING

    def test_playback_complete_transitions_to_idle(self):
        controller = BargeInController()
        controller.on_ptt_start("user1")
        controller.on_ptt_end("user1")
        controller.on_response_start()
        
        controller.on_playback_complete()
        
        assert controller.state == BridgeState.IDLE

    def test_full_cycle(self):
        """Test a complete interaction cycle."""
        controller = BargeInController()
        
        # User presses PTT
        controller.on_ptt_start("alice")
        assert controller.state == BridgeState.LISTENING
        
        # User releases PTT
        controller.on_ptt_end("alice")
        assert controller.state == BridgeState.PROCESSING
        
        # Prepare for new response
        controller.prepare_new_response()
        assert not controller.is_cancelled
        
        # First TTS chunk ready
        controller.on_response_start()
        assert controller.state == BridgeState.SPEAKING
        
        # Playback finishes
        controller.on_playback_complete()
        assert controller.state == BridgeState.IDLE


class TestBargeIn:
    """Test barge-in behavior (interruption during playback)."""

    def test_barge_in_during_speaking(self):
        controller = BargeInController()
        # Get to SPEAKING state
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.prepare_new_response()
        controller.on_response_start()
        assert controller.state == BridgeState.SPEAKING
        
        # Another user (or same user) starts PTT = barge-in
        barged = controller.on_ptt_start("bob")
        
        assert barged is True
        assert controller.state == BridgeState.LISTENING
        assert controller.is_cancelled
        assert controller.barge_in_count == 1

    def test_barge_in_during_processing(self):
        controller = BargeInController()
        # Get to PROCESSING state
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        assert controller.state == BridgeState.PROCESSING
        
        # Another user starts PTT while we're waiting for OpenClaw
        barged = controller.on_ptt_start("bob")
        
        assert barged is True
        assert controller.state == BridgeState.LISTENING
        assert controller.is_cancelled
        assert controller.barge_in_count == 1

    def test_barge_in_sets_cancellation_event(self):
        controller = BargeInController()
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.on_response_start()
        
        assert not controller.cancellation_event.is_set()
        
        controller.on_ptt_start("bob")
        
        assert controller.cancellation_event.is_set()
        assert controller.is_cancelled

    def test_multiple_barge_ins_increment_counter(self):
        controller = BargeInController()
        
        # First cycle with barge-in
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.on_response_start()
        controller.on_ptt_start("bob")  # Barge-in 1
        
        # Second cycle with barge-in
        controller.on_ptt_end("bob")
        controller.prepare_new_response()
        controller.on_response_start()
        controller.on_ptt_start("charlie")  # Barge-in 2
        
        assert controller.barge_in_count == 2

    def test_same_user_can_barge_in(self):
        """The same user who asked the question can interrupt."""
        controller = BargeInController()
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.prepare_new_response()
        controller.on_response_start()
        
        # Alice interrupts her own response
        barged = controller.on_ptt_start("alice")
        
        assert barged is True
        assert controller.state == BridgeState.LISTENING


class TestCancellationEvent:
    """Test cancellation event management."""

    def test_prepare_new_response_clears_cancellation(self):
        controller = BargeInController()
        
        # Trigger a barge-in to set cancellation
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.on_response_start()
        controller.on_ptt_start("bob")
        
        assert controller.is_cancelled
        
        # Start new response cycle
        controller.on_ptt_end("bob")
        controller.prepare_new_response()
        
        assert not controller.is_cancelled

    def test_cancellation_event_is_asyncio_event(self):
        controller = BargeInController()
        assert isinstance(controller.cancellation_event, asyncio.Event)


class TestErrorRecovery:
    """Test error recovery via on_error()."""

    def test_error_returns_to_idle_from_listening(self):
        controller = BargeInController()
        controller.on_ptt_start("alice")
        assert controller.state == BridgeState.LISTENING
        
        controller.on_error()
        assert controller.state == BridgeState.IDLE

    def test_error_returns_to_idle_from_processing(self):
        controller = BargeInController()
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        assert controller.state == BridgeState.PROCESSING
        
        controller.on_error()
        assert controller.state == BridgeState.IDLE

    def test_error_returns_to_idle_from_speaking(self):
        controller = BargeInController()
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.on_response_start()
        assert controller.state == BridgeState.SPEAKING
        
        controller.on_error()
        assert controller.state == BridgeState.IDLE

    def test_error_from_idle_is_noop(self):
        controller = BargeInController()
        assert controller.state == BridgeState.IDLE
        
        controller.on_error()  # Should not raise
        assert controller.state == BridgeState.IDLE


class TestStateChangeCallback:
    """Test the on_state_change callback."""

    def test_callback_is_called_on_transition(self):
        transitions = []
        
        def on_change(old, new):
            transitions.append((old, new))
        
        controller = BargeInController(on_state_change=on_change)
        
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.on_response_start()
        controller.on_playback_complete()
        
        assert len(transitions) == 4
        assert transitions[0] == (BridgeState.IDLE, BridgeState.LISTENING)
        assert transitions[1] == (BridgeState.LISTENING, BridgeState.PROCESSING)
        assert transitions[2] == (BridgeState.PROCESSING, BridgeState.SPEAKING)
        assert transitions[3] == (BridgeState.SPEAKING, BridgeState.IDLE)

    def test_callback_called_on_barge_in(self):
        transitions = []
        
        def on_change(old, new):
            transitions.append((old, new))
        
        controller = BargeInController(on_state_change=on_change)
        controller.on_ptt_start("alice")
        controller.on_ptt_end("alice")
        controller.on_response_start()
        transitions.clear()
        
        controller.on_ptt_start("bob")  # Barge-in
        
        assert len(transitions) == 1
        assert transitions[0] == (BridgeState.SPEAKING, BridgeState.LISTENING)


class TestEdgeCases:
    """Test edge cases and unusual sequences."""

    def test_ptt_start_while_listening_is_ignored(self):
        """If already listening, another PTT start is a no-op."""
        controller = BargeInController()
        controller.on_ptt_start("alice")
        assert controller.state == BridgeState.LISTENING
        
        # Another user also PTTs
        barged = controller.on_ptt_start("bob")
        
        assert controller.state == BridgeState.LISTENING  # Still listening
        assert not barged

    def test_playback_complete_from_wrong_state_is_safe(self):
        """on_playback_complete from non-SPEAKING state should be safe."""
        controller = BargeInController()
        
        # Should not raise
        controller.on_playback_complete()
        assert controller.state == BridgeState.IDLE

    def test_response_start_from_wrong_state_is_safe(self):
        """on_response_start from non-PROCESSING state should be safe."""
        controller = BargeInController()
        
        # Should not raise
        controller.on_response_start()
        assert controller.state == BridgeState.IDLE

    def test_ptt_end_from_idle_is_safe(self):
        """on_ptt_end from IDLE should be safe (no-op)."""
        controller = BargeInController()
        
        controller.on_ptt_end("alice")  # Should not raise
        assert controller.state == BridgeState.IDLE


class TestTransitionValidation:
    """Test that invalid direct transitions are rejected."""

    def test_cannot_transition_idle_to_processing(self):
        controller = BargeInController()
        
        with pytest.raises(InvalidTransitionError):
            controller.transition_to(BridgeState.PROCESSING)

    def test_cannot_transition_idle_to_speaking(self):
        controller = BargeInController()
        
        with pytest.raises(InvalidTransitionError):
            controller.transition_to(BridgeState.SPEAKING)

    def test_cannot_transition_listening_to_speaking(self):
        controller = BargeInController()
        controller.transition_to(BridgeState.LISTENING)
        
        with pytest.raises(InvalidTransitionError):
            controller.transition_to(BridgeState.SPEAKING)
