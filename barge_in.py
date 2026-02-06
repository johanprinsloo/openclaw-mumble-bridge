"""Barge-in controller state machine.

Manages the lifecycle of voice interactions:
  IDLE → LISTENING → PROCESSING → SPEAKING → IDLE

When a user starts PTT while in SPEAKING state, triggers barge-in:
  SPEAKING → (cancel playback + pending TTS) → LISTENING
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


class BridgeState(enum.Enum):
    """States of the Mumble-OpenClaw voice bridge."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# Valid state transitions
_VALID_TRANSITIONS: dict[BridgeState, set[BridgeState]] = {
    BridgeState.IDLE: {BridgeState.LISTENING},
    BridgeState.LISTENING: {BridgeState.PROCESSING, BridgeState.IDLE},
    BridgeState.PROCESSING: {BridgeState.SPEAKING, BridgeState.IDLE, BridgeState.LISTENING},
    BridgeState.SPEAKING: {BridgeState.IDLE, BridgeState.LISTENING},
}


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


@dataclass
class BargeInController:
    """Controls the barge-in state machine and cancellation.

    Thread-safe: uses asyncio primitives. All methods should be called
    from the same event loop.

    Attributes:
        state: Current bridge state.
        cancellation_event: Set when barge-in occurs. Checked by TTS and
            playback tasks to abort early.
        on_state_change: Optional callback invoked on each transition.
    """

    on_state_change: Callable[[BridgeState, BridgeState], None] | None = None
    _state: BridgeState = field(default=BridgeState.IDLE, init=False)
    _cancellation_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _barge_in_count: int = field(default=0, init=False)
    _last_transition_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> BridgeState:
        return self._state

    @property
    def cancellation_event(self) -> asyncio.Event:
        """Event that is set when barge-in cancellation is requested.

        TTS tasks, playback loops, and the OpenClaw SSE reader should
        check `cancellation_event.is_set()` between operations and abort
        if True.
        """
        return self._cancellation_event

    @property
    def is_cancelled(self) -> bool:
        """Convenience: check if cancellation is active."""
        return self._cancellation_event.is_set()

    @property
    def barge_in_count(self) -> int:
        """Number of barge-in interrupts since creation."""
        return self._barge_in_count

    def transition_to(self, new_state: BridgeState) -> None:
        """Transition to a new state.

        Args:
            new_state: The target state.

        Raises:
            InvalidTransitionError: If the transition is not valid.
        """
        old_state = self._state
        if new_state not in _VALID_TRANSITIONS.get(old_state, set()):
            raise InvalidTransitionError(
                f"Cannot transition from {old_state.value} to {new_state.value}"
            )

        self._state = new_state
        self._last_transition_time = time.monotonic()
        logger.debug("State: %s → %s", old_state.value, new_state.value)

        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def on_ptt_start(self, user: str) -> bool:
        """Handle a user pressing PTT (starting to transmit).

        If currently IDLE, transitions to LISTENING.
        If currently SPEAKING, triggers barge-in: cancels playback and
        transitions to LISTENING.

        Args:
            user: Mumble username who started PTT.

        Returns:
            True if a barge-in occurred (was SPEAKING), False otherwise.
        """
        barged = False

        if self._state == BridgeState.IDLE:
            self.transition_to(BridgeState.LISTENING)

        elif self._state == BridgeState.SPEAKING:
            logger.info("Barge-in by %s — cancelling playback", user)
            self._trigger_cancellation()
            self._state = BridgeState.LISTENING
            self._last_transition_time = time.monotonic()
            barged = True
            if self.on_state_change:
                self.on_state_change(BridgeState.SPEAKING, BridgeState.LISTENING)

        elif self._state == BridgeState.PROCESSING:
            # Someone PTTs while we're waiting for OpenClaw response
            # Barge-in: cancel the in-flight request and listen to new input
            logger.info("Barge-in by %s during processing — cancelling", user)
            self._trigger_cancellation()
            self._state = BridgeState.LISTENING
            self._last_transition_time = time.monotonic()
            barged = True
            if self.on_state_change:
                self.on_state_change(BridgeState.PROCESSING, BridgeState.LISTENING)

        elif self._state == BridgeState.LISTENING:
            # Already listening (maybe another user also PTT'd) — no-op
            logger.debug("Already LISTENING, PTT from %s ignored", user)

        return barged

    def on_ptt_end(self, user: str) -> None:
        """Handle a user releasing PTT (stopping transmission).

        If currently LISTENING, transitions to PROCESSING.

        Args:
            user: Mumble username who released PTT.
        """
        if self._state == BridgeState.LISTENING:
            self.transition_to(BridgeState.PROCESSING)

    def on_response_start(self) -> None:
        """Called when the first TTS audio chunk is ready to play.

        Transitions from PROCESSING to SPEAKING.
        """
        if self._state == BridgeState.PROCESSING:
            self.transition_to(BridgeState.SPEAKING)

    def on_playback_complete(self) -> None:
        """Called when all TTS audio has been played.

        Transitions from SPEAKING to IDLE.
        """
        if self._state == BridgeState.SPEAKING:
            self.transition_to(BridgeState.IDLE)

    def on_error(self) -> None:
        """Called when an error occurs during processing.

        Returns to IDLE from any state.
        """
        if self._state != BridgeState.IDLE:
            old = self._state
            self._state = BridgeState.IDLE
            self._last_transition_time = time.monotonic()
            logger.warning("Error recovery: %s → IDLE", old.value)
            if self.on_state_change:
                self.on_state_change(old, BridgeState.IDLE)

    def prepare_new_response(self) -> None:
        """Clear the cancellation event before starting a new response cycle.

        Call this at the beginning of the STT→OpenClaw→TTS pipeline,
        after transitioning to PROCESSING.
        """
        self._cancellation_event.clear()

    def _trigger_cancellation(self) -> None:
        """Set the cancellation event and increment barge-in counter."""
        self._cancellation_event.set()
        self._barge_in_count += 1
        logger.debug("Cancellation triggered (total barge-ins: %d)", self._barge_in_count)
