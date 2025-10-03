"""Reasoning recovery state and utilities for confidence-based early
stopping."""

import random
from dataclasses import dataclass

from aphrodite.common.sampling_params import SamplingParams


# Default recovery phrases
DEFAULT_RECOVERY_PHRASES = [
    "–wait...",
    "–actually, let me think differently...",
    "–I need to approach this another way...",
    "–let me reconsider this...",
    "–I need to step back and think...",
    "–let me approach this differently...",
    "–let me try a different approach...",
    "–I need to think more carefully...",
    "–let me reconsider my reasoning...",
    "–I should approach this step by step...",
]

DEFAULT_FINAL_ADMISSION = "–I'm not confident in my reasoning here, so let me \
                          provide what I can:"


@dataclass
class ReasoningRecoveryState:
    """State management for reasoning recovery during confidence-based early
    stopping."""

    recovery_count: int = 0
    max_attempts: int = 3
    recovery_phrases: list[str] = None
    final_admission: str = ""
    in_recovery: bool = False
    original_prompt_tokens: list[int] = None
    original_output_tokens: list[int] = None
    recovery_point_tokens: list[int] = None

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.recovery_phrases is None:
            self.recovery_phrases = DEFAULT_RECOVERY_PHRASES.copy()
        if not self.final_admission:
            self.final_admission = DEFAULT_FINAL_ADMISSION
        if self.original_prompt_tokens is None:
            self.original_prompt_tokens = []
        if self.original_output_tokens is None:
            self.original_output_tokens = []
        if self.recovery_point_tokens is None:
            self.recovery_point_tokens = []

    @classmethod
    def from_sampling_params(
        cls, 
        sampling_params: SamplingParams,
        prompt_tokens: list[int],
        output_tokens: list[int]
    ) -> "ReasoningRecoveryState":
        """Create ReasoningRecoveryState from SamplingParams and current
        tokens."""
        recovery_phrases = sampling_params.recovery_phrases
        if recovery_phrases is None:
            recovery_phrases = DEFAULT_RECOVERY_PHRASES.copy()

        final_admission = sampling_params.final_admission
        if final_admission is None:
            final_admission = DEFAULT_FINAL_ADMISSION

        return cls(
            max_attempts=sampling_params.max_recovery_attempts,
            recovery_phrases=recovery_phrases,
            final_admission=final_admission,
            original_prompt_tokens=prompt_tokens.copy(),
            original_output_tokens=output_tokens.copy(),
        )

    def can_recover(self) -> bool:
        """Check if we can still attempt recovery."""
        return self.recovery_count < self.max_attempts

    def get_recovery_phrase(self) -> str:
        """Get a random recovery phrase."""
        if not self.recovery_phrases:
            return DEFAULT_RECOVERY_PHRASES[0]
        return random.choice(self.recovery_phrases)

    def get_final_admission(self) -> str:
        """Get the final admission phrase."""
        return self.final_admission

    def start_recovery(self) -> str:
        """Start a recovery attempt and return the recovery phrase."""
        if not self.can_recover():
            raise ValueError("Cannot start recovery: max attempts reached")

        self.recovery_count += 1
        self.in_recovery = True
        return self.get_recovery_phrase()

    def finish_recovery(self) -> str:
        """Finish recovery and return the final admission phrase."""
        self.in_recovery = False
        return self.get_final_admission()

    def get_full_sequence(self) -> list[int]:
        """Get the full sequence (prompt + output tokens)."""
        return self.original_prompt_tokens + self.original_output_tokens

    def update_output_tokens(self, new_tokens: list[int]) -> None:
        """Update the current output tokens."""
        self.original_output_tokens.extend(new_tokens)

    def set_recovery_point(self, tokens: list[int]) -> None:
        """Set the recovery point tokens."""
        self.recovery_point_tokens = tokens.copy()

    def reset(self) -> None:
        """Reset the recovery state."""
        self.recovery_count = 0
        self.in_recovery = False
        self.recovery_point_tokens.clear()
