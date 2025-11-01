from aphrodite.reasoning.abs_reasoning_parsers import ReasoningParserManager
from aphrodite.reasoning.basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module("seed_oss")
class SeedOSSReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for SeedOSS model.

    The SeedOSS model uses <seed:think>...</seed:think> tokens to
    denote reasoning content text. This parser extracts
    the reasoning content from the model output.
    Similar to DeepSeek R1, it supports cases
    where the model doesn't generate the start token.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<seed:think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</seed:think>"
