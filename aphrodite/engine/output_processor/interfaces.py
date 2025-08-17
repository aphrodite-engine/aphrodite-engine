from abc import ABC, abstractmethod
from typing import Callable, List

from aphrodite.common.sequence import (Sequence, SequenceGroup,
                                       SequenceGroupOutput)
from aphrodite.config import SchedulerConfig
from aphrodite.engine.output_processor.stop_checker import StopChecker
from aphrodite.processing.scheduler import Scheduler
from aphrodite.transformers_utils.detokenizer import Detokenizer
from aphrodite.transformers_utils.tokenizer import AnyTokenizer
from aphrodite.utils import Counter


class SequenceGroupOutputProcessor(ABC):
    """Interface for logic that processes new token ids in sequence groups,
    managing detokenization, stop checking, and freeing/forking sequences with
    the scheduler.

    This is highly coupled with the AphroditeEngine and should be seen as an
    extension of it. The logic is separated to simplify the AphroditeEngine
    class and allow separate implementations for single-step decoding
    (which supports beam search sequence forking) and multi-step decoding
    (which does not support beam search, but does support speculative decoding)
    """

    @staticmethod
    def create_output_processor(
        scheduler_config: SchedulerConfig,
        detokenizer: Detokenizer,
        scheduler: List[Scheduler],
        seq_counter: Counter,
        get_tokenizer_for_seq: Callable[[Sequence], AnyTokenizer],
        stop_checker: "StopChecker",
    ):
        """Create an output processor.

        single-step output processor.
        """
        from aphrodite.engine.output_processor.single_step import (
            SingleStepOutputProcessor)
        return SingleStepOutputProcessor(scheduler_config, detokenizer,
                                         scheduler, seq_counter, stop_checker)

    @abstractmethod
    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput],
                        is_async: bool) -> None:
        """Process new token ids for the sequence group. Handles logic such as
        detokenization, stop checking, and freeing/forking sequences in the
        scheduler.
        """
        pass

    @abstractmethod
    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        """Update prompt logprobs received from outputs to seq_group."""
        pass
