import itertools
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from aphrodite.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from aphrodite.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_ids_list_to_tokens)
from aphrodite.v1.engine import EngineCoreOutput, EngineCoreRequest
from aphrodite.v1.engine.reasoning_recovery import ReasoningRecoveryState
from aphrodite.v1.outputs import LogprobsLists, LogprobsTensors

NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:

    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Optional[AnyTokenizer]

    # Logprobs for this request
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]
    num_logprobs: Optional[int]
    num_prompt_logprobs: Optional[int]

    conf_grouped: Optional[float]
    conf_list: Optional[list[float]]
    conf_group_list: Optional[deque[float]]
    conf_group_size: Optional[int]
    conf_threshold: Optional[float]

    reasoning_recovery_state: Optional[ReasoningRecoveryState] = None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        assert request.sampling_params is not None
        num_logprobs = request.sampling_params.logprobs
        num_prompt_logprobs = request.sampling_params.prompt_logprobs

        if request.sampling_params.enable_deepconf:
            conf_group_size = request.sampling_params.deepconf_window_size
            conf_threshold = request.sampling_params.deepconf_threshold
            conf_grouped = 0.0
            conf_group_list = deque(maxlen=conf_group_size)
            conf_list = []
        else:
            conf_group_size = -1
            conf_threshold = None
            conf_grouped = 0.0
            conf_group_list = None
            conf_list = None

        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.),
            logprobs=(None if num_logprobs is None else []),
            # NOTE: logprob of first prompt token is None.
            prompt_logprobs=(None if num_prompt_logprobs is None else [None]),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
            conf_group_size=conf_group_size,
            conf_grouped=conf_grouped,
            conf_list=conf_list,
            conf_threshold=conf_threshold,
            conf_group_list=conf_group_list,
        )

    def check_conf_stop(self) -> bool:
        """Return True if the confidence window triggers early stopping."""
        if self.conf_group_list is None or len(self.conf_group_list) == 0:
            return False
        # Require a full window; trigger when the moving average is below
        # threshold.
        return (len(self.conf_group_list) >= self.conf_group_size
                and self.conf_grouped / len(self.conf_group_list) <
                self.conf_threshold)
    
    def check_reasoning_recovery_stop(self) -> tuple[bool, str, bool]:
        """
        Check if reasoning recovery should be triggered.
        
        Returns:
            tuple[bool, str, bool]: (should_stop, recovery_phrase, 
                                   is_final_admission)
        """
        if self.reasoning_recovery_state is None:
            return self.check_conf_stop(), "", False

        if not self.check_conf_stop():
            return False, "", False

        if self.reasoning_recovery_state.can_recover():
            recovery_phrase = self.reasoning_recovery_state.start_recovery()
            return False, recovery_phrase, False
        else:
            final_phrase = self.reasoning_recovery_state.finish_recovery()
            return True, final_phrase, True

    def initialize_reasoning_recovery(
        self, 
        sampling_params, 
        prompt_tokens: list[int], 
        output_tokens: list[int]
    ) -> None:
        """Initialize reasoning recovery state if enabled."""
        if sampling_params.enable_reasoning_recovery:
            self.reasoning_recovery_state = (
                ReasoningRecoveryState.from_sampling_params(
                    sampling_params, prompt_tokens, output_tokens
                )
            )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst = logprobs_lists

        for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst,
                                             token_ids_lst):

            # Detokenize (non-incrementally).
            decoded_tokens = NONES if self.tokenizer is None else (
                convert_ids_list_to_tokens(self.tokenizer, token_ids))

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob dictionary for this pos.
            self.logprobs.append(
                self._make_logprob_dict(
                    logprobs,
                    token_ids,
                    decoded_tokens,
                    rank,
                    self.num_logprobs,
                ))

            if self.conf_list is not None:
                # logprobs[0] is the sampled token; use the remaining
                # candidates
                if len(logprobs) > 1:
                    new_conf = -sum(logprobs[1:]) / len(logprobs[1:])
                else:
                    new_conf = 0.0
                self.conf_list.append(new_conf)

                if len(self.conf_group_list) < self.conf_group_size:
                    self.conf_group_list.append(new_conf)
                    self.conf_grouped += new_conf
                else:
                    self.conf_grouped -= self.conf_group_list.popleft()
                    self.conf_group_list.append(new_conf)
                    self.conf_grouped += new_conf

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = None if self.tokenizer is None else (
            convert_ids_list_to_tokens(self.tokenizer,
                                       token_ids.flatten().tolist()))

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = NONES \
            if decoded_tokens is None else decoded_tokens[offset:offset_end]

            # Update with the Logprob dictionary for this pos.
            self.prompt_logprobs.append(
                self._make_logprob_dict(prompt_logprobs[pos], token_ids[pos],
                                        decoded_tokens_for_pos,
                                        prompt_token_ranks[pos],
                                        self.num_prompt_logprobs))

    def pop_prompt_logprobs(self) -> Optional[PromptLogprobs]:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    @staticmethod
    def _make_logprob_dict(
        logprobs: list[float],
        logprob_token_ids: list[int],
        decoded_tokens: Iterable[Optional[str]],
        rank: int,
        num_logprobs: int,
    ) -> dict[int, Logprob]:
        """Make a Logprob dictionary for a position.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)

        Returns:
          dict[token id, Logprob]
        """
        if num_logprobs == -1:
            num_logprobs = len(logprobs)
        # We do not need a special case for the sampled token
        # being in the topk, since inserting duplicated data
        # into a dictionary twice is the same as doing it once.
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank, ), topk_ranks)

        return {
            token_id: Logprob(
                logprob=logprob,
                rank=rank,
                decoded_token=token,
            )
            for token_id, logprob, rank, token in zip(
                logprob_token_ids, logprobs, ranks, decoded_tokens)
        }

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
