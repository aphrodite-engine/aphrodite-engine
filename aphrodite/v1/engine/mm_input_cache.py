from collections.abc import Sequence
from typing import Optional

from aphrodite.common.envs import APHRODITE_MM_INPUT_CACHE_GIB
from aphrodite.common.utils import is_list_of
from aphrodite.multimodal import MultiModalKwargs
from aphrodite.multimodal.processing import ProcessingCache

# The idea of multimodal preprocessing caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# -- Client:
#  - BaseMultiModalProcessor to process MultiModalData into MultiModalKwargs
#    with built-in caching functionality, with mm_hash as its identifier.
#  - MirroredProcessingCache to keep track of the cached entries and
#    determine whether to send the MultiModalKwargs to P1.
#
# -- Server:
#  - MirroredProcessingCache to store the MultiModalKwargs from P0.
#
# The caching for both client and server is mirrored, and this allows us
# to avoid the serialization of "mm_inputs" (like pixel values) between
# client (=P0) and server (=P1) processes if the mm_hash is found in the client
# cache.

# Both Client and Server must use the same cache size
# (to perform mirrored caching). This cache size is set by the environment
# variable APHRODITE_MM_INPUT_CACHE_GIB.


class MirroredProcessingCache:

    def __init__(self, model_config):
        mm_config = model_config.multimodal_config
        disable_mm_preprocessor_cache = mm_config is not None and \
            not mm_config.disable_mm_preprocessor_cache
        self.use_cache = not disable_mm_preprocessor_cache
        self.mm_cache = ProcessingCache.get_lru_cache(APHRODITE_MM_INPUT_CACHE_GIB,
                                                      MultiModalKwargs)

    def get_and_update_p0(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs

        full_mm_inputs = list[Optional[MultiModalKwargs]]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if self.mm_cache.get(mm_hash) is not None:
                mm_input = None
            else:
                self.mm_cache[mm_hash] = mm_input

            full_mm_inputs.append(mm_input)

        return full_mm_inputs

    def get_and_update_p1(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs

        full_mm_inputs = list[MultiModalKwargs]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if mm_input is None:
                mm_input = self.mm_cache[mm_hash]
            else:
                self.mm_cache[mm_hash] = mm_input

            full_mm_inputs.append(mm_input)

        return full_mm_inputs
