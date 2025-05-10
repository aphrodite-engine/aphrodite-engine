import itertools
import logging
import operator
import os
from dataclasses import dataclass, field
from typing import List, Tuple
from .common import WeakSingleton

logger = logging.getLogger(__name__)

@dataclass
class HPUBucketingGlobalState(metaclass=WeakSingleton):
    prompt_bs_bucket_cfg: Tuple[int, int, int] = field(init=False)
    decode_bs_bucket_cfg: Tuple[int, int, int] = field(init=False)
    prompt_seq_bucket_cfg: Tuple[int, int, int] = field(init=False)
    decode_block_bucket_cfg: Tuple[int, int, int] = field(init=False)
    prompt_buckets: List[Tuple[int, int]] = field(init=False)
    decode_buckets: List[Tuple[int, int]] = field(init=False)


class HPUBucketingContext(metaclass=WeakSingleton):
    global_state = HPUBucketingGlobalState()

    def __init__(self, max_num_seqs, max_num_prefill_seqs, block_size,
                 max_num_batched_tokens, use_merged_prefill, max_model_len=None,
                 max_prompt_seq=None, max_decode_seq=None):
        """
        Initializes the bucketing parameters for sequence padding.

        Args:
            max_num_seqs (int): The maximum number of sequences.
            max_num_prefill_seqs (int): The maximum number of prefill sequences.
            block_size (int): The size cache block.
            max_num_batched_tokens (int): The maximum number of batched tokens.
            max_model_len (int, optional): The maximum length of the model. This serves as the default value for max_prompt_seq and max_decode_seq. Defaults to None.
            max_prompt_seq (int, optional): The maximum length of the prompt sequence. Defaults to max_model_len. Must be less than or equal to max_model_len.
            max_decode_seq (int, optional): The maximum length of the decode sequence. Defaults to max_model_len. Must be less than or equal to max_model_len.
        """
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.use_merged_prefill = use_merged_prefill
        self.num_hpu_blocks = None
        self.max_model_len = max_model_len
        self.max_prompt_seq = max_prompt_seq
        self.max_decode_seq = max_decode_seq
        self._setup_buckets()
        self.generate_prompt_buckets()

    def _setup_buckets(self) -> None:
        # FIXME: The default values should be max_model_len

        default_max_prompt_seq = 1024
        default_max_decode_seq = 2048
        if self.max_model_len is None and self.max_prompt_seq is None:
            logger.warning(f"max_model_len and max_prompt_seq are not set. Using default value max_prompt_seq={default_max_prompt_seq}. This may cause issues.")
        if self.max_model_len is None and self.max_decode_seq is None:
            logger.warning(f"max_model_len and max_decode_seq are not set. Using default value max_decode_seq={default_max_decode_seq}. This may cause issues.")

        max_prompt_seq = next((item for item in [self.max_prompt_seq, self.max_model_len] if item is not None), default_max_prompt_seq)
        max_decode_seq = next((item for item in [self.max_decode_seq, self.max_model_len] if item is not None), default_max_decode_seq)
        max_blocks = max(
            self.block_size,
            self.max_num_seqs * max_decode_seq // self.block_size)

        self.global_state.prompt_bs_bucket_cfg = read_bucket_settings(
            'prompt', 'bs', min=1, step=32,
            max=self.max_num_prefill_seqs)
        self.global_state.decode_bs_bucket_cfg = read_bucket_settings(
            'decode', 'bs', min=1, step=32,
            max=self.max_num_seqs)
        self.global_state.prompt_seq_bucket_cfg = read_bucket_settings(
            'prompt', 'seq', min=self.block_size,
            step=self.block_size, max=max_prompt_seq)
        self.global_state.decode_block_bucket_cfg = read_bucket_settings(
            'decode', 'block', min=self.block_size,
            step=self.block_size, max=max_blocks)

        if self.use_merged_prefill:
            prev_prompt_bs_bucket_cfg = tuple(self.global_state.prompt_bs_bucket_cfg)
            prev_prompt_seq_bucket_cfg = tuple(self.global_state.prompt_seq_bucket_cfg)
            seq_min, seq_step, seq_max = prev_prompt_seq_bucket_cfg
            max_bs = self.global_state.prompt_bs_bucket_cfg[2]
            self.global_state.prompt_bs_bucket_cfg = (1, 1, 1)
            self.global_state.prompt_seq_bucket_cfg = (seq_min, seq_step, min(max_bs * seq_max, self.max_num_batched_tokens))
            new_prompt_bs_bucket_cfg = self.global_state.prompt_bs_bucket_cfg
            new_prompt_seq_bucket_cfg = self.global_state.prompt_seq_bucket_cfg
            print('Merged prefill is enabled!\n'
                  'Overriding prompt bucketing settings!\n'
                  f'prompt bs cfg: {prev_prompt_bs_bucket_cfg} -> {new_prompt_bs_bucket_cfg}\n'
                  f'prompt seq cfg: {prev_prompt_seq_bucket_cfg} -> {new_prompt_seq_bucket_cfg}\n')

        msg = ("Prompt bucket config (min, step, max_warmup) "
               f"bs:{self.global_state.prompt_bs_bucket_cfg}, "
               f"seq:{self.global_state.prompt_seq_bucket_cfg}")
        logger.info(msg)

        msg = ("Decode bucket config (min, step, max_warmup) "
               f"bs:{self.global_state.decode_bs_bucket_cfg}, "
               f"block:{self.global_state.decode_block_bucket_cfg}")
        logger.info(msg)

    def generate_prompt_buckets(self):
        self.global_state.prompt_buckets, prompt_omitted_buckets = \
            generate_prompt_buckets(
            self.global_state.prompt_bs_bucket_cfg,
            self.global_state.prompt_seq_bucket_cfg,
            self.max_num_batched_tokens)

        msg = (f"Generated {len(self.global_state.prompt_buckets)} "
               f"prompt buckets [bs, seq]: "
               f"{list(sorted(self.global_state.prompt_buckets))}")
        logger.info(msg)

        msg = (f"Omitted {len(prompt_omitted_buckets)} "
               "prompt buckets due to exceeded token budget "
               f"(max_num_batched_tokens={self.max_num_batched_tokens})")
        logger.info(msg)

        msg = f"Omitted prompt buckets: {list(sorted(prompt_omitted_buckets))}"
        logger.info(msg)

    def generate_decode_buckets(self, max_blocks):
        self.global_state.decode_buckets = generate_decode_buckets(
            self.global_state.decode_bs_bucket_cfg,
            self.global_state.decode_block_bucket_cfg, max_blocks)
        logger.info(f"Generated {len(self.global_state.decode_buckets)} "
              f"decode buckets [bs, total_blocks]: "
              f"{list(sorted(self.global_state.decode_buckets))}")

    def get_max_prompt_shape(self):
        return (self.global_state.prompt_bs_bucket_cfg[-1],
                self.global_state.prompt_seq_bucket_cfg[-1])

    def get_padded_prompt_batch_size(self, batch_size):
        return find_bucket(batch_size,
                           self.global_state.prompt_bs_bucket_cfg)

    def get_padded_decode_batch_size(self, batch_size):
        return find_bucket(batch_size,
                           self.global_state.decode_bs_bucket_cfg)

    def get_padded_prompt_seq_len(self, seq_len):
        return find_bucket(seq_len,
                           self.global_state.prompt_seq_bucket_cfg)

    def get_padded_decode_num_blocks(self, num_blocks):
        assert self.num_hpu_blocks is not None, "num_hpu_blocks is not set"
        bucket_size = find_bucket(num_blocks,
                                  self.global_state.decode_block_bucket_cfg)
        return min(bucket_size, self.num_hpu_blocks)

    def get_padded_batch_size(self, batch_size, is_prompt):
        if is_prompt:
            return self.get_padded_prompt_batch_size(batch_size)
        return self.get_padded_decode_batch_size(batch_size)

    def get_padded_seq_or_block(self, seq_or_block, is_prompt):
        if is_prompt:
            return self.get_padded_prompt_seq_len(seq_or_block)
        return self.get_padded_decode_num_blocks(seq_or_block)

    @property
    def prompt_buckets(self):
        return self.global_state.prompt_buckets

    @property
    def decode_buckets(self):
        return self.global_state.decode_buckets

    @classmethod
    def get_instance(cls):
        """
        Retrieve the singleton instance of the class.

        Returns:
            The singleton instance of the class.

        Raises:
            AssertionError: If the class has not been initialized and no instance exists.
        """
        assert cls in cls._instances, "Singleton instance not initialized"
        return type(cls)._instances[cls]

def read_bucket_settings(phase: str, dim: str, **defaults):
    """Read bucketing configuration from env variables.

    phase is either 'prompt' or 'decode'
    dim is either 'bs', 'seq' or 'block'
    param is either 'min', 'step' or 'max'
    example env variable: APHRODITE_DECODE_BS_BUCKET_STEP=128
    """
    params = ['min', 'step', 'max']
    env_vars = [f'APHRODITE_{phase}_{dim}_BUCKET_{p}'.upper() for p in params]
    default_values = [defaults[p] for p in params]
    values = [
        int(os.environ.get(e, d)) for e, d in zip(env_vars, default_values)
    ]
    for e, v, d in zip(env_vars, values, default_values):
        logger.info(f'{e}={v} (default:{d})')
    return values


def warmup_range(config: Tuple[int, int, int]):
    """Generate a warmup range.

    Start from bmin and multiply by 2 until you reach bstep.
    Then, increase the values in the range by the value of bstep until you 
    reach bmax.

    Example:
    bmin = 2, bstep = 32, bmax = 64
    => ramp_up = (2, 4, 8, 16)
    => stable = (32, 64)
    => return ramp_up + stable => (2, 4, 8, 16, 32, 64)
    """
    bmin, bstep, bmax = config
    assert bmin <= bmax, ("Min. batch size cannot be greater than max. "
                          "batch size. If you want to skip warmup, "
                          "set APHRODITE_SKIP_WARMUP=true")
    base = itertools.repeat(2)
    ramp_up_acc = itertools.accumulate(base, func=operator.mul, initial=bmin)
    ramp_up_tw = itertools.takewhile(lambda x: x < bstep and x <= bmax, \
        ramp_up_acc)
    stable = range(bstep, bmax + 1, bstep)
    buckets = list(ramp_up_tw) + list(stable)
    return list(filter(lambda bucket: bucket >= bmin, buckets))


def generate_prompt_buckets(bs_bucket_config,
                            seq_bucket_config,
                            max_num_batched_tokens=None):
    buckets = list(
        itertools.product(warmup_range(bs_bucket_config),
                          warmup_range(seq_bucket_config)))
    if len(buckets) == 0:
        msg = ("No buckets could be captured with following config "
               f"(min, step, max_warmup): "
               f"bs:{bs_bucket_config}, "
               f"seq:{seq_bucket_config}")
        raise ValueError(msg)

    filtered_buckets = buckets
    if max_num_batched_tokens is not None:
        # Remove buckets exceeding batch token budget
        filtered_buckets = list(
            filter(
                lambda bucket: bucket[0] * bucket[1] <= max_num_batched_tokens,
                buckets))

        if len(filtered_buckets) == 0:
            # we can handle this if we ignore max_num_batched_tokens
            min_bucket_bs, min_bucket_seq = min(buckets,
                                                key=lambda b: (b[0] * b[1]))
            min_reqd_budget = min_bucket_bs * min_bucket_seq
            msg = (
                "The current bucketing configuration "
                f"(min, step, max_warmup): "
                f"bs:{bs_bucket_config}, "
                f"seq:{seq_bucket_config} cannot be used with specified "
                f"max_num_batched_tokens ({max_num_batched_tokens}), as the "
                f"smallest bucket ({min_reqd_budget}) would exceed token "
                "budget. Please increase max_num_batched_tokens or decrease "
                "bucket minimum. Ignoring max_num_batched_tokens at risk of "
                "out-of-memory errors.")
            logger.info(msg)
            return list(
                sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0]))), []

    captured_buckets = list(
        sorted(filtered_buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))
    omitted_buckets = list(
        sorted([x for x in buckets if x not in filtered_buckets]))
    return captured_buckets, omitted_buckets


def generate_decode_buckets(bs_bucket_config, blocks_bucket_config,
                            max_blocks):
    buckets = []
    bs_buckets = warmup_range(bs_bucket_config)
    block_buckets = warmup_range(blocks_bucket_config)
    last_bucket = max_blocks
    for bs in bs_buckets:
        for blocks in block_buckets:
            if blocks >= last_bucket:
                buckets.append((bs, last_bucket))
                break
            buckets.append((bs, blocks))
    return list(sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))


def next_pow2(value: int, base: int) -> int:
    res = base
    while value > res:
        res *= 2
    return res


def round_up(value: int, k: int) -> int:
    return (value + k - 1) // k * k


def find_bucket(value: int, config: Tuple[int, int, int]) -> int:
    bmin, bstep, _ = config
    if value <= bmin:
        return bmin
    else:
        next_step = round_up(value, bstep)
        next_pow = next_pow2(value, bmin)
        return next_pow if next_pow <= bstep else next_step
