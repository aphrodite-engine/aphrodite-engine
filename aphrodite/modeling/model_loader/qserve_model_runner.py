import math
from typing import Dict, Optional, Union

import numpy as np
from loguru import logger
from QEfficient.generation.cloud_infer import QAICInferenceSession


def init_qaic_session(
    qpc_path: str, device_id: Union[str, int]
) -> QAICInferenceSession:
    # instantiate QPC QAIC Inference Session
    # TODO: wrap in try-except block once you analyze the possible errors this
    # can produce
    if isinstance(device_id, int):
        device_id = [device_id]
    session = QAICInferenceSession(qpc_path, device_ids=device_id)

    # Skip inputs/outputs
    session.skip_buffers(
        set([x for x in session.input_names if x.startswith("past_")])
    )
    session.skip_buffers(
        set([x for x in session.output_names if x.endswith("_RetainedState")])
    )
    return session

class QServeModelRunner:
    """model runner tasked with generating tokens from a given input"""

    def __init__(
        self,
        qpc_path: str,
        vocab_size: int,
        device_id: list,
        seq_len: Optional[int] = None,
        ctx_len: Optional[int] = None,
        decode_bsz: Optional[int] = None,
        prefill_bsz: int = 1,
    ) -> None:
        """initialize ModelRunner

        Args:
            qpc_path (str): path to qpc
            vocab_size (int): vocab size
            device_id (str, int): path to mdp json file or int QID
            seq_len (int): prompt length
            ctx_len (int): context length
            decode_bsz (int): decode batch size
            prefill_bsz (int): prefill batch size
        """
        self.qpc_path: str = qpc_path
        self.vocab_size: int = vocab_size
        self.device_id: list = device_id
        self.seq_len: Optional[int] = seq_len
        self.ctx_len: Optional[int] = ctx_len
        self.decode_bsz: Optional[int] = decode_bsz
        self.prefill_bsz: Optional[int] = prefill_bsz
        logger.info("Loading QPC...")
        self.session = init_qaic_session(qpc_path, device_id)
        logger.info("Successfully loaded QPC")
        #self.validate_input_arguments()
        self.attention_mask = None
        self.prefill_batch_inputs = None
        self.decode_batch_inputs = None
        self.decode_single_inputs = None
        self.prefill_logits = dict(logits=np.random.randn(
            self.prefill_bsz, 1, self.vocab_size).astype(np.float32))
        self.decode_logits = dict(logits=np.random.randn(
            self.decode_bsz, 1, self.vocab_size).astype(np.float32))
        try:
            self.dummy_run()
        except ValueError:
            logger.info("Re-runing with different logit dimension...")
            self.prefill_logits = dict(logits=np.random.randn(
                self.prefill_bsz, self.vocab_size).astype(np.float32))
            self.decode_logits = dict(logits=np.random.randn(
                self.decode_bsz, self.vocab_size).astype(np.float32))
            self.dummy_run()

    def get_qpc_IO_dims(self):

        bindings = self.session.get_bindings_shapes(["input_ids",'past_key.0'])
        decode_bsz, seq_len = np.max(bindings['input_ids'], axis=0)
        prefill_bsz, decode_id_sz = np.min(bindings['input_ids'], axis=0)

        # Only prefill batch size 1 is supported
        if prefill_bsz!=1 or decode_id_sz!=1:
            raise ValueError(
                    self.device_id,
                    message=("QPC not compiled for either decode or has "
                             "prefill bsz>1!!"),
            )
        ctx_len = bindings["past_key.0"][0][2]

        return dict(
            prefill_bsz=prefill_bsz,
            seq_len=seq_len,
            decode_bsz=decode_bsz,
            ctx_len=ctx_len,
        )

    def validate_input_arguments(self, class_object=None):
        if class_object is None:
            class_object = self

        arg_dims: dict = self.get_qpc_IO_dims()
        for arg, dim in arg_dims.items():
            instance_val: Optional[int] = getattr(class_object, arg)
            if instance_val is None:
                setattr(class_object, arg, dim)
            elif instance_val != dim:
                raise ValueError(
                    self.device_id,
                    message=(f"arg {arg}={instance_val} does not match "
                             f"corresponding qpc value of {dim}"),
                )

    def dummy_run(self):
        """assert prefill and decode work by running dummy inputs

        also creates attention_mask and decode input buffers
        that will be used throughout the life of qserve
        """

        # prepare dummy run inputs

        # prefill inputs
        #attention_mask = np.zeros((1, self.ctx_len))
        #attention_mask[0, self.seq_len - 1] = True
        prefill_inputs = dict(
            input_ids=np.zeros((self.prefill_bsz, self.seq_len),
                               dtype=np.int64),
            #position_ids=np.tile(
            #    np.zeros((self.seq_len),
            #             dtype=np.int64).reshape(1,self.seq_len),
            #    (self.prefill_bsz, 1),
            #),
            position_ids = np.tile(np.full((self.seq_len), -1,
                                           dtype=np.int64).reshape(
                                               1,self.seq_len),
                                               (self.prefill_bsz, 1)),
            batch_index=np.arange(self.prefill_bsz).reshape(-1, 1),
        )

        prefill_qpc_inputs = {0: prefill_inputs}

        # decode inputs
        decode_single_inputs = dict(
            input_ids=np.array([[0]]),
            position_ids=np.array([[0]]),
            batch_index=np.array([[0]]),
        )
        decode_batch_inputs = dict(
            input_ids=np.zeros((self.decode_bsz, 1), dtype=np.int64),
            position_ids = np.full((self.decode_bsz,1), -1, dtype=np.int64),
            #position_ids=np.zeros((self.decode_bsz, 1), dtype=np.int64),
            batch_index=np.arange(self.decode_bsz,
                                  dtype=np.int64).reshape(-1, 1),
        )
        decode_qpc_inputs = {0: decode_single_inputs}
        self.decode_single_inputs = decode_single_inputs
        self.decode_batch_inputs = decode_batch_inputs
        self.prefill_batch_inputs = prefill_inputs.copy()

        # run dummy inputs
        logger.debug("starting dummy run...")
        _: Dict[int, np.ndarray] = self.run(prefill_qpc_inputs, True)  # prefill
        _: Dict[int, np.ndarray] = self.run(decode_qpc_inputs, False)  # decode
        logger.debug("finished dummy run")

    def run(self, qpc_inputs: Dict[int, dict], is_prompt: bool) -> np.ndarray:
        """run qpc_inputs

        Args:
            qpc_inputs (Dict[int, dict]): qpc inputs of incoming requests to
                process
            is_prompt (bool): whether this is a prefill or decode run

        Returns:
            np.ndarray: fixed slot generated tokens
        """

        if is_prompt:
            next_token_ids: np.ndarray = self._run_prefill(qpc_inputs)
        else:
            next_token_ids: np.ndarray = self._run_decode(qpc_inputs)

        return next_token_ids

    def _run_prefill(
        self,
        qpc_inputs: Dict[int, dict],
    ) -> np.ndarray:
        """run qpc prefill inputs

        Args:
            qpc_inputs (Dict[int, dict]): qpc inputs of incoming requests to
                process

        Returns:
            np.ndarray: fixed slot generated tokens
        """

        # set qpc prefill state
        self.session.set_buffers(self.prefill_logits)
        # perform prefill (only prefill_bsz=1 is supported)
        bidx, inputs = next(iter(qpc_inputs.items()))
        n_prompt_tokens = inputs["input_ids"].shape[-1]  # n_prompt_tokens >= self.seq_len  # noqa: E501
        n_chunks: int = math.ceil(n_prompt_tokens / self.seq_len)
        assert n_chunks > 0
        for chunk in range(n_chunks):
            if chunk+1 == n_chunks:
                lower_idx = -self.seq_len
                upper_idx = n_prompt_tokens
            else:
                lower_idx = int(chunk * self.seq_len)
                upper_idx = int((chunk + 1) * self.seq_len)
            input_ids: np.ndarray = inputs["input_ids"][:, lower_idx:upper_idx]
            position_ids: np.ndarray = inputs["position_ids"][:,
                                                              lower_idx:upper_idx]
            batch_index: np.ndarray = inputs["batch_index"]
            chunk_inputs = dict(
                input_ids=input_ids,
                position_ids=position_ids,
                batch_index=batch_index,
            )
            outputs: dict = self.session.run(chunk_inputs)

        logits = outputs["logits"]

        return logits.squeeze(1)

    def _run_decode(self, qpc_inputs: Dict[int, dict]) -> np.ndarray:
        """run qpc decode inputs

        Args:
            qpc_inputs (Dict[int, dict]): qpc inputs of incoming requests to
                process

        Returns:
            np.ndarray: fixed slot generated tokens
        """
        # set qpc sesstion state to decode phase
        self.session.set_buffers(self.decode_logits)
        # fill buffer with valid bidx entries
        for bidx in qpc_inputs:
            for input in ("input_ids", "position_ids"):
                self.decode_batch_inputs[input][bidx] = qpc_inputs[bidx][input]
        # run decode step
        outputs: dict = self.session.run(self.decode_batch_inputs)
        logits: np.ndarray = outputs["logits"]
        indices = list(qpc_inputs.keys())

        return logits[indices].squeeze(1)
