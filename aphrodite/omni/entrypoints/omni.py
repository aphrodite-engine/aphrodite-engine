# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from tqdm.auto import tqdm

from aphrodite.logger import init_logger
from aphrodite.omni.engine.messages import OutputMessage
from aphrodite.omni.entrypoints.client_request_state import ClientRequestState
from aphrodite.omni.entrypoints.omni_base import OmniBase
from aphrodite.omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from aphrodite.omni.outputs import OmniRequestOutput
from aphrodite.sampling_params import RequestOutputKind, SamplingParams

if TYPE_CHECKING:
    from aphrodite.omni.inputs.data import OmniPromptType, OmniSamplingParams

logger = init_logger(__name__)


class Omni(OmniBase):
    """Synchronous entrypoint for offline generation."""

    def check_health(self) -> None:
        self._check_health()

    def start_profile(self, profile_prefix: str | None = None, stages: list[int] | None = None) -> list[Any]:
        return self._start_profile(profile_prefix, stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        return self._stop_profile(stages)

    def _maybe_force_final_only_for_llm_stages(
        self,
        sampling_params_list: Sequence[OmniSamplingParams],
    ) -> list[OmniSamplingParams]:
        """Return per-stage params with LLM stages forced to FINAL_ONLY.

        The caller may explicitly request ``output_kind = DELTA`` on a stage to
        opt into streaming; such stages are left alone.  All other LLM stages
        are forced to FINAL_ONLY.
        """
        effective_params: list[OmniSamplingParams] = []
        for stage_id, params in enumerate(sampling_params_list):
            sp = copy.deepcopy(params)
            stage_meta = self.engine.get_stage_metadata(stage_id)
            if (
                stage_meta.stage_type != "diffusion"
                and hasattr(sp, "output_kind")
                and sp.output_kind != RequestOutputKind.DELTA
            ):
                sp.output_kind = RequestOutputKind.FINAL_ONLY
            effective_params.append(sp)
        return effective_params

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[True],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]: ...

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[False] = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[OmniRequestOutput]: ...

    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: bool = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None] | list[OmniRequestOutput]:
        # Expand sampling params for PD disaggregation (user may provide N-1 params)
        if (
            sampling_params_list is not None
            and isinstance(sampling_params_list, Sequence)
            and not isinstance(sampling_params_list, (str, bytes))
        ):
            sampling_params_list = self._maybe_expand_sampling_params(list(sampling_params_list))
        sampling_params_list = self.resolve_sampling_params_list(sampling_params_list)
        try:
            if py_generator:
                return self._run_generation_with_generator(prompts, sampling_params_list, use_tqdm)
            return list(self._run_generation(prompts, sampling_params_list, use_tqdm))
        except Exception as e:
            logger.exception("[Omni] Failed to run generation: %s", e)
            self.close()
            raise

    def _run_generation_with_generator(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        gen = self._run_generation(prompts, sampling_params_list, use_tqdm)
        try:
            yield from gen
        finally:
            self.close()

    def _run_generation(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        try:
            sampling_params_list = self._maybe_force_final_only_for_llm_stages(sampling_params_list)

            if isinstance(prompts, str) or not isinstance(prompts, Sequence):
                request_prompts: list[OmniPromptType] = [prompts]
            elif isinstance(prompts, list) and prompts and isinstance(prompts[0], int):
                if not all(isinstance(token, int) for token in prompts):
                    raise ValueError("Token prompts must contain only integers")
                request_prompts = [cast(list[int], prompts)]
            else:
                request_prompts = []
                for prompt in prompts:
                    if isinstance(prompt, int):
                        raise ValueError("A prompt batch cannot mix token IDs with prompts")
                    request_prompts.append(prompt)

            if not request_prompts:
                return

            request_ids = [f"{i}_{uuid.uuid4()}" for i in range(len(request_prompts))]
            req_start_ts: dict[str, float] = {}
            wall_start_ts = time.time()
            req_final_stage_ids: dict[str, int] = {}

            for req_id, prompt in zip(request_ids, request_prompts):
                prompt_modalities = prompt.get("modalities", None) if isinstance(prompt, dict) else None
                if prompt_modalities is not None and (
                    not isinstance(prompt_modalities, list) or not all(isinstance(m, str) for m in prompt_modalities)
                ):
                    raise ValueError("Prompt modalities must be a list of strings")
                final_stage_id = self._compute_final_stage_id(prompt_modalities)
                final_output_stage_ids = self._compute_final_output_stage_ids(prompt_modalities) or [final_stage_id]
                req_final_stage_ids[req_id] = final_stage_id

                metrics = OrchestratorMetrics(
                    self.num_stages,
                    self.log_stats,
                    wall_start_ts,
                    final_stage_id,
                )
                req_state = ClientRequestState(req_id)
                req_state.metrics = metrics
                self.request_states[req_id] = req_state

                # PD disaggregation: modify stage-0 (prefill) sampling params per request
                req_sp_list = list(sampling_params_list)
                pd_pair = self._get_pd_separation_pair()
                if pd_pair is not None:
                    p_id = pd_pair[0]
                    prefill_params = req_sp_list[p_id]
                    if not isinstance(prefill_params, SamplingParams):
                        raise TypeError("Prefill stage requires SamplingParams")
                    req_sp_list[p_id] = self._prepare_prefill_sampling_params(req_id, prefill_params)

                self.engine.add_request(
                    request_id=req_id,
                    prompt=prompt,
                    sampling_params_list=req_sp_list,
                    final_stage_id=final_stage_id,
                    final_output_stage_ids=final_output_stage_ids,
                )
                submit_ts = time.time()
                req_state.metrics.stage_first_ts[0] = submit_ts
                req_start_ts[req_id] = submit_ts

            active_reqs = set(request_ids)
            pbar = None
            if use_tqdm:
                tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
                pbar = tqdm_func(total=len(request_ids), desc="Processed prompts", dynamic_ncols=True)

            while active_reqs:
                msg = self.engine.try_get_output()

                should_continue, output_req_id, stage_id, output_req_state = self._handle_output_message(msg)
                if should_continue:
                    continue

                assert isinstance(msg, OutputMessage)
                assert output_req_id is not None and stage_id is not None and output_req_state is not None
                req_id = output_req_id
                req_state = output_req_state

                if req_id not in active_reqs:
                    logger.warning("[Omni] Received output for unknown/finished request_id=%s", req_id)
                    continue

                self._check_engine_output_error(msg, req_id, stage_id)

                if req_state.metrics is None:
                    continue
                output_to_yield = self._process_single_result(
                    result=msg,
                    stage_id=stage_id,
                    metrics=req_state.metrics,
                    req_start_ts=req_start_ts,
                    wall_start_ts=wall_start_ts,
                    final_stage_id_for_e2e=req_final_stage_ids[req_id],
                )
                if output_to_yield is not None:
                    yield output_to_yield

                if isinstance(msg, OutputMessage) and msg.finished:
                    active_reqs.discard(req_id)
                    if pbar is not None:
                        pbar.update(1)
                    self._log_summary_and_cleanup(req_id)
        except Exception:
            if "active_reqs" in locals() and active_reqs:
                for req_id in active_reqs:
                    self._record_request_failure_once(req_id, reason="stage_error")
                self.abort(list(active_reqs))
            raise
        finally:
            if "pbar" in locals() and pbar is not None:
                pbar.close()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        self.engine.abort(request_ids)
        for req_id in request_ids:
            self._record_request_failure_once(req_id, reason="client_abort")
            self.request_states.pop(req_id, None)
        if self.log_stats:
            logger.info("[Omni] Aborted request(s) %s", ",".join(request_ids))
