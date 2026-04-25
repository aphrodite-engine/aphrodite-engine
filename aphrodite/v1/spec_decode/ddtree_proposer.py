# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import torch

from aphrodite.config import replace
from aphrodite.v1.sample.metadata import SamplingMetadata
from aphrodite.v1.spec_decode.ddtree import DDTreeRuntimeTree, build_ddtree_tree
from aphrodite.v1.spec_decode.dflash import DFlashProposer


class DDTreeProposer(DFlashProposer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_runtime_trees: list[DDTreeRuntimeTree | None] | None = None

    def _create_draft_aphrodite_config(self):
        draft_cfg = super()._create_draft_aphrodite_config()
        spec_cfg = draft_cfg.speculative_config
        assert spec_cfg is not None
        draft_spec_cfg = copy.copy(spec_cfg)
        draft_spec_cfg.method = "dflash"
        return replace(
            draft_cfg,
            attention_config=replace(
                draft_cfg.attention_config,
                backend=spec_cfg.draft_attention_backend,
            ),
            speculative_config=draft_spec_cfg,
        )

    def pop_runtime_trees(self) -> list[DDTreeRuntimeTree | None] | None:
        trees = self._last_runtime_trees
        self._last_runtime_trees = None
        return trees

    def _can_use_ddtree(self, sampling_metadata: SamplingMetadata, batch_size: int) -> bool:
        if batch_size != 1:
            return False
        if not sampling_metadata.all_greedy:
            return False
        if not sampling_metadata.no_penalties:
            return False
        if sampling_metadata.allowed_token_ids_mask is not None:
            return False
        if sampling_metadata.bad_words_token_ids:
            return False
        if sampling_metadata.logit_bias:
            return False
        return True

    def propose(self, *args, sampling_metadata: SamplingMetadata, **kwargs) -> torch.Tensor:
        common_attn_metadata = kwargs["common_attn_metadata"]
        batch_size = common_attn_metadata.batch_size()
        if not self._can_use_ddtree(sampling_metadata, batch_size):
            self._last_runtime_trees = None
            original_method = self.method
            self.method = "dflash"
            try:
                return super().propose(*args, sampling_metadata=sampling_metadata, **kwargs)
            finally:
                self.method = original_method

        target_token_ids = kwargs["target_token_ids"]
        target_positions = kwargs["target_positions"]
        target_hidden_states = kwargs["target_hidden_states"]
        next_token_ids = kwargs["next_token_ids"]
        token_indices_to_sample = kwargs["token_indices_to_sample"]
        mm_embed_inputs = kwargs.get("mm_embed_inputs")
        num_rejected_tokens_gpu = kwargs.get("num_rejected_tokens_gpu")
        slot_mappings = kwargs.get("slot_mappings")

        if hasattr(self.model, "combine_hidden_states"):
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)

        num_tokens, token_indices_to_sample, common_attn_metadata = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )

        per_group_attn_metadata, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata
        )
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = self._determine_batch_execution_and_padding(
            num_tokens
        )
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        # DDTree currently runs the drafter eagerly. Use the real query-token
        # extent rather than the padded cudagraph extent, otherwise the DFlash
        # query-position buffer (sized to the draft horizon) and the padded
        # input-id buffer diverge.
        actual_num_tokens = num_tokens
        model_kwargs["input_ids"] = model_kwargs["input_ids"][:actual_num_tokens]
        model_kwargs["positions"] = self._get_positions(actual_num_tokens)
        inputs_embeds = model_kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds[:actual_num_tokens]

        from aphrodite.forward_context import set_forward_context

        with set_forward_context(
            per_layer_attn_metadata,
            self.aphrodite_config,
            num_tokens=actual_num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(actual_num_tokens, common_attn_metadata.slot_mapping),
        ):
            # The DDTree drafter currently uses a different verifier-slot budget
            # than its draft horizon, and some draft models still carry shape
            # assumptions that break under Dynamo tracing. Bypass the inner
            # model's compile wrapper for this path so the corrected runtime
            # shapes can execute while the wider model path is ported to
            # compile-safe symbolic shapes.
            with torch.compiler.set_stance("force_eager"):
                inner_model = getattr(self.model, "model", None)
                if inner_model is not None:
                    ret_hidden_states = inner_model.forward(
                        model_kwargs["input_ids"],
                        model_kwargs["positions"],
                        model_kwargs.get("inputs_embeds"),
                    )
                else:
                    ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
            else:
                last_hidden_states = ret_hidden_states[0]

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        draft_logits = self.model.compute_logits(sample_hidden_states)
        budget = self.speculative_config.ddtree_tree_budget or self.num_speculative_tokens
        runtime_tree = build_ddtree_tree(draft_logits, budget)
        self._last_runtime_trees = [runtime_tree]
        return runtime_tree.node_token_ids.view(1, -1)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().dummy_run(
            min(num_tokens, self.max_query_tokens),
            use_cudagraphs=use_cudagraphs,
            is_graph_capturing=is_graph_capturing,
            slot_mappings=slot_mappings,
        )
