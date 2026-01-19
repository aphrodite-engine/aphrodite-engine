from transformers.configuration_utils import PretrainedConfig


class Glm4MoeLiteConfig(PretrainedConfig):
    model_type = "glm4_moe_lite"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size=154880,
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_hidden_layers=47,
        num_attention_heads=20,
        num_key_value_heads=20,
        n_shared_experts=1,
        n_routed_experts=64,
        routed_scaling_factor=1.8,
        kv_lora_rank=512,
        q_lora_rank=768,
        qk_rope_head_dim=64,
        v_head_dim=256,
        qk_nope_head_dim=192,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=202752,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=True,
        mlp_layer_types=None,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        index_topk=None,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)

        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        if index_topk is not None:
            self.index_topk = index_topk
        self.topk_method = topk_method
        self.scoring_func = scoring_func

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
