import torch
import pytest

from aphrodite.common.sampling_params import SamplerID
from aphrodite.sampling_params import SamplingParams
from aphrodite.v1.sample.logits_processor import LogitsProcessors
from aphrodite.v1.sample.metadata import SamplingMetadata
from aphrodite.v1.sample.ops.dry import apply_all_dry
from aphrodite.v1.sample.ops.epsilon_cutoff import epsilon_cutoff
from aphrodite.v1.sample.ops.eta_cutoff import eta_cutoff
from aphrodite.v1.sample.ops.mirostat import mirostat
from aphrodite.v1.sample.ops.no_repeat_ngram import no_repeat_ngram
from aphrodite.v1.sample.ops.quadratic import quadratic
from aphrodite.v1.sample.ops.tfs import tfs
from aphrodite.v1.sample.ops.top_a import top_a
from aphrodite.v1.sample.ops.top_nsigma import top_nsigma
from aphrodite.v1.sample.ops.typical_p import typical_p
from aphrodite.v1.sample.ops.xtc import xtc
from aphrodite.v1.sample.sampler import Sampler
from aphrodite.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def _metadata(**overrides) -> SamplingMetadata:
    data = dict(
        temperature=torch.tensor([1.0], dtype=torch.float32),
        dynatemp_min=None,
        dynatemp_max=None,
        dynatemp_exp=None,
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        top_a=None,
        dry_multiplier=None,
        dry_base=None,
        dry_allowed_length=None,
        dry_sequence_breaker_ids=None,
        dry_ranges=None,
        dry_max_ngram=None,
        dry_max_occurrences=None,
        dry_early_exit_match_len=None,
        no_repeat_ngram_size=None,
        tfs=None,
        eta_cutoff=None,
        epsilon_cutoff=None,
        typical_p=None,
        quadratic_smoothing_factor=None,
        quadratic_smoothing_curve=None,
        xtc_threshold=None,
        xtc_probability=None,
        top_nsigma=None,
        mirostat_mode=None,
        mirostat_tau=None,
        mirostat_eta=None,
        skew=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        frequency_penalties=torch.tensor([0.0], dtype=torch.float32),
        presence_penalties=torch.tensor([0.0], dtype=torch.float32),
        repetition_penalties=torch.tensor([1.0], dtype=torch.float32),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logit_bias={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        sampler_priority=None,
        temperature_last=False,
        persistent_data={},
        spec_token_ids=[[]],
    )
    data.update(overrides)
    return SamplingMetadata(**data)


def test_top_a_masks_tokens_below_threshold():
    logits = torch.tensor([[2.0, 1.0, 0.0, -2.0]], dtype=torch.float32)
    metadata = _metadata(top_a=torch.tensor([0.5], dtype=torch.float32))

    result = top_a(logits.clone(), metadata)

    assert torch.isfinite(result[0, 0])
    assert torch.isfinite(result[0, 1])
    assert torch.isneginf(result[0, 2])
    assert torch.isneginf(result[0, 3])


def test_tfs_preserves_best_token_and_masks_tail():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    metadata = _metadata(tfs=torch.tensor([0.1], dtype=torch.float32))

    result = tfs(logits.clone(), metadata)

    assert torch.isfinite(result[0, 0])
    assert torch.isneginf(result[0]).any()


def test_eta_cutoff_keeps_top_token_when_cutoff_would_remove_everything():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    metadata = _metadata(eta_cutoff=torch.tensor([0.95], dtype=torch.float32))

    result = eta_cutoff(logits.clone(), metadata)

    assert torch.isfinite(result[0, 0])
    assert torch.isneginf(result[0, 1:]).all()


def test_epsilon_cutoff_keeps_top_token_when_cutoff_would_remove_everything():
    logits = torch.zeros((1, 4), dtype=torch.float32)
    metadata = _metadata(epsilon_cutoff=torch.tensor([0.95], dtype=torch.float32))

    result = epsilon_cutoff(logits.clone(), metadata)

    assert torch.isfinite(result[0, 0])
    assert torch.isneginf(result[0, 1:]).all()


def test_typical_p_keeps_at_least_one_token_and_trims_tail():
    logits = torch.tensor([[8.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    metadata = _metadata(typical_p=torch.tensor([0.2], dtype=torch.float32))

    result = typical_p(logits.clone(), metadata)

    assert torch.isfinite(result[0]).any()
    assert torch.isneginf(result[0]).any()
    assert torch.isfinite(result[0, 0])


def test_quadratic_keeps_peak_logit_and_penalizes_lower_logits():
    logits = torch.tensor([[5.0, 4.0, 3.0]], dtype=torch.float32)
    metadata = _metadata(
        quadratic_smoothing_factor=torch.tensor([1.0], dtype=torch.float32),
        quadratic_smoothing_curve=torch.tensor([2.0], dtype=torch.float32),
    )

    result = quadratic(logits.clone(), metadata)

    assert result[0, 0].item() == logits[0, 0].item()
    assert result[0, 1].item() == logits[0, 1].item()
    assert result[0, 2].item() < logits[0, 2].item()


def test_xtc_masks_top_choices_above_threshold():
    torch.manual_seed(0)
    logits = torch.tensor([[4.0, 3.5, 3.0, 0.0]], dtype=torch.float32)
    metadata = _metadata(
        xtc_threshold=torch.tensor([0.1], dtype=torch.float32),
        xtc_probability=torch.tensor([1.0], dtype=torch.float32),
    )

    result = xtc(logits.clone(), metadata)

    assert torch.isneginf(result[0, 0])
    assert torch.isneginf(result[0, 1])
    assert torch.isfinite(result[0, 2])


def test_top_nsigma_masks_values_far_below_the_max():
    logits = torch.tensor([[10.0, 9.0, 0.0, -1.0]], dtype=torch.float32)
    metadata = _metadata(top_nsigma=torch.tensor([0.1], dtype=torch.float32))

    result = top_nsigma(logits.clone(), metadata)

    assert torch.isfinite(result[0, 0])
    assert torch.isneginf(result[0, 1:]).all()


def test_mirostat_returns_one_hot_logits_and_updates_mu():
    torch.manual_seed(0)
    logits = torch.tensor([[4.0, 3.0, 2.0]], dtype=torch.float32)
    metadata = _metadata(
        mirostat_mode=torch.tensor([2], dtype=torch.int32),
        mirostat_tau=torch.tensor([5.0], dtype=torch.float32),
        mirostat_eta=torch.tensor([0.1], dtype=torch.float32),
        persistent_data={},
    )

    result = mirostat(logits.clone(), metadata)

    assert (result[0] == 1.0).sum().item() == 1
    assert torch.isneginf(result[0]).sum().item() == 2
    assert 0 in metadata.persistent_data
    assert "miro_mu" in metadata.persistent_data[0]
    assert metadata.persistent_data[0]["miro_mu"] != 10.0


def test_mirostat_updates_persistent_mu_across_steps():
    torch.manual_seed(0)
    metadata = _metadata(
        mirostat_mode=torch.tensor([2], dtype=torch.int32),
        mirostat_tau=torch.tensor([5.0], dtype=torch.float32),
        mirostat_eta=torch.tensor([0.2], dtype=torch.float32),
        persistent_data={},
    )

    first_logits = torch.tensor([[4.0, 3.0, 2.0]], dtype=torch.float32)
    second_logits = torch.tensor([[2.5, 2.0, 1.0]], dtype=torch.float32)

    mirostat(first_logits.clone(), metadata)
    first_mu = metadata.persistent_data[0]["miro_mu"]
    mirostat(second_logits.clone(), metadata)
    second_mu = metadata.persistent_data[0]["miro_mu"]

    assert first_mu != 10.0
    assert second_mu != first_mu


def test_no_repeat_ngram_masks_repeated_continuation():
    logits = torch.zeros((1, 10), dtype=torch.float32)
    metadata = _metadata(
        prompt_token_ids=None,
        output_token_ids=[[2, 5, 2]],
        no_repeat_ngram_size=torch.tensor([2], dtype=torch.int32),
    )

    result = no_repeat_ngram(logits.clone(), metadata)

    assert torch.isneginf(result[0, 5])


def test_no_repeat_ngram_uses_prompt_plus_output_history():
    logits = torch.zeros((1, 10), dtype=torch.float32)
    metadata = _metadata(
        prompt_token_ids=torch.tensor([[2, 5, 10]], dtype=torch.int64),
        output_token_ids=[[2]],
        no_repeat_ngram_size=torch.tensor([2], dtype=torch.int32),
    )

    result = no_repeat_ngram(logits.clone(), metadata)

    assert torch.isneginf(result[0, 5])


def test_no_repeat_ngram_handles_empty_output_history():
    logits = torch.zeros((1, 10), dtype=torch.float32)
    metadata = _metadata(
        prompt_token_ids=torch.tensor([[2, 5, 2]], dtype=torch.int64),
        output_token_ids=[],
        no_repeat_ngram_size=torch.tensor([2], dtype=torch.int32),
    )

    result = no_repeat_ngram(logits.clone(), metadata)

    assert torch.isneginf(result[0, 5])


def test_no_repeat_ngram_ignores_padded_prompt_tokens():
    logits = torch.zeros((1, 10), dtype=torch.float32)
    metadata = _metadata(
        prompt_token_ids=torch.tensor([[2, 5, 2, 10, 10]], dtype=torch.int64),
        output_token_ids=[],
        no_repeat_ngram_size=torch.tensor([2], dtype=torch.int32),
    )

    result = no_repeat_ngram(logits.clone(), metadata)

    assert torch.isneginf(result[0, 5])


def test_dry_penalizes_repeated_continuation():
    vocab_size = 16
    logits = torch.zeros((1, vocab_size), dtype=torch.float32)
    prompt_token_ids = torch.tensor([[1, 2, 1, 2]], dtype=torch.int64)
    output_token_ids = torch.full((1, 1), vocab_size, dtype=torch.int64)

    result = apply_all_dry(
        logits.clone(),
        prompt_token_ids,
        output_token_ids,
        torch.tensor([1.25], dtype=torch.float32),
        torch.tensor([1.75], dtype=torch.float32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[15]], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([20], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int32),
        torch.tensor([20], dtype=torch.int32),
    )

    assert result[0, 1].item() < logits[0, 1].item()


def test_dry_does_nothing_when_last_token_is_a_breaker():
    vocab_size = 16
    logits = torch.zeros((1, vocab_size), dtype=torch.float32)
    prompt_token_ids = torch.tensor([[1, 2, 1, 2]], dtype=torch.int64)
    output_token_ids = torch.full((1, 1), vocab_size, dtype=torch.int64)

    result = apply_all_dry(
        logits.clone(),
        prompt_token_ids,
        output_token_ids,
        torch.tensor([1.25], dtype=torch.float32),
        torch.tensor([1.75], dtype=torch.float32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[2]], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([20], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int32),
        torch.tensor([20], dtype=torch.int32),
    )

    assert torch.equal(result, logits)


def test_dry_respects_range_limit():
    vocab_size = 16
    logits = torch.zeros((1, vocab_size), dtype=torch.float32)
    prompt_token_ids = torch.tensor([[1, 2, 1, 2]], dtype=torch.int64)
    output_token_ids = torch.full((1, 1), vocab_size, dtype=torch.int64)

    result = apply_all_dry(
        logits.clone(),
        prompt_token_ids,
        output_token_ids,
        torch.tensor([1.25], dtype=torch.float32),
        torch.tensor([1.75], dtype=torch.float32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[15]], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([20], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int32),
        torch.tensor([20], dtype=torch.int32),
    )

    assert torch.equal(result, logits)


def test_xtc_probability_zero_is_noop():
    torch.manual_seed(0)
    logits = torch.tensor([[4.0, 3.5, 3.0, 0.0]], dtype=torch.float32)
    metadata = _metadata(
        xtc_threshold=torch.tensor([0.1], dtype=torch.float32),
        xtc_probability=torch.tensor([0.0], dtype=torch.float32),
    )

    result = xtc(logits.clone(), metadata)

    assert torch.equal(result, logits)


def test_temperature_last_moves_temperature_to_the_end(monkeypatch):
    sampler = Sampler()
    call_order = []

    monkeypatch.setattr(
        sampler.sampling_ops,
        "apply_top_a",
        lambda logits, metadata: call_order.append("top_a") or logits,
    )
    monkeypatch.setattr(
        sampler.sampling_ops,
        "apply_tfs",
        lambda logits, metadata: call_order.append("tfs") or logits,
    )
    monkeypatch.setattr(
        sampler,
        "apply_temperature",
        lambda logits, metadata: call_order.append("temperature") or logits,
    )

    metadata = _metadata(
        top_a=torch.tensor([0.5], dtype=torch.float32),
        tfs=torch.tensor([0.5], dtype=torch.float32),
        temperature_last=True,
    )

    sampler._execute_samplers_in_order(torch.zeros((1, 4), dtype=torch.float32), metadata)

    assert call_order == ["top_a", "tfs", "temperature"]


def test_custom_sampler_priority_is_respected(monkeypatch):
    sampler = Sampler()
    call_order = []

    monkeypatch.setattr(
        sampler.sampling_ops,
        "apply_top_a",
        lambda logits, metadata: call_order.append("top_a") or logits,
    )
    monkeypatch.setattr(
        sampler.sampling_ops,
        "apply_tfs",
        lambda logits, metadata: call_order.append("tfs") or logits,
    )
    monkeypatch.setattr(
        sampler,
        "apply_temperature",
        lambda logits, metadata: call_order.append("temperature") or logits,
    )

    metadata = _metadata(
        top_a=torch.tensor([0.5], dtype=torch.float32),
        tfs=torch.tensor([0.5], dtype=torch.float32),
        sampler_priority=[SamplerID.TFS, SamplerID.TOP_A, SamplerID.TEMPERATURE],
    )

    sampler._execute_samplers_in_order(torch.zeros((1, 4), dtype=torch.float32), metadata)

    assert call_order == ["tfs", "top_a", "temperature"]


def test_mirostat_short_circuits_other_samplers(monkeypatch):
    sampler = Sampler()
    call_order = []

    monkeypatch.setattr(
        sampler.sampling_ops,
        "apply_mirostat",
        lambda logits, metadata: call_order.append("mirostat") or logits,
    )
    monkeypatch.setattr(
        sampler.sampling_ops,
        "apply_top_a",
        lambda logits, metadata: call_order.append("top_a") or logits,
    )

    metadata = _metadata(
        top_a=torch.tensor([0.5], dtype=torch.float32),
        mirostat_mode=torch.tensor([2], dtype=torch.int32),
        mirostat_tau=torch.tensor([5.0], dtype=torch.float32),
        mirostat_eta=torch.tensor([0.1], dtype=torch.float32),
    )

    sampler._execute_samplers_in_order(torch.zeros((1, 4), dtype=torch.float32), metadata)

    assert call_order == ["mirostat"]


def test_skew_is_applied_before_sampling(monkeypatch):
    sampler = Sampler()
    observed = {}

    def fake_topk_topp(logits, generators, top_k, top_p):
        observed["logits"] = logits.clone()
        return torch.argmax(logits, dim=-1), None

    monkeypatch.setattr(sampler.topk_topp_sampler, "forward", fake_topk_topp)

    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    metadata = _metadata(skew=torch.tensor([1.0], dtype=torch.float32))

    sampler.sample(logits.clone(), metadata)

    assert "logits" in observed
    assert not torch.allclose(observed["logits"], logits)


def test_input_batch_preserves_custom_sampler_metadata():
    sampling_params = SamplingParams(
        temperature=0.7,
        top_a=0.25,
        tfs=0.4,
        eta_cutoff=0.0002,
        epsilon_cutoff=0.0003,
        typical_p=0.8,
        smoothing_factor=0.5,
        smoothing_curve=2.0,
        xtc_threshold=0.2,
        xtc_probability=0.6,
        nsigma=0.75,
        mirostat_mode=2,
        mirostat_tau=4.5,
        mirostat_eta=0.15,
        skew=0.3,
        dry_multiplier=1.2,
        dry_base=1.6,
        dry_allowed_length=2,
        dry_sequence_breaker_ids=[9, 10],
        dry_range=32,
        dry_max_ngram=12,
        dry_max_occurrences=4,
        dry_early_exit_match_len=8,
        no_repeat_ngram_size=3,
        sampler_priority=[SamplerID.TOP_A, SamplerID.TFS, SamplerID.TEMPERATURE],
        temperature_last=True,
    )
    request = CachedRequestState(
        req_id="req",
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=[],
        sampling_params=sampling_params,
        generator=None,
        block_ids=([],),
        num_computed_tokens=2,
        output_token_ids=[5, 6],
    )
    input_batch = InputBatch(
        max_num_reqs=1,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=128,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    input_batch.add_request(request)

    metadata = input_batch._make_sampling_metadata()

    assert metadata.top_a is not None and metadata.top_a[0].item() == pytest.approx(sampling_params.top_a)
    assert metadata.tfs is not None and metadata.tfs[0].item() == pytest.approx(sampling_params.tfs)
    assert metadata.eta_cutoff is not None and metadata.eta_cutoff[0].item() == pytest.approx(
        sampling_params.eta_cutoff
    )
    assert (
        metadata.epsilon_cutoff is not None
        and metadata.epsilon_cutoff[0].item() == pytest.approx(sampling_params.epsilon_cutoff)
    )
    assert metadata.typical_p is not None and metadata.typical_p[0].item() == pytest.approx(
        sampling_params.typical_p
    )
    assert (
        metadata.quadratic_smoothing_factor is not None
        and metadata.quadratic_smoothing_factor[0].item() == pytest.approx(sampling_params.smoothing_factor)
    )
    assert (
        metadata.quadratic_smoothing_curve is not None
        and metadata.quadratic_smoothing_curve[0].item() == pytest.approx(sampling_params.smoothing_curve)
    )
    assert metadata.xtc_threshold is not None and metadata.xtc_threshold[0].item() == pytest.approx(
        sampling_params.xtc_threshold
    )
    assert (
        metadata.xtc_probability is not None
        and metadata.xtc_probability[0].item() == pytest.approx(sampling_params.xtc_probability)
    )
    assert metadata.top_nsigma is not None and metadata.top_nsigma[0].item() == pytest.approx(
        sampling_params.nsigma
    )
    assert metadata.mirostat_mode is not None and metadata.mirostat_mode[0].item() == sampling_params.mirostat_mode
    assert metadata.mirostat_tau is not None and metadata.mirostat_tau[0].item() == pytest.approx(
        sampling_params.mirostat_tau
    )
    assert metadata.mirostat_eta is not None and metadata.mirostat_eta[0].item() == pytest.approx(
        sampling_params.mirostat_eta
    )
    assert metadata.skew is not None and metadata.skew[0].item() == pytest.approx(sampling_params.skew)
    assert metadata.dry_multiplier is not None and metadata.dry_multiplier[0].item() == pytest.approx(
        sampling_params.dry_multiplier
    )
    assert metadata.no_repeat_ngram_size is not None
    assert metadata.no_repeat_ngram_size[0].item() == sampling_params.no_repeat_ngram_size
    assert metadata.sampler_priority == [
        SamplerID.TOP_A,
        SamplerID.TFS,
        SamplerID.TEMPERATURE,
    ]
    assert metadata.temperature_last is True
    assert metadata.dry_sequence_breaker_ids is not None
    assert metadata.dry_sequence_breaker_ids[0, :2].tolist() == [9, 10]
    assert metadata.prompt_token_ids is not None
    assert metadata.output_token_ids == [[5, 6]]


def test_input_batch_keeps_token_history_for_no_repeat_ngram_only():
    sampling_params = SamplingParams(
        temperature=0.0,
        no_repeat_ngram_size=2,
    )
    request = CachedRequestState(
        req_id="req",
        prompt_token_ids=[2, 5, 2],
        mm_features=[],
        sampling_params=sampling_params,
        generator=None,
        block_ids=([],),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    input_batch = InputBatch(
        max_num_reqs=1,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=128,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    input_batch.add_request(request)

    metadata = input_batch._make_sampling_metadata()

    assert metadata.no_repeat_ngram_size is not None
    assert metadata.prompt_token_ids is not None
    assert metadata.output_token_ids == [[]]
