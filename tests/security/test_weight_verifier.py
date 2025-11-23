import json
from datetime import timedelta

import pytest

from aphrodite.security import weight_proofs
from aphrodite.security.weight_verifier import VerificationOutcome, WeightVerificationService


@pytest.fixture(autouse=True)
def clear_cache():
    weight_proofs.clear_cache()
    yield
    weight_proofs.clear_cache()


def _write_bundle(tmp_path, model_name: str):
    sanitized = model_name.replace("/", "__")
    path = tmp_path / f"{sanitized}.json"
    bundle = {
        "version": "test",
        "challenges": [
            {
                "id": "c1",
                "layer": {"layer": "decoder.layers.0.linear", "checksum": "abc123"},
                "input": {"data": [1.0, 2.0], "dtype": "float32", "shape": [2]},
                "output": {"data": [3.0, 5.0], "dtype": "float32", "shape": [2]},
            }
        ],
    }
    path.write_text(json.dumps(bundle))
    return path


def test_issue_and_verify_success(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)

    issued = service.issue_challenge(model, "worker-a")
    assert issued is not None

    result = service.verify_response(
        challenge_id=issued.challenge_id,
        worker_id="worker-a",
        result_vector=[3.0, 5.0],
    )
    assert result.outcome == VerificationOutcome.PASSED


def test_issue_returns_none_when_bundle_missing(tmp_path):
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)
    assert service.issue_challenge("unknown/model", "worker") is None


def test_failures_increment_and_block(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, failure_threshold=2, asset_root=tmp_path)

    issued = service.issue_challenge(model, "worker-a")
    assert issued is not None
    result = service.verify_response(
        challenge_id=issued.challenge_id,
        worker_id="worker-a",
        result_vector=[0.0, 0.0],
    )
    assert result.outcome == VerificationOutcome.FAILED
    assert result.failure_streak == 1
    assert not result.blocked

    issued = service.issue_challenge(model, "worker-a")
    assert issued is not None
    result = service.verify_response(
        challenge_id=issued.challenge_id,
        worker_id="worker-a",
        result_vector=[0.0, 0.0],
    )
    assert result.outcome == VerificationOutcome.FAILED
    assert result.failure_streak == 2
    assert result.blocked


def test_expired_challenge_returns_expired(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=1, asset_root=tmp_path)

    issued = service.issue_challenge(model, "worker-a")
    assert issued is not None
    # Force expiry
    service._issued[issued.challenge_id].expires_at = issued.issued_at - timedelta(seconds=1)

    result = service.verify_response(
        challenge_id=issued.challenge_id,
        worker_id="worker-a",
        result_vector=[3.0, 5.0],
    )
    assert result.outcome == VerificationOutcome.EXPIRED


def test_unknown_challenge_returns_not_found(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)

    result = service.verify_response(
        challenge_id="missing",
        worker_id="worker-a",
        result_vector=[0.0],
    )
    assert result.outcome == VerificationOutcome.NOT_FOUND
