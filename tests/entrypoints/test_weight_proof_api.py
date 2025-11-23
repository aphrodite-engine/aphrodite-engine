import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aphrodite.endpoints.openai import api_server
from aphrodite.security import weight_proofs
from aphrodite.security.weight_executor import WeightExecutionError
from aphrodite.security.weight_verifier import WeightVerificationService


@pytest.fixture(autouse=True)
def clear_weight_proof_cache():
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


def _build_app(
    service: WeightVerificationService | None,
    model_name: str,
    executor=None,
) -> TestClient:
    app = FastAPI()
    app.state.model_registry = {model_name: object()}
    app.state.weight_verification_service = service
    app.state.weight_execution_executor = executor
    app.include_router(api_server.weight_router)
    return TestClient(app)


def test_weight_challenge_and_verify_flow(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)
    client = _build_app(service, model)

    challenge = client.post("/weights/challenge", json={"model": model, "worker_id": "worker-a"})
    assert challenge.status_code == 200
    payload = challenge.json()
    assert payload["layer"]["layer"] == "decoder.layers.0.linear"

    verify = client.post(
        "/weights/verify",
        json={
            "challenge_id": payload["challenge_id"],
            "worker_id": "worker-a",
            "result": [3.0, 5.0],
        },
    )
    assert verify.status_code == 200
    assert verify.json()["outcome"] == "passed"


def test_challenge_rejects_unknown_model(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)
    client = _build_app(service, model)

    response = client.post("/weights/challenge", json={"model": "unknown/model", "worker_id": "worker-a"})
    assert response.status_code == 404


def test_verify_reports_failure_details(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)
    client = _build_app(service, model)

    challenge = client.post("/weights/challenge", json={"model": model, "worker_id": "worker-a"})
    payload = challenge.json()

    verify = client.post(
        "/weights/verify",
        json={
            "challenge_id": payload["challenge_id"],
            "worker_id": "worker-a",
            "result": [0.0, 0.0],
        },
    )
    assert verify.status_code == 200
    body = verify.json()
    assert body["outcome"] == "failed"
    assert body["failure_streak"] == 1
    assert body["blocked"] is False


def test_routes_return_not_found_when_service_disabled(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    app = FastAPI()
    app.state.model_registry = {model: object()}
    app.state.weight_verification_service = None
    app.include_router(api_server.weight_router)
    client = TestClient(app)

    response = client.post("/weights/challenge", json={"model": model, "worker_id": "worker-a"})
    assert response.status_code == 404


class DummyExecutor:
    def __init__(self, vector=None, shards=None):
        self.vector = vector or [0.5, -0.5]
        self.shards = shards or [{"shard_type": "column", "length": len(self.vector), "partition_index": 0}]
        self.last_request = None

    async def execute(self, request):
        self.last_request = request
        return self.vector, self.shards


class FailingExecutor:
    async def execute(self, request):
        raise WeightExecutionError(409, "boom")


def test_execute_weight_returns_vector(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)
    executor = DummyExecutor(vector=[0.1, 0.2], shards=[{"shard_type": "column", "length": 1, "partition_index": 0}])
    client = _build_app(service, model, executor=executor)

    response = client.post(
        "/weights/execute",
        json={
            "model": model,
            "worker_id": "worker-a",
            "layer": "model.layers.0.self_attn.k_proj",
            "input": {"data": [0.5, -0.5], "dtype": "float32", "shape": [2]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output"]["data"] == [0.1, 0.2]
    assert executor.last_request.layer == "model.layers.0.self_attn.k_proj"


def test_execute_weight_handles_executor_error(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)
    executor = FailingExecutor()
    client = _build_app(service, model, executor=executor)

    response = client.post(
        "/weights/execute",
        json={
            "model": model,
            "worker_id": "worker-a",
            "layer": "model.layers.0.self_attn.k_proj",
            "input": {"data": [0.5, -0.5], "dtype": "float32", "shape": [2]},
        },
    )
    assert response.status_code == 409


def test_execute_weight_missing_executor(tmp_path):
    model = "Qwen/Qwen3-0.6B"
    _write_bundle(tmp_path, model)
    service = WeightVerificationService(ttl_seconds=30, asset_root=tmp_path)
    client = _build_app(service, model, executor=None)

    response = client.post(
        "/weights/execute",
        json={
            "model": model,
            "worker_id": "worker-a",
            "layer": "model.layers.0.self_attn.k_proj",
            "input": {"data": [0.5, -0.5], "dtype": "float32", "shape": [2]},
        },
    )
    assert response.status_code == 404
