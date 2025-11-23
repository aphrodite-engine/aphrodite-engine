from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from itertools import chain
from typing import Any

import prometheus_client
import torch
from torch import nn

import aphrodite.envs as envs
from aphrodite.security.weight_proofs import ChallengeVector

try:
    from aphrodite.modeling.layers.linear import ColumnParallelLinear, LinearBase, RowParallelLinear
except Exception:  # pragma: no cover - during worker bootstrap minimal imports may fail temporarily
    ColumnParallelLinear = RowParallelLinear = LinearBase = None  # type: ignore[misc]

_EXEC_COUNTER = prometheus_client.Counter(
    "aphrodite_weightproof_execute_total",
    "Count of weight execution requests serviced",
    ["model", "status"],
)


@dataclass
class WorkerExecutionPayload:
    layer: str
    parameter: str
    input: ChallengeVector

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkerExecutionPayload:
        vector = ChallengeVector.model_validate(payload["input"])
        parameter = payload.get("parameter") or "weight"
        return cls(layer=payload["layer"], parameter=parameter, input=vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "parameter": self.parameter,
            "input": self.input.model_dump(),
        }


@dataclass
class WorkerExecutionResult:
    status: str
    shard_type: str | None = None
    tp_rank: int | None = None
    tp_size: int | None = None
    partition_index: int | None = None
    vector: list[float] | None = None
    detail: str | None = None
    dp_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "shard_type": self.shard_type,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "partition_index": self.partition_index,
            "vector": self.vector,
            "detail": self.detail,
            "dp_rank": self.dp_rank,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkerExecutionResult:
        return cls(
            status=payload.get("status", "error"),
            shard_type=payload.get("shard_type"),
            tp_rank=payload.get("tp_rank"),
            tp_size=payload.get("tp_size"),
            partition_index=payload.get("partition_index"),
            vector=payload.get("vector"),
            detail=payload.get("detail"),
            dp_rank=payload.get("dp_rank"),
        )


def _flatten_layer_path(layer: str, parameter: str) -> tuple[str, str]:
    if layer.endswith((".weight", ".bias")):
        module_path, param_name = layer.rsplit(".", 1)
        return module_path, param_name
    return layer, parameter


def _vector_to_tensor(vector: ChallengeVector, device: torch.device | None) -> torch.Tensor:
    tensor = torch.tensor(vector.data, dtype=torch.float32, device=device)
    if vector.shape:
        tensor = tensor.view(*vector.shape)
    return tensor.reshape(-1)


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    if tensor.dim() != 1:
        tensor = tensor.reshape(-1)
    return [float(v) for v in tensor.detach().cpu().to(torch.float32)]


def run_weight_execution_on_model(
    model: nn.Module,
    payload: WorkerExecutionPayload,
    *,
    dp_rank: int,
) -> WorkerExecutionResult:
    module_path, parameter_name = _flatten_layer_path(payload.layer, payload.parameter)
    try:
        module = model.get_submodule(module_path)
    except AttributeError:
        return WorkerExecutionResult(status="missing", detail=f"Module {module_path} unavailable", dp_rank=dp_rank)

    try:
        param = dict(module.named_parameters(recurse=False))[parameter_name]
    except KeyError:
        return WorkerExecutionResult(
            status="missing",
            detail=f"Parameter {parameter_name} unavailable in {module_path}",
            dp_rank=dp_rank,
        )

    weight = param.detach()
    if weight is None:
        return WorkerExecutionResult(status="missing", detail="Parameter has no data", dp_rank=dp_rank)

    device = weight.device
    input_tensor = _vector_to_tensor(payload.input, device)
    weight_tensor = weight.to(torch.float32)

    try:
        if ColumnParallelLinear is not None and isinstance(module, ColumnParallelLinear):
            output = torch.matmul(weight_tensor, input_tensor)
            return WorkerExecutionResult(
                status="ok",
                shard_type="column",
                tp_rank=getattr(module, "tp_rank", 0),
                tp_size=getattr(module, "tp_size", 1),
                partition_index=getattr(module, "tp_rank", 0),
                vector=_tensor_to_list(output),
                dp_rank=dp_rank,
            )

        if RowParallelLinear is not None and isinstance(module, RowParallelLinear):
            chunk = getattr(module, "input_size_per_partition", input_tensor.numel())
            tp_rank = getattr(module, "tp_rank", 0)
            tp_size = getattr(module, "tp_size", 1)
            start = tp_rank * chunk
            end = start + chunk
            if end > input_tensor.numel():
                return WorkerExecutionResult(
                    status="error",
                    detail=f"Input vector shorter than expected for rank {tp_rank}",
                    dp_rank=dp_rank,
                )
            local_input = input_tensor[start:end]
            output = torch.matmul(weight_tensor, local_input)
            return WorkerExecutionResult(
                status="ok",
                shard_type="row",
                tp_rank=tp_rank,
                tp_size=tp_size,
                vector=_tensor_to_list(output),
                dp_rank=dp_rank,
            )

        if isinstance(module, nn.Linear) or LinearBase is None or isinstance(module, LinearBase):
            output = torch.matmul(weight_tensor, input_tensor)
            return WorkerExecutionResult(
                status="ok",
                shard_type="replicated",
                tp_rank=0,
                tp_size=1,
                vector=_tensor_to_list(output),
                dp_rank=dp_rank,
            )

        return WorkerExecutionResult(
            status="unsupported",
            detail=f"Unsupported module type {module.__class__.__name__}",
            dp_rank=dp_rank,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return WorkerExecutionResult(status="error", detail=str(exc), dp_rank=dp_rank)


class WeightExecutionError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class WeightChallengeExecutor:
    def __init__(self, app_state):
        self._app_state = app_state

    async def execute(self, request) -> tuple[list[float], list[dict[str, Any]]]:
        engine_client = self._resolve_engine(request.model)
        payload = WorkerExecutionPayload(
            layer=request.layer,
            parameter=getattr(request, "parameter", None) or "weight",
            input=request.input,
        )
        worker_results = await engine_client.collective_rpc("execute_weight_challenge", args=(payload.to_dict(),))
        response_vector, shards = self._reduce_results(worker_results, payload.layer, request.model)
        _EXEC_COUNTER.labels(model=request.model, status="success").inc()
        return response_vector, shards

    def _resolve_engine(self, model_name: str):
        if envs.APHRODITE_ENABLE_MULTI_MODEL:
            registry = getattr(self._app_state, "model_registry", None)
            if not registry or model_name not in registry:
                raise WeightExecutionError(HTTPStatus.NOT_FOUND, f"Model '{model_name}' is not loaded")
            return registry[model_name].engine_client

        engine_client = getattr(self._app_state, "engine_client", None)
        if engine_client is None:
            raise WeightExecutionError(HTTPStatus.SERVICE_UNAVAILABLE, "Engine client not initialized")
        return engine_client

    def _reduce_results(
        self,
        results: Sequence[Any],
        layer: str,
        model_key: str,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        parsed = [WorkerExecutionResult.from_dict(r) for r in results if isinstance(r, dict)]
        if not parsed:
            _EXEC_COUNTER.labels(model=model_key, status="missing").inc()
            raise WeightExecutionError(HTTPStatus.NOT_FOUND, f"No workers responded for layer {layer}")

        dp_groups: dict[int, list[WorkerExecutionResult]] = {}
        for res in parsed:
            dp_rank = res.dp_rank or 0
            dp_groups.setdefault(dp_rank, []).append(res)

        selected_dp = min(dp_groups.keys())
        active = dp_groups[selected_dp]

        ranked_ok = [res for res in active if res.status == "ok"]
        if not ranked_ok:
            first_error = next((res for res in active if res.status == "error"), None)
            if first_error:
                _EXEC_COUNTER.labels(model=model_key, status="error").inc()
                raise WeightExecutionError(HTTPStatus.INTERNAL_SERVER_ERROR, first_error.detail or "Execution failed")
            first_unsupported = next((res for res in active if res.status == "unsupported"), None)
            if first_unsupported:
                _EXEC_COUNTER.labels(model=model_key, status="unsupported").inc()
                raise WeightExecutionError(
                    HTTPStatus.NOT_IMPLEMENTED,
                    first_unsupported.detail or f"Unsupported layer {layer}",
                )
            _EXEC_COUNTER.labels(model=model_key, status="missing").inc()
            raise WeightExecutionError(HTTPStatus.NOT_FOUND, f"Layer {layer} not present on any worker")

        return self.aggregate_vectors(ranked_ok)

    @staticmethod
    def aggregate_vectors(results: Sequence[WorkerExecutionResult]) -> tuple[list[float], list[dict[str, Any]]]:
        column_parts: dict[int, list[float]] = {}
        row_contribs: dict[int, list[float]] = {}
        replicated: list[float] | None = None

        for res in results:
            if res.shard_type == "column" and res.partition_index is not None and res.vector:
                column_parts.setdefault(res.partition_index, res.vector)
            elif res.shard_type == "row" and res.tp_rank is not None and res.vector:
                row_contribs.setdefault(res.tp_rank, res.vector)
            elif res.shard_type == "replicated" and res.vector and replicated is None:
                replicated = res.vector

        shards_meta: list[dict[str, Any]] = []

        if column_parts:
            ordered = [column_parts[idx] for idx in sorted(column_parts.keys())]
            vector = list(chain.from_iterable(ordered))
            for idx, part in sorted(column_parts.items()):
                shards_meta.append({"shard_type": "column", "partition_index": idx, "length": len(part)})
            return vector, shards_meta

        if row_contribs:
            accum: torch.Tensor | None = None
            for vec in row_contribs.values():
                tensor = torch.tensor(vec, dtype=torch.float32)
                accum = tensor if accum is None else accum + tensor
            assert accum is not None
            for rank, vec in sorted(row_contribs.items()):
                shards_meta.append({"shard_type": "row", "tp_rank": rank, "length": len(vec)})
            return _tensor_to_list(accum), shards_meta

        if replicated:
            shards_meta.append({"shard_type": "replicated", "length": len(replicated)})
            return replicated, shards_meta

        raise WeightExecutionError(HTTPStatus.INTERNAL_SERVER_ERROR, "No usable shard contributions found")
