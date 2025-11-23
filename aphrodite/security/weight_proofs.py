from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, field_validator

from aphrodite import envs
from aphrodite.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_ASSET_ROOT = Path(__file__).resolve().parent.parent / "assets" / "weight_proofs"
_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def _dtype_from_name(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{name}' in weight proof")
    return _DTYPE_MAP[name]


class LayerAddress(BaseModel):
    """Unique identifier for the target weight tensor."""

    layer: str
    weight: str | None = None
    checksum: str | None = None


class ChallengeVector(BaseModel):
    """Serialized tensor payload."""

    data: list[float]
    dtype: str = "float32"
    shape: tuple[int, ...] | None = None

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return None
        if any(dim <= 0 for dim in value):
            raise ValueError("ChallengeVector shape must be positive")
        return value

    def numel(self) -> int:
        return len(self.data)

    def to_tensor(self) -> torch.Tensor:
        dtype = _dtype_from_name(self.dtype)
        tensor = torch.tensor(self.data, dtype=dtype)
        if self.shape:
            expected = math.prod(self.shape)
            if tensor.numel() != expected:
                raise ValueError(f"Vector size mismatch: {tensor.numel()} vs {expected}")
            tensor = tensor.view(*self.shape)
        return tensor


class WeightChallenge(BaseModel):
    """Single micro-task definition."""

    challenge_id: str = Field(alias="id")
    layer: LayerAddress
    input_vector: ChallengeVector = Field(alias="input")
    expected_output: ChallengeVector = Field(alias="output")
    rtol: float | None = None
    atol: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolved_rtol(self, override: float | None = None) -> float:
        if override is not None:
            return override
        if self.rtol is not None:
            return self.rtol
        return float(envs.APHRODITE_WEIGHT_PROOF_RTOL)

    def resolved_atol(self, override: float | None = None) -> float:
        if override is not None:
            return override
        if self.atol is not None:
            return self.atol
        return float(envs.APHRODITE_WEIGHT_PROOF_ATOL)

    def matches(
        self,
        candidate: torch.Tensor | Sequence[float],
        *,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> bool:
        expected = self.expected_output.to_tensor()
        candidate_tensor = _tensorize(candidate, expected.dtype, self.expected_output.shape)
        return torch.allclose(
            candidate_tensor,
            expected,
            rtol=self.resolved_rtol(rtol),
            atol=self.resolved_atol(atol),
            equal_nan=True,
        )


class WeightProofBundle(BaseModel):
    """Collection of challenges for a single model version."""

    model_key: str
    version: str | None = None
    challenges: list[WeightChallenge] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.challenges

    @classmethod
    def from_json(cls, model_key: str, payload: str) -> WeightProofBundle:
        raw = json.loads(payload)
        raw.setdefault("model_key", model_key)
        return cls.model_validate(raw)


def _tensorize(
    value: torch.Tensor | Sequence[float],
    dtype: torch.dtype,
    shape: tuple[int, ...] | None,
) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=dtype)
    tensor = tensor.to(dtype=dtype)
    if shape:
        expected = math.prod(shape)
        if tensor.numel() != expected:
            raise ValueError(f"Candidate size mismatch: {tensor.numel()} vs {expected}")
        tensor = tensor.view(*shape)
    return tensor


def _asset_root(override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    configured = envs.APHRODITE_WEIGHT_PROOF_ASSET_ROOT
    if configured:
        return Path(configured).expanduser().resolve()
    return _DEFAULT_ASSET_ROOT


_BUNDLE_CACHE: dict[str, WeightProofBundle] = {}


def bundle_path(model_key: str, *, assets_dir: str | Path | None = None) -> Path:
    sanitized = model_key.replace("/", "__")
    root = _asset_root(str(assets_dir) if assets_dir is not None else None)
    return root / f"{sanitized}.json"


def load_bundle(
    model_key: str, *, assets_dir: str | Path | None = None, reload: bool = False
) -> WeightProofBundle | None:
    cache_key = f"{model_key}:{assets_dir or ''}"
    if not reload and cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]

    path = bundle_path(model_key, assets_dir=assets_dir)
    if not path.exists():
        logger.debug("No weight proof bundle for %s at %s", model_key, path)
        return None

    payload = path.read_text()
    bundle = WeightProofBundle.from_json(model_key=model_key, payload=payload)
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle


def available_bundles(*, assets_dir: str | Path | None = None) -> list[str]:
    root = _asset_root(str(assets_dir) if assets_dir is not None else None)
    if not root.exists():
        return []
    return [path.stem for path in root.glob("*.json")]


def clear_cache() -> None:
    _BUNDLE_CACHE.clear()
