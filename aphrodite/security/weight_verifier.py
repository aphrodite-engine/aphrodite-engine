from __future__ import annotations

import secrets
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import prometheus_client
import torch

from aphrodite import envs
from aphrodite.logger import init_logger
from aphrodite.security.weight_proofs import WeightChallenge, WeightProofBundle, load_bundle

logger = init_logger(__name__)

_COUNTER_LABELS = ("model",)
_CHALLENGE_ISSUED = prometheus_client.Counter(
    "aphrodite_weightproof_challenges_issued_total",
    "Issued weight verification challenges",
    _COUNTER_LABELS,
)
_CHALLENGE_PASSED = prometheus_client.Counter(
    "aphrodite_weightproof_challenges_passed_total",
    "Passed weight verification challenges",
    _COUNTER_LABELS,
)
_CHALLENGE_FAILED = prometheus_client.Counter(
    "aphrodite_weightproof_challenges_failed_total",
    "Failed weight verification challenges",
    _COUNTER_LABELS,
)
_CHALLENGE_EXPIRED = prometheus_client.Counter(
    "aphrodite_weightproof_challenges_expired_total",
    "Expired weight verification challenges",
    _COUNTER_LABELS,
)


class VerificationOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"


@dataclass
class IssuedChallenge:
    challenge_id: str
    worker_id: str
    model_key: str
    challenge: WeightChallenge
    issued_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.expires_at


@dataclass
class VerificationResult:
    challenge_id: str
    worker_id: str
    outcome: VerificationOutcome
    message: str = ""
    failure_streak: int = 0
    blocked: bool = False

    @property
    def success(self) -> bool:
        return self.outcome == VerificationOutcome.PASSED


class WeightVerificationService:
    """Issues and validates weight micro-tasks."""

    def __init__(
        self,
        *,
        ttl_seconds: int | None = None,
        failure_threshold: int = 3,
        asset_root: str | Path | None = None,
    ):
        self._ttl = ttl_seconds or int(envs.APHRODITE_WEIGHT_PROOF_TASK_TTL_S)
        self._failure_threshold = max(1, failure_threshold)
        self._asset_root = Path(asset_root) if asset_root else None
        self._issued: MutableMapping[str, IssuedChallenge] = {}
        self._failure_streaks: defaultdict[str, int] = defaultdict(int)
        self._bundle_cache: dict[str, WeightProofBundle | None] = {}

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _load_bundle(self, model_key: str) -> WeightProofBundle | None:
        if model_key not in self._bundle_cache:
            bundle = load_bundle(model_key, assets_dir=self._asset_root)
            if bundle:
                logger.debug("Loaded %s weight challenges for %s", len(bundle.challenges), model_key)
            self._bundle_cache[model_key] = bundle
        return self._bundle_cache[model_key]

    def issue_challenge(self, model_key: str, worker_id: str) -> IssuedChallenge | None:
        bundle = self._load_bundle(model_key)
        if not bundle or bundle.is_empty():
            logger.debug("Weight proof bundle missing or empty for %s", model_key)
            return None

        challenge = secrets.choice(bundle.challenges)
        challenge_id = secrets.token_hex(16)
        issued_at = self._now()
        expires_at = issued_at + timedelta(seconds=self._ttl)
        issued = IssuedChallenge(
            challenge_id=challenge_id,
            worker_id=worker_id,
            model_key=model_key,
            challenge=challenge,
            issued_at=issued_at,
            expires_at=expires_at,
        )
        self._issued[challenge_id] = issued
        _CHALLENGE_ISSUED.labels(model=model_key).inc()
        return issued

    def verify_response(
        self,
        *,
        challenge_id: str,
        worker_id: str,
        result_vector: torch.Tensor | list[float],
    ) -> VerificationResult:
        issued = self._issued.pop(challenge_id, None)
        if issued is None:
            return VerificationResult(
                challenge_id=challenge_id,
                worker_id=worker_id,
                outcome=VerificationOutcome.NOT_FOUND,
                message="Unknown challenge id",
            )

        model_label = {"model": issued.model_key}
        now = self._now()

        if issued.worker_id != worker_id:
            self._failure_streaks[worker_id] += 1
            _CHALLENGE_FAILED.labels(**model_label).inc()
            return VerificationResult(
                challenge_id=challenge_id,
                worker_id=worker_id,
                outcome=VerificationOutcome.UNKNOWN,
                message="Worker mismatch",
                failure_streak=self._failure_streaks[worker_id],
                blocked=self._is_blocked(worker_id),
            )

        if issued.is_expired(now):
            self._failure_streaks[worker_id] += 1
            _CHALLENGE_EXPIRED.labels(**model_label).inc()
            return VerificationResult(
                challenge_id=challenge_id,
                worker_id=worker_id,
                outcome=VerificationOutcome.EXPIRED,
                message="Challenge expired",
                failure_streak=self._failure_streaks[worker_id],
                blocked=self._is_blocked(worker_id),
            )

        try:
            passed = issued.challenge.matches(result_vector)
            failure_reason = "Mismatch beyond tolerance"
        except ValueError as err:
            logger.debug("Challenge %s validation error: %s", challenge_id, err)
            passed = False
            failure_reason = str(err)
        if passed:
            self._failure_streaks.pop(worker_id, None)
            _CHALLENGE_PASSED.labels(**model_label).inc()
            return VerificationResult(
                challenge_id=challenge_id,
                worker_id=worker_id,
                outcome=VerificationOutcome.PASSED,
                message="Challenge satisfied",
            )

        streak = self._failure_streaks[worker_id] + 1
        self._failure_streaks[worker_id] = streak
        _CHALLENGE_FAILED.labels(**model_label).inc()
        return VerificationResult(
            challenge_id=challenge_id,
            worker_id=worker_id,
            outcome=VerificationOutcome.FAILED,
            message=failure_reason,
            failure_streak=streak,
            blocked=self._is_blocked(worker_id),
        )

    def purge_expired(self) -> int:
        now = self._now()
        expired = [key for key, issued in self._issued.items() if issued.is_expired(now)]
        for key in expired:
            issued = self._issued.pop(key)
            _CHALLENGE_EXPIRED.labels(model=issued.model_key).inc()
        return len(expired)

    def _is_blocked(self, worker_id: str) -> bool:
        return self._failure_streaks.get(worker_id, 0) >= self._failure_threshold

    def snapshot_active(self) -> Mapping[str, IssuedChallenge]:
        return dict(self._issued)
