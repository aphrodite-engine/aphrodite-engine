import pytest

from aphrodite.security.weight_executor import (
    WeightChallengeExecutor,
    WeightExecutionError,
    WorkerExecutionResult,
)


def test_aggregate_column_vectors():
    results = [
        WorkerExecutionResult(
            status="ok",
            shard_type="column",
            partition_index=0,
            vector=[0.1, 0.2],
        ),
        WorkerExecutionResult(
            status="ok",
            shard_type="column",
            partition_index=1,
            vector=[0.3],
        ),
    ]

    vector, shards = WeightChallengeExecutor.aggregate_vectors(results)
    assert vector == [0.1, 0.2, 0.3]
    assert shards == [
        {"shard_type": "column", "partition_index": 0, "length": 2},
        {"shard_type": "column", "partition_index": 1, "length": 1},
    ]


def test_aggregate_row_vectors_sums_partitions():
    results = [
        WorkerExecutionResult(
            status="ok",
            shard_type="row",
            tp_rank=0,
            vector=[0.1, 0.2],
        ),
        WorkerExecutionResult(
            status="ok",
            shard_type="row",
            tp_rank=1,
            vector=[0.3, -0.2],
        ),
    ]

    vector, shards = WeightChallengeExecutor.aggregate_vectors(results)
    assert vector == pytest.approx([0.4, 0.0])
    assert shards == [
        {"shard_type": "row", "tp_rank": 0, "length": 2},
        {"shard_type": "row", "tp_rank": 1, "length": 2},
    ]


def test_aggregate_replicated_returns_first():
    results = [
        WorkerExecutionResult(
            status="ok",
            shard_type="replicated",
            vector=[0.5],
        ),
        WorkerExecutionResult(
            status="ok",
            shard_type="replicated",
            vector=[1.0],
        ),
    ]

    vector, shards = WeightChallengeExecutor.aggregate_vectors(results)
    assert vector == [0.5]
    assert shards == [{"shard_type": "replicated", "length": 1}]


def test_aggregate_no_shards_raises():
    results = [
        WorkerExecutionResult(
            status="ok",
            shard_type="column",
            partition_index=None,
            vector=None,
        )
    ]

    try:
        WeightChallengeExecutor.aggregate_vectors(results)
    except WeightExecutionError as exc:
        assert exc.status_code == 500
    else:  # pragma: no cover
        raise AssertionError("Expected WeightExecutionError")
