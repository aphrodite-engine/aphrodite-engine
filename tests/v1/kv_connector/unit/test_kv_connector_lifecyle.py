from aphrodite.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (  # noqa: E501
    SharedStorageConnectorMetadata,
)
from aphrodite.distributed.kv_transfer.kv_transfer_state import ensure_kv_transfer_initialized, get_kv_transfer_group
from aphrodite.v1.core.sched.output import CachedRequestData, SchedulerOutput
from aphrodite.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

# Importing utils registers TestSharedStorageConnector with the factory
from .utils import create_aphrodite_config


def _make_empty_scheduler_output():
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=SharedStorageConnectorMetadata(),
    )


def test_kv_connector_mixin_clears_metadata():
    aphrodite_config = create_aphrodite_config()
    aphrodite_config.kv_transfer_config.kv_connector = "TestSharedStorageConnector"
    aphrodite_config.kv_transfer_config.kv_role = "kv_both"
    aphrodite_config.kv_transfer_config.kv_connector_extra_config["name"] = "unit"

    # Initialize the global connector instance
    ensure_kv_transfer_initialized(aphrodite_config)

    try:
        # Minimal scheduler output with empty metadata; mixin should still
        # bind/clear metadata even if no loads happen
        scheduler_output = _make_empty_scheduler_output()

        # Invoke the no-forward path which uses the mixin context manager
        KVConnectorModelRunnerMixin.kv_connector_no_forward(scheduler_output, aphrodite_config)

        # Verify clear_connector_metadata was called on the connector
        connector = get_kv_transfer_group()
        assert connector._connector_metadata is None
        # Test connector wrapper records method calls
        assert connector.call_record.get("bind_connector_metadata", 0) == 1
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        # Ensure we clean up the global connector between tests
        KVConnectorModelRunnerMixin.ensure_kv_transfer_shutdown()
