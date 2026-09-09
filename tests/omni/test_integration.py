# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from functools import wraps
from types import SimpleNamespace
from unittest.mock import patch


def isolated(test):
    @wraps(test)
    def run():
        subprocess.run(
            [sys.executable, "-c", f"import runpy; runpy.run_path({__file__!r})[{test.__name__!r}].__wrapped__()"],
            check=True,
        )

    return run


def test_disabled_plugin_does_not_import_or_patch_omni():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
os.environ.pop('APHRODITE_OMNI_ENABLED', None)
from aphrodite.v1.engine import EngineCoreRequest
from aphrodite.plugins.omni import register
register()
import aphrodite.v1.engine as engine
assert engine.EngineCoreRequest is EngineCoreRequest
assert 'aphrodite.omni' not in sys.modules
""",
        ],
        check=True,
    )


@isolated
def test_video_task_registry_replacement_and_shutdown():
    import asyncio

    from aphrodite.omni.entrypoints.openai.stores import TaskRegistry

    async def check():
        registry = TaskRegistry()
        finished = asyncio.Event()
        cleaned = asyncio.Event()

        async def job():
            try:
                await finished.wait()
            finally:
                cleaned.set()

        old = asyncio.create_task(asyncio.sleep(0))
        await registry.upsert("video", old)
        replacement = asyncio.create_task(job())
        await registry.upsert("video", replacement)
        await old
        await asyncio.sleep(0)
        assert await registry.get("video") is replacement
        await registry.cancel_all()
        assert replacement.cancelled()
        assert cleaned.is_set()
        assert await registry.get("video") is None
        await registry.cancel_all()
        admissions = await asyncio.gather(*(registry.try_start(str(i), job, limit=2) for i in range(20)))
        assert sum(admissions) == 2
        await registry.cancel_all()
        assert await asyncio.gather(*(registry.get(str(i)) for i in range(20))) == [None] * 20

    asyncio.run(check())


@isolated
def test_video_storage_cancellation_waits_for_write():
    import asyncio
    import tempfile
    import threading

    from aphrodite.omni.entrypoints.openai.storage import LocalStorageManager

    async def check(directory):
        manager = LocalStorageManager(directory)
        started = threading.Event()
        release = threading.Event()
        save = manager._save_sync

        def delayed_save(data, name):
            started.set()
            assert release.wait(timeout=10)
            return save(data, name)

        manager._save_sync = delayed_save
        task = asyncio.create_task(manager.save(b"video", "job"))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        assert await manager.delete("job")
        assert await manager.open("job") is None

    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(check(directory))


@isolated
def test_video_upload_limits_and_cancelled_upload_cleanup():
    import asyncio
    import io
    import tempfile
    from pathlib import Path

    from fastapi import HTTPException, UploadFile

    from aphrodite.omni.entrypoints.openai import api_server

    async def check(directory):
        with patch.object(api_server.SERVER_SETTINGS_CONFIG, "max_reference_upload_bytes", 4):
            assert await api_server._read_upload_limited(UploadFile(io.BytesIO(b"1234"))) == b"1234"
            try:
                await api_server._read_upload_limited(UploadFile(io.BytesIO(b"12345")))
            except HTTPException as error:
                assert error.status_code == 400
            else:
                raise AssertionError("Oversized upload accepted")

            mkstemp = tempfile.mkstemp
            with patch.object(api_server.tempfile, "mkstemp", side_effect=lambda **kw: mkstemp(dir=directory, **kw)):
                try:
                    await api_server._persist_uploaded_video_references([UploadFile(io.BytesIO(b"12345"))])
                except HTTPException as error:
                    assert error.status_code == 413
                else:
                    raise AssertionError("Oversized video accepted")
                assert not list(Path(directory).iterdir())

                class InterruptedUpload:
                    filename = "test.mp4"

                    async def read(self, size):
                        raise asyncio.CancelledError()

                try:
                    await api_server._persist_uploaded_video_references([InterruptedUpload()])
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("Cancellation swallowed")
                assert not list(Path(directory).iterdir())

    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(check(directory))


@isolated
def test_omni_input_payload_survives_core_validation():
    from aphrodite.omni.engine.serialization import deserialize_additional_information
    from aphrodite.omni.inputs.processor import OmniInputProcessor
    from aphrodite.v1.engine.input_processor import InputProcessor

    validated = []

    def validate(self, request_id, prompt, params, supported_tasks, **kwargs):
        validated.append((prompt, kwargs))
        return SimpleNamespace(prompt_embeds=None)

    processor = object.__new__(OmniInputProcessor)
    prompt = {
        "type": "token",
        "prompt_token_ids": [1, 2],
        "additional_information": {"speaker": ["Ryan"]},
        "model_intermediate_buffer": {"stage": 0},
    }
    with patch.object(InputProcessor, "process_inputs", validate):
        request = processor.process_inputs("test", prompt, None, ("generate",), priority=3, session_id="session")
    assert deserialize_additional_information(request.additional_information) == {"speaker": ["Ryan"]}
    assert request.model_intermediate_buffer == {"stage": 0}
    assert validated == [(prompt, {"priority": 3, "session_id": "session"})]


@isolated
def test_encoder_decoder_adapter_preserves_renderer_policy():
    from aphrodite.omni.inputs.preprocess import OmniInputPreprocessor

    renderer = SimpleNamespace(
        default_cmpl_tok_params=SimpleNamespace(with_kwargs=lambda **kw: kw),
        _tokenize_singleton_prompt=lambda prompt, params: prompt,
        get_dec_start_token_id=lambda: 7,
        _get_skip_decoder_start_token=lambda: False,
    )
    processor = OmniInputPreprocessor(SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=True)), renderer)
    result = processor.preprocess({"encoder_prompt": {"prompt_token_ids": [1, 2]}, "decoder_prompt": None})
    from aphrodite.inputs import split_enc_dec_input

    encoder, decoder = split_enc_dec_input(result)
    assert encoder["prompt_token_ids"] == []
    assert decoder["prompt_token_ids"] == [7, 1, 2]


@isolated
def test_weight_loader_skips_stage_weights_after_mapping():
    import torch
    from torch import nn

    from aphrodite.model_executor.models.utils import WeightsMapper
    from aphrodite.omni.model_executor.weight_loader import AutoWeightsLoader

    model = nn.Linear(2, 2, bias=False)
    loader = AutoWeightsLoader(model, skip_prefixes=["decoder."], skip_substrs=["unused"])
    loaded = loader.load_weights(
        iter(
            [("model.weight", torch.ones(2, 2)), ("model.decoder.weight", torch.zeros(3)), ("unused", torch.zeros(1))]
        ),
        mapper=WeightsMapper(orig_to_new_prefix={"model.": ""}),
    )
    assert loaded == {"weight"}
    assert torch.equal(model.weight, torch.ones(2, 2))


@isolated
def test_codec_stop_id_filter_preserves_structured_output_recovery():
    import torch

    from aphrodite.omni.worker.sampling_utils import sanitize_min_tokens_stop_ids
    from aphrodite.v1.sample.logits_processor import MinTokensLogitsProcessor

    proc = MinTokensLogitsProcessor(None, torch.device("cpu"), False)
    proc.min_toks = {0: (2, [], {2, 99}, True), 1: (2, [], {1, 99}, False)}
    sanitize_min_tokens_stop_ids(SimpleNamespace(non_argmax_invariant=[proc]), 3)
    logits = torch.tensor([[float("-inf"), float("-inf"), 0.0], [0.0, 0.0, 0.0]])
    result = proc.apply(logits)
    assert result[0, 2] == 0
    assert torch.isneginf(result[1, 1])
    assert proc.min_toks[0][2] == {2}
    assert proc.min_toks[1][2] == {1}


@isolated
def test_diffusion_custom_op_namespace():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aphrodite.omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

    with FakeTensorMode():
        q = torch.empty(2, 4, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(128, device="cuda", dtype=q.dtype)
        rope = torch.empty(2, 96, device="cuda", dtype=q.dtype)
        outputs = fused_qk_norm_rope(q, q, weight, weight, rope, 1e-5)
        assert all(output.shape == q.shape for output in outputs)
