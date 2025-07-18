import asyncio
import os
from typing import Any, Callable, Optional, Union

import pytest

from aphrodite.engine.args_tools import AsyncEngineArgs, EngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.engine.aphrodite_engine import AphroditeEngine
from aphrodite.executor.uniproc_executor import UniProcExecutor
from aphrodite.common.sampling_params import SamplingParams


class Mock:
    ...


class CustomUniExecutor(UniProcExecutor):

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None) -> list[Any]:
        # Drop marker to show that this was ran
        with open(".marker", "w"):
            ...
        return super().collective_rpc(method, timeout, args, kwargs)


CustomUniExecutorAsync = CustomUniExecutor


@pytest.mark.parametrize("model", ["distilbert/distilgpt2"])
def test_custom_executor_type_checking(model):
    with pytest.raises(ValueError):
        engine_args = EngineArgs(model=model,
                                 distributed_executor_backend=Mock)
        AphroditeEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(model=model,
                                      distributed_executor_backend=Mock)
        AsyncAphrodite.from_engine_args(engine_args)


@pytest.mark.parametrize("model", ["distilbert/distilgpt2"])
def test_custom_executor(model, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=model,
            distributed_executor_backend=CustomUniExecutor,
            enforce_eager=True,  # reduce test time
        )
        engine = AphroditeEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("model", ["distilbert/distilgpt2"])
def test_custom_executor_async(model, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=model,
            distributed_executor_backend=CustomUniExecutorAsync,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncAphrodite.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = await engine.add_request("0", "foo", sampling_params)
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("model", ["distilbert/distilgpt2"])
def test_respect_ray(model):
    # even for TP=1 and PP=1,
    # if users specify ray, we should use ray.
    # users might do this if they want to manage the
    # resources using ray.
    engine_args = EngineArgs(
        model=model,
        distributed_executor_backend="ray",
        enforce_eager=True,  # reduce test time
    )
    engine = AphroditeEngine.from_engine_args(engine_args)
    assert engine.model_executor.uses_ray
