import pytest

from aphrodite.processing.scheduler import Scheduler
from aphrodite.engine.args_tools import EngineArgs
from aphrodite.engine.aphrodite_engine import AphroditeEngine
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.v1.core.sched.scheduler import Scheduler as V1Scheduler
from aphrodite.v1.engine.llm_engine import AphroditeEngine as V1LLMEngine


class DummyV0Scheduler(Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyV0Scheduler")


class DummyV1Scheduler(V1Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyV1Scheduler")


def test_scheduler_plugins_v0(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("APHRODITE_USE_V1", "0")
        with pytest.raises(Exception) as exception_info:

            engine_args = EngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,  # reduce test time
                scheduler_cls=DummyV0Scheduler,
            )

            engine = AphroditeEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

        assert str(
            exception_info.value) == "Exception raised by DummyV0Scheduler"


def test_scheduler_plugins_v1(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("APHRODITE_USE_V1", "1")
        # Explicitly turn off engine multiprocessing so
        # that the scheduler runs in this process
        m.setenv("APHRODITE_ENABLE_V1_MULTIPROCESSING", "0")

        with pytest.raises(Exception) as exception_info:

            engine_args = EngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,  # reduce test time
                scheduler_cls=DummyV1Scheduler,
            )

            engine = V1LLMEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

        assert str(
            exception_info.value) == "Exception raised by DummyV1Scheduler"
