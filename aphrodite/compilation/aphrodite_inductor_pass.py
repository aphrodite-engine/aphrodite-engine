
import time

import torch
# yapf: enable
from loguru import logger

from aphrodite.common.config import CompilationConfig, AphroditeConfig
# yapf: disable
from aphrodite.distributed import get_tensor_model_parallel_rank as get_tp_rank
from aphrodite.distributed import (
    get_tensor_model_parallel_world_size as get_tp_world_size)
from aphrodite.distributed import model_parallel_is_initialized as p_is_init

from .inductor_pass import InductorPass


class AphroditeInductorPass(InductorPass):
    """
    An inductor pass with access to Aphrodite PassConfig.
    It provides timing, logging, and dumping utilities.
    """

    def __init__(self, config: AphroditeConfig):
        self.pass_config = config.compilation_config.pass_config
        self.dtype = config.model_config.dtype if config.model_config else None
        self.device = config.device_config.device if config.device_config \
            else None
        self.pass_name = self.__class__.__name__

    def dump_graph(self, graph: torch.fx.Graph, stage: str, always=False):
        if stage in self.pass_config.dump_graph_stages or always:
            # Make sure filename includes rank in the distributed setting
            parallel = p_is_init() and get_tp_world_size() > 1
            rank = f"-{get_tp_rank()}" if parallel else ""
            filepath = self.pass_config.dump_graph_dir / f"{stage}{rank}.py"

            logger.info("{} printing graph to {}", self.pass_name, filepath)
            with open(filepath, "w") as f:
                src = graph.python_code(root_module="self", verbose=True).src
                # Add imports so it's not full of errors
                print("import torch; from torch import device", file=f)
                print(src, file=f)

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("{} completed in %.1f ms", self.pass_name, duration_ms)


class PrinterInductorPass(AphroditeInductorPass):

    def __init__(self,
                 name: str,
                 config: CompilationConfig.PassConfig,
                 always=False):
        super().__init__(config)
        self.name = name
        self.always = always

    def __call__(self, graph: torch.fx.Graph):
        self.dump_graph(graph, self.name, always=self.always)
