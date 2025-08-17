from typing import Any, Callable, Protocol

import torch.fx as fx

from aphrodite.config import AphroditeConfig
from aphrodite.compilation.backends import AphroditeBackend


class AbstractPiecewiseBackend(Protocol):
    """
    PiecewiseBackend interface that allows platforms to extend 
    piecewise static graph.
    """

    def __init__(self, graph: fx.GraphModule,
                 aphrodite_config: AphroditeConfig,
                 graph_pool: Any, piecewise_compile_index: int,
                 total_piecewise_compiles: int, sym_shape_indices: list[int],
                 compiled_graph_for_general_shape: Callable,
                 aphrodite_backend: AphroditeBackend, **kwargs):
        """
        Initializes the PiecewiseBackend class with compilation and 
        execution-related configurations.

        This class handles piecewise compilation, graph capturing, 
        and dispatching for specific input shapes.

        Args:
            graph (fx.GraphModule): The graph represented in fx.
            aphrodite_config (AphroditeConfig): Global configuration for
            Aphrodite.
            graph_pool (Any): 
                Graph memory pool handle, e.g., 
                    `torch.cuda.graph_pool_handle()`.
            piecewise_compile_index (int): 
                Index of the current piecewise subgraph.
            total_piecewise_compiles (int): 
                Total number of piecewise-compiled graphs.
            sym_shape_indices (list[int]): 
                Indices of symbolic shape.
            compiled_graph_for_general_shape (Callable): 
                Callable that executes the graph compiled for general shapes.
            aphrodite_backend (AphroditeBackend): 
                Backend compiler that manages compilation and graph runtime 
                for Aphrodite.

        Keyword Args:
            kwargs: Additional keyword arguments reserved for future 
                extensions or custom platforms.
        """
        raise NotImplementedError

    def __call__(self, *args) -> Any:
        """Executes the compiled graph for given input args.

        If this is the first invocation, executes the general compiled graph
        and initiates the compilation process tracking. For subsequent calls,
        dynamically dispatches execution to either a compiled graph or a static
        graph based on the input shape.

        Args:
            *args: Variable length input arguments to be passed into the 
                graph. The symbolic shape is expected to be in position 
                `sym_shape_indices[0]`.

        Returns:
            Any: Output of the executed graph. This can be from the general
            compiled graph, a specialized compiled version for the given shape,
            or a replayed static graph.
        """
        raise NotImplementedError
