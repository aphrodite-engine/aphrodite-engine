import contextlib
import copy
import hashlib
import os
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import torch
import torch._inductor.compile_fx
import torch.fx as fx

import aphrodite.common.envs as envs
from aphrodite.common.config import AphroditeConfig
from aphrodite.common.utils import is_torch_equal_or_newer

from .inductor_pass import pass_context


class CompilerInterface:
    """
    The interface for a compiler that can be used by Aphrodite.
    """
    # The name of the compiler, e.g. inductor.
    # This is a class-level attribute.
    name: str

    def initialize_cache(self, cache_dir: str, disable_cache: bool = False):
        """
        when the Aphrodite process uses `cache_dir` as the cache directory,
        the compiler should initialize itself with the cache directory,
        e.g. by re-directing its own cache directory to a sub-directory.
        """
        pass

    def compute_hash(self, aphrodite_config: AphroditeConfig) -> str:
        """
        Gather all the relevant information from the Aphrodite config,
        to compute a hash so that we can cache the compiled model.

        See :meth:`AphroditeConfig.compute_hash` to check what information
        is already considered by default. This function should only
        consider the information that is specific to the compiler.
        """
        return ""

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: List[Any],
        compiler_config: Dict[str, Any],
        runtime_shape: Optional[int] = None
    ) -> Tuple[Optional[Callable], Optional[Any]]:
        """
        Compile the graph with the given example inputs and compiler config,
        with a runtime shape. If the `runtime_shape` is None, it means
        the `example_inputs` have a dynamic shape. Otherwise, the
        `runtime_shape` specifies the shape of the inputs. Right now we only
        support one variable shape for all inputs, which is the batchsize
        (number of tokens) during inference.

        Dynamo will make sure `graph(*example_inputs)` is valid.

        The function should return a compiled callable function, as well as
        a handle that can be used to directly load the compiled function.

        The handle should be a plain Python object, preferably a string or a
        file path for readability.

        If the compiler doesn't support caching, it should return None for the
        handle. If the compiler fails to compile the graph, it should return
        None for the compiled function as well.
        """
        return None, None

    def load(self,
             handle: Any,
             graph: fx.GraphModule,
             example_inputs: List[Any],
             graph_index: int,
             runtime_shape: Optional[int] = None) -> Callable:
        """
        Load the compiled function from the handle.
        Raises an error if the handle is invalid.

        The handle is the second return value of the `compile` function.
        """
        raise NotImplementedError("caching is not supported")


class AlwaysHitShapeEnv:
    """
    Why do we need this class:

    For normal `torch.compile` usage, every compilation will have
    one Dynamo bytecode compilation and one Inductor compilation.
    The Inductor compilation happens under the context of the
    Dynamo bytecode compilation, and that context is used to
    determine the dynamic shape information, etc.

    For our use case, we only run Dynamo bytecode compilation once,
    and run Inductor compilation multiple times with different shapes
    plus a general shape. The compilation for specific shapes happens
    outside of the context of the Dynamo bytecode compilation. At that
    time, we don't have shape environment to provide to Inductor, and
    it will fail the Inductor code cache lookup.

    By providing a dummy shape environment that always hits, we can
    make the Inductor code cache lookup always hit, and we can
    compile the graph for different shapes as needed.

    The following dummy methods are obtained by trial-and-error
    until it works.
    """

    def __init__(self) -> None:
        self.guards: List[Any] = []

    def evaluate_guards_expression(self, *args, **kwargs):
        return True

    def get_pruned_guards(self, *args, **kwargs):
        return []

    def produce_guards_expression(self, *args, **kwargs):
        return ""


class InductorAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler, version 2.5 and 2.6.
    """
    name = "inductor"

    def compute_hash(self, aphrodite_config: AphroditeConfig) -> str:
        factors: List[Any] = []
        # summarize system state
        from torch._inductor.codecache import CacheBase
        system_factors = CacheBase.get_system()
        factors.append(system_factors)

        # summarize pytorch state
        from torch._inductor.codecache import torch_key
        torch_factors = torch_key()
        factors.append(torch_factors)
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def initialize_cache(self, cache_dir: str, disable_cache: bool = False):
        self.cache_dir = cache_dir
        if disable_cache:
            return
        # redirect the cache directory to a sub-directory
        # set flags so that Inductor and Triton store their cache
        # in the cache_dir, then users only need to copy the cache_dir
        # to another machine to reuse the cache.
        inductor_cache = os.path.join(cache_dir, "inductor_cache")
        os.makedirs(inductor_cache, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
        triton_cache = os.path.join(cache_dir, "triton_cache")
        os.makedirs(triton_cache, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: List[Any],
        compiler_config: Dict[str, Any],
        runtime_shape: Optional[int] = None
    ) -> Tuple[Optional[Callable], Optional[Any]]:
        current_config = {}
        from torch._inductor.compile_fx import compile_fx

        # disable remote cache
        current_config["fx_graph_cache"] = True
        current_config["fx_graph_remote_cache"] = False

        if compiler_config is not None:
            current_config.update(compiler_config)

        if isinstance(runtime_shape, int):
            # for a specific batchsize, tuning triton kernel parameters
            # can be beneficial
            current_config["max_autotune"] = True
            current_config["coordinate_descent_tuning"] = True

        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        # it's the first time we compile this graph
        # the assumption is that we don't have nested Inductor compilation.
        # compiled_fx_graph_hash will only be called once, and we can hook
        # it to get the hash of the compiled graph directly.

        hash_str, file_path = None, None
        from torch._inductor.codecache import (FxGraphCache,
                                               compiled_fx_graph_hash)
        if torch.__version__.startswith("2.5"):
            original_load = FxGraphCache.load
            original_load_name = "torch._inductor.codecache.FxGraphCache.load"

            def hijack_load(*args, **kwargs):
                inductor_compiled_graph = original_load(*args, **kwargs)
                nonlocal file_path
                compiled_fn = inductor_compiled_graph.current_callable
                file_path = compiled_fn.__code__.co_filename  # noqa
                if not file_path.startswith(self.cache_dir):
                    # hooked in the align_inputs_from_check_idxs function
                    # in torch/_inductor/utils.py
                    for cell in compiled_fn.__closure__:
                        if not callable(cell.cell_contents):
                            continue
                        if cell.cell_contents.__code__.co_filename.startswith(
                                self.cache_dir):
                            # this is the real file path compiled from Inductor
                            file_path = cell.cell_contents.__code__.co_filename
                            break
                return inductor_compiled_graph

            hijacked_compile_fx_inner = torch._inductor.compile_fx.compile_fx_inner  # noqa
        elif torch.__version__ >= "2.6":
            # function renamed in 2.6
            original_load_name = None

            def hijacked_compile_fx_inner(*args, **kwargs):
                output = torch._inductor.compile_fx.compile_fx_inner(
                    *args, **kwargs)
                nonlocal hash_str
                inductor_compiled_graph = output
                if inductor_compiled_graph is not None:
                    nonlocal file_path
                    compiled_fn = inductor_compiled_graph.current_callable
                    file_path = compiled_fn.__code__.co_filename  # noqa
                    if not file_path.startswith(self.cache_dir):
                        # hooked in the align_inputs_from_check_idxs function
                        # in torch/_inductor/utils.py
                        for cell in compiled_fn.__closure__:
                            if not callable(cell.cell_contents):
                                continue
                            code = cell.cell_contents.__code__
                            if code.co_filename.startswith(self.cache_dir):
                                # this is the real file path
                                # compiled from Inductor
                                file_path = code.co_filename
                                break
                    hash_str = inductor_compiled_graph._fx_graph_cache_key
                return output

        def hijack_compiled_fx_graph_hash(*args, **kwargs):
            out = compiled_fx_graph_hash(*args, **kwargs)
            nonlocal hash_str
            hash_str = out[0]
            return out

        def _check_can_cache(*args, **kwargs):
            # no error means it can be cached.
            # Inductor refuses to cache the graph outside of Dynamo
            # tracing context, and also disables caching for graphs
            # with high-order ops.
            # For Aphrodite, in either case, we want to cache the graph.
            # see https://github.com/pytorch/pytorch/blob/9f5ebf3fc609105a74eab4ccc24932d6353ff566/torch/_inductor/codecache.py#L1221 # noqa
            return

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        with ExitStack() as stack:
            # hijack to get the compiled graph itself
            if original_load_name is not None:
                stack.enter_context(patch(original_load_name, hijack_load))

            # for hijacking the hash of the compiled graph
            stack.enter_context(
                patch("torch._inductor.codecache.compiled_fx_graph_hash",
                      hijack_compiled_fx_graph_hash))

            # for providing a dummy shape environment
            stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      _get_shape_env))

            from torch._functorch._aot_autograd.autograd_cache import (
                AOTAutogradCache)

            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        _get_shape_env))

            # for forcing the graph to be cached
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache))

            # Dynamo metrics context, see method for more details.
            stack.enter_context(self.metrics_context())

            # Disable remote caching. When these are on, on remote cache-hit,
            # the monkey-patched functions never actually get called.
            # Aphrodite today assumes and requires the monkey-patched functions
            # to get hit.
            # TODO(zou3519): we're going to replace this all with
            # standalone_compile sometime.
            if is_torch_equal_or_newer("2.6"):
                stack.enter_context(
                    torch._inductor.config.patch(fx_graph_remote_cache=False))
                stack.enter_context(
                    torch._functorch.config.patch(
                        enable_remote_autograd_cache=False))

            with pass_context(runtime_shape):
                compiled_graph = compile_fx(
                    graph,
                    example_inputs,
                    inner_compile=hijacked_compile_fx_inner,
                    config_patches=current_config)

        # We treat APHRODITE_DISABLE_COMPILE_CACHE as the overall switch for
        # torch compilation cache. So turn off the checks if we disable the
        # compilation cache.
        if not envs.APHRODITE_DISABLE_COMPILE_CACHE:
            assert hash_str is not None, (
                "failed to get the hash of the compiled graph")
            assert file_path is not None, (
                "failed to get the file path of the compiled graph")
        return compiled_graph, (hash_str, file_path)

    def load(self,
             handle: Any,
             graph: fx.GraphModule,
             example_inputs: List[Any],
             graph_index: int,
             runtime_shape: Optional[int] = None) -> Callable:
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)
        hash_str = handle[0]

        from torch._functorch._aot_autograd.autograd_cache import (
            AOTAutogradCache)
        from torch._inductor.codecache import FxGraphCache
        with ExitStack() as exit_stack:
            exit_stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      lambda *args, **kwargs: AlwaysHitShapeEnv()))
            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                exit_stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        lambda *args, **kwargs: AlwaysHitShapeEnv()))

            # Dynamo metrics context, see method for more details.
            exit_stack.enter_context(self.metrics_context())

            if torch.__version__.startswith("2.5"):
                inductor_compiled_graph = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, False)
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove"
                    f"the cache directory and try again."  # noqa
                )
            elif torch.__version__ >= "2.6":
                from torch._inductor.output_code import (
                    CompiledFxGraphConstantsWithGm)
                constants = CompiledFxGraphConstantsWithGm(graph)
                inductor_compiled_graph, _ = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, None, constants)
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove"
                    f"the cache directory and try again."  # noqa
                )

        # Inductor calling convention (function signature):
        # f(list) -> tuple
        # Dynamo calling convention (function signature):
        # f(*args) -> Any

        # need to know if the graph returns a tuple
        from torch._inductor.compile_fx import graph_returns_tuple
        returns_tuple = graph_returns_tuple(graph)

        # this is the callable we return to Dynamo to run
        def compiled_graph(*args):
            # convert args to list
            list_args = list(args)
            graph_output = inductor_compiled_graph(list_args)
            # unpack the tuple if needed
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph

    def metrics_context(self) -> contextlib.AbstractContextManager:
        """
        This method returns the Dynamo metrics context (if it exists,
        otherwise a null context). It is used by various compile components.
        Present in torch>=2.6, it's used inside FxGraphCache in
        torch==2.6 (but not after). It might also be used in various other
        torch.compile internal functions.

        Because it is re-entrant, we always set it (even if entering via Dynamo
        and the context was already entered). We might want to revisit if it
        should be set at a different level of compilation.

        This is likely a bug in PyTorch: public APIs should not rely on
        manually setting up internal contexts. But we also rely on non-public
        APIs which might not provide these guarantees.
        """
        if is_torch_equal_or_newer("2.6"):
            import torch._dynamo.utils
            return torch._dynamo.utils.get_metrics_context()
        else:
            return contextlib.nullcontext()


class EagerAdaptor(CompilerInterface):
    name = "eager"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: List[Any],
        compiler_config: Dict[str, Any],
        runtime_shape: Optional[int] = None
    ) -> Tuple[Optional[Callable], Optional[Any]]:
        # we don't need to compile the graph, just return the graph itself.
        # It does not support caching, return None for the handle.
        return graph, None
