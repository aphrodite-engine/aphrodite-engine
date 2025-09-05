from collections import defaultdict
from functools import wraps
from numbers import Number
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import aphrodite.modeling.model_loader as loader
from aphrodite.config import AphroditeConfig

aten = torch.ops.aten


def _prod(x):
    res = 1
    for i in x:
        res *= i
    return res


def matmul_flop(inputs: list[Any], outputs: list[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs contains the shapes of two matrices.
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = _prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop


def addmm_flop(inputs: list[Any], outputs: list[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    input_shapes = [v.shape for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


def bmm_flop(inputs: list[Any], outputs: list[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def log_softmax_flop(inputs: list[Any], outputs: list[Any]) -> Number:
    return 2 * inputs[0].numel()


def nln_flop(inputs: list[Any], outputs: list[Any]) -> Number:
    return 4 * inputs[0].numel()


def softmax_flop(inputs: list[Any], outputs: list[Any]) -> Number:
    return 2 * inputs[0].numel()


def relu_flop(inputs: list[Any], outputs: list[Any]) -> Number:
    return 2 * inputs[0].numel()


def attn_flop(q, k, v, *args) -> Number:
    """
    Count flops for attention operation.
    Calculation of QK^T and PV each contribute bns^d FLOPS.
    """
    macs = _prod(q.shape) * k.shape[-2]
    macs += _prod(q.shape[:-1]) * k.shape[-2] * v.shape[-1]

    return 2 * macs


flop_mapping = {
    aten.mm: matmul_flop,
    aten.mm.default: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten._log_softmax: log_softmax_flop,
    aten.native_layer_norm: nln_flop,
    aten._softmax: softmax_flop,
    aten.relu: relu_flop
}

context_manager = None


class FlopContextManager(TorchDispatchMode):
    """
    Creates a Context Manager to count FLOPS for each operation and sub-module
    of an LLM ran with Aphrodite.
    """

    # @param kwargs should consist of functions to add to the flop_mapping if
    # there are any operations not included in the above mapping.
    def __init__(self, load_format='default', flop_funcs_dict={}):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.funcs = set()
        self.parents = ['Global']
        self.flop_mapping = flop_mapping
        self.model = None
        for key, value in flop_funcs_dict.items():
            if isinstance(value, Callable):
                self.flop_mapping[key] = value

        self._wrap_model(load_format)
        global context_manager
        context_manager = self

    def _set_model(self, model):
        assert model is not None
        if self.model is not None:
            self.remove_hooks()

        self.model = model
        if model is not None:
            self.model.apply(self.register_hooks)

    def register_hooks(self, module):
        name = module.__class__.__name__
        module.__pre_hook__ = module.register_forward_pre_hook(
            self.enter_module(name))
        module.__post_hook__ = module.register_forward_hook(
            self.exit_module(name))

    def remove_hooks(self):
        if hasattr(self.model, "__pre_hook__"):
            self.model.__pre_hook__.remove()
            del self.model.__pre_hook__
        if hasattr(self.model, "__post_hook__"):
            self.model.__post_hook__.remove()
            del self.model.__post_hook__

    def enter_module(self, name):
        def f(model, inputs):
            self.parents.append(name)
            return inputs

        return f

    def exit_module(self, name):
        def f(model, inputs, outputs):
            assert self.parents[-1] == name
            self.parents.pop()
            return outputs
        return f

    def _wrap_model(self, load_format):

        self.loader_class = get_model_loader_type(load_format)
        if self.loader_class is None:
            return

        self.original_func = load_model_func = self.loader_class.load_model
        assert load_model_func is not None

        @wraps(load_model_func)
        def wrapper(self, *, aphrodite_config: AphroditeConfig):
            model = load_model_func(self, aphrodite_config=aphrodite_config)
            context_manager._set_model(model)
            return model

        setattr(self.loader_class, 'load_model', wrapper)

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        print(
            f"\nTotal: {sum(self.flop_counts['Global'].values())/1e9} GFLOPS")
        for mod in self.flop_counts.keys():
            print(f"Module: {mod}")
            for k,v in self.flop_counts[mod].items():
                print(f"{k}: {v/1e9} GFLOPS")
            print()

        self.remove_hooks()
        setattr(
            self.loader_class,
            self.original_func.__name__,
            self.original_func,
        )
        global context_manager 
        context_manager = None 
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs={}):
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        self.funcs.add(func_packet)
        if func_packet in self.flop_mapping:
            flop_count = self.flop_mapping[func_packet](
                args, out if isinstance(out, tuple) else (out,),
            )
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        elif 'attention' in func_packet.op.__name__: 
            flop_count = attn_flop(*args)
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        return out


def get_model_loader_type(load_format):
    from aphrodite.modeling.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER

    return _LOAD_FORMAT_TO_MODEL_LOADER.get(
        load_format, loader.DefaultModelLoader
    )
