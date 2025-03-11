---
title: Adding a Custom Class in PyTorch
---

PyTorch uses a framework called [TorchBind](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html) to bind custom C++ classes, which may carry mutable state to Python. PyTorch has built the infrastructure support for these TorchBind classes within the PyTorch 2 stack (`torch.compile` and `torch.export`).

## How to support existing TorchBind classes

There are a few steps you'd have to follow:

1. Implement an `__obj_flatten__` method to the C++ custom class implementation to allow us to inspect its states and guard the changes. The method should return a tuple of tuple of `attribute_name`, value (`tuple[tuple[str, value] * n]`).
2. Register a python fake class using `@torch._library.register_fake_class`.
    1. Implement "fake methods" of each of the class's C++ methods, which should have the same schema as the C++ implementation.
    2. Additionally, implement an `__obj_flatten__` classmethod in the Python fake class to tell us how to create a fake class from the flattened states returned by `__obj_flatten__`.

Here's a breakdown of how to do this in practice. Following the guide in [Extending TorchScript with Custom C++ Classes](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html), we can create a  [thread-safe tensor queue](https://github.com/pytorch/pytorch/blob/24f69eef6add3a5446d1d4f0651e401c2220d95d/test/cpp/jit/test_custom_class_registrations.cpp#L130) (copied from fbgemm) and build it.


```cpp
// Thread-safe Tensor Queue
struct TensorQueue : torch::CustomClassHolder {
 ...
 private:
  std::deque<at::Tensor> queue_;
  std::mutex mutex_;
  at::Tensor init_tensor_;
};
// The torch binding code
TORCH_LIBRARY(MyCustomClass, m) {
  m.class_<TensorQueue>("TensorQueue")
      .def(torch::init<at::Tensor>())
      .def("push", &TensorQueue::push)
      .def("pop", &TensorQueue::pop)
      .def("top", &TensorQueue::top)
      .def("size", &TensorQueue::size)
      .def("clone_queue", &TensorQueue::clone_queue)
      .def("get_raw_queue", &TensorQueue::get_raw_queue)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<TensorQueue>& self)
              -> c10::Dict<std::string, at::Tensor> {
            return self->serialize();
          },
          // __setstate__
          [](c10::Dict<std::string, at::Tensor> data)
              -> c10::intrusive_ptr<TensorQueue> {
            return c10::make_intrusive<TensorQueue>(std::move(data));
          });
}
```

**Step 1**: Add an `__obj_flatten__` method to the C++ Custom Class implementation:
```cpp
// Thread-safe Tensor Queue
struct TensorQueue : torch::CustomClassHolder {
  ...
  std::tuple<std::tuple<std::string, std::vector<at::Tensor>>, std::tuple<std::string, at::Tensor>> __obj_flatten__() {  // [!code highlight]
    return std::tuple(std::tuple("queue", this->get_raw_queue()), std::tuple("init_tensor_", this->init_tensor_.clone()));  // [!code highlight]
  }  // [!code highlight]
  ...
}

TORCH_LIBRARY(MyCustomClass, m) {
  m.class_<TensorQueue>("TensorQueue")
      .def(torch::init<at::Tensor>())
      .def("__obj_flatten__", &TensorQueue::__obj_flatten__)  // [!code highlight]
      ...
}
```

**Step 2a**: Register a fake class in Python that implements each method.

```py
# namespace::class_name
@torch._library.register_fake_class("MyCustomClass::TensorQueue")  # [!code highlight]
class FakeTensorQueue:
    def __init__(
        self, 
        queue: List[torch.Tensor], 
        init_tensor_: torch.Tensor
    ) -> None:
        self.queue = queue
        self.init_tensor_ = init_tensor_
    
    def push(self, tensor: torch.Tensor) -> None:  # [!code highlight]
        self.queue.append(tensor)
    
    def pop(self) -> torch.Tensor:  # [!code highlight]
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return self.init_tensor_
    
    def size(self) -> int:  # [!code highlight]
	 return len(self.queue)
```

**Step 2b**: Implement an `__obj_unflatten__` classmethod in Python:

```py
# namespace::class_name
@torch._library.register_fake_class("MyCustomClass::TensorQueue")
class FakeTensorQueue:
  ...
    @classmethod  # [!code highlight]
    def __obj_unflatten__(cls, flattened_tq):  # [!code highlight]
        return cls(**dict(flattened_tq))  # [!code highlight]
    
  ...
```

That's it! Now we can create a module that uses this object and run it with `torch.compile` or `torch.export`:

```py
import torch

torch.ops.load_library("//caffe2/test:test_torchbind_cpp_impl")
tq = torch.classes.MyCustomClass.TensorQueue(torch.empty(0).fill_(-1))

class Mod(torch.nn.Module):
    def forward(self, tq, x):
        tq.push(x.sin())
        tq.push(x.cos())
        poped_t = tq.pop()
        assert torch.allclose(poped_t, x.sin())
        return tq, poped_t

tq, poped_t = torch.compile(Mod(), backend="eager", fullgraph=True)(tq, torch.randn(2, 3))  # [!code highlight]
assert tq.size() == 1

exported_program = torch.export.export(Mod(), (tq, torch.randn(2, 3),), strict=False)  # [!code highlight]
exported_program.module()(tq, torch.randn(2, 3))  # [!code highlight]

```

We can also implement custom ops that take custom classes as input. For example, we could register a custom op `for_each_add_(tq, tensor)`

```cpp
struct TensorQueue : torch::CustomClassHolder {
 ...
 void for_each_add_(at::Tensor inc) {  // [!code highlight]
    for (auto& t : queue_) {  // [!code highlight]
      t.add_(inc);  // [!code highlight]
    }  // [!code highlight]
  }  // [!code highlight]
 ...
}


TORCH_LIBRARY_FRAGMENT(MyCustomClass, m) {
  m.class_<TensorQueue>("TensorQueue")  // [!code highlight]
   .def("for_each_add_", &TensorQueue::for_each_add_);  // [!code highlight]

  m.def(  // [!code highlight]
      "for_each_add_(__torch__.torch.classes.MyCustomClass.TensorQueue foo, Tensor inc) -> ()");  // [!code highlight]

}

void for_each_add_(c10::intrusive_ptr<TensorQueue> tq, at::Tensor inc) {  // [!code highlight]
  tq->for_each_add_(inc);  // [!code highlight]
}  // [!code highlight]

TORCH_LIBRARY_IMPL(MyCustomClass, CPU, m) {
  m.impl("for_each_add_", for_each_add_);  // [!code highlight]
}
```

Since the fake class is implemented in python, we require the fake implementation of custom op **must also be registered in python**:

```py
@torch.library.register_fake("MyCustomClass::for_each_add_")
def fake_for_each_add_(tq, inc):
    tq.for_each_add_(inc)
```

After re-compilation, we can export the custom op with:

```py
class ForEachAdd(torch.nn.Module):
  def forward(self, tq: torch.ScriptObject, a: torch.Tensor) -> torch.ScriptObject:
    torch.ops.MyCustomClass.for_each_add_(tq, a)
    return tq
mod = ForEachAdd()
tq = empty_tensor_queue()
qlen = 10
for i in range(qlen):
  tq.push(torch.zeros(1))

ep = torch.export.export(mod, (tq, torch.ones(1)), strict=False)

assertExpectedInline(ep.graph_module.code.strip(), """\
def forward(self, token, tq, a):
    with_effects = torch._higher_order_ops.effects.with_effects(token, torch.ops.MyCustomClass.for_each_add_.default, tq, a);  token = a = None
    getitem = with_effects[0];  with_effects = None
    return (getitem, tq)""")

output_tq = ep.module()(tq, torch.ones(1))
assertEqual(output_tq.size(), qlen)
assertEqual(output_tq.top(), torch.ones(1))
```

### Why do we need to make a Fake Class?

Tracing with real custom object has several major downsides:

1. Operators on real objects can be **time consuming**, e.g. the custom object might be reading from the network or loading data from the disk.
2. We don't want to mutate the real custom object or create **side effects** to the environment while tracing.
3. It cannot support **dynamic shape**.

However, it may be difficult for users to write a fake class: the original class may use some third-party library (e.g. koski) that determines the output shape of the methods, or is complicated and written by others. Besides, users may not care  about the limitations listed above. In that case, please reach out to the PyTorch developers.

### Known Limitations

**Aliasing and mutation**:

Currently, `aot_autograd` can't functionalize the script object's methods due to a lack of accurate schema that shows the aliasing/mutation relationship.

This causes the current PyTorch support to have the limitation that once a tensor is passed into the script object method or returned from a script object method, **it's not safe to do in-place mutation** to the tensor in the python program anymore. See this example:


```py
tq.push(a)
a.sin_()
b = tq.pop()
return a + b

# aot_autograd captured graph
call_torchbind( tq, 'push', clone)
sin = torch.ops.aten.sin.default(a);  clone = None
popped = call_torchbind(tq, 'pop');  getitem = tq = None
return sin + popped
```

`sin_` is replaced by `sin` as a result of functionalization but the mutation to the content of `tq` is not showing up in the graph, causing the captured graph to produce a different result than the original program.

**`is` operator**:

Besides mutation, we shouldn't rely on comparing a tensor with a tensor contained inside a script object with `is`. For example:

```py
# will return a + 1
tq.push(a)
if tq.pop() is a:
return a + 1
return a - 1

# aot_autograd captured graph:
call_torchbind(tq, 'push', a)
return a - 1
```

The captured graph of `aot_autograd` returns `a - 1` while eager mode returns `a + 1`. This is because `is` compares real tensors in eager mode but compares `FunctionalTensor` wrappers during `aot_autograd` tracing. The wrapper comparison fails to capture the information that `a` and `top.pop()` are actually referring to the same tensor.

**Constant burning-in graph**:

For methods that return non-tensors, e.g. `int`, `float`, `string`, `tuple`, etc, they will be treated as constants and burned in the graph (consistent with how export deals with constant inputs). For example:

```py
def f(tq, x):
  return x + tq.size()

# captured graph for tq = [torch.randn(3,3), torch.randn(3, 3)]
def graph(tq, x):
  sz = call_torchbind(tq, "size", (,), {})
  return x + 2 # instead of x + sz
```

## How do things work under the hood
1. Fakify custom object (`dynamo` and `aot_autograd`)
Before tracing, when receiving a torch bind object, we’ll:
  1. Call `__obj_flatten__` on the torchbind object to get a `flattend_obj`.
  2. If we’re in dynamo, install guards on `flattened_obj`. Specifically, we guard the `flattened_obj` directly as if it’s the input.
  3. Fakify the tensors in `flattened_obj`. 
  4. Retrieve the fake class registered with `register_fake_class`. 
  5. Call `fake_class.__obj_unflatten__` to create the fake object.
  6. If we’re in dynamo, we additionally create `TorchScriptObjectVariable` that’s backed by the `FakeScriptObject` to simulate the behavior of torchbind object.

During tracing, method calls will use the fake methods to create example values for downstream to use. Custom ops that take torch bind object inputs will call the python fake implementation and call the methods of fake script object.
 
2. Method Calls are turned into Operators

Each method call of a custom object potentially mutates its states. Therefore, we need to record all the methods of the custom object into the graph. The way we do it is to turn method calls into a higher order operator `call_torchbind`. The signature of the operator looks like:

```py
call_torchbind(custom_obj: torch.ScriptObject, method_name: str, args: Tuple[Args], kwargs, Dict[str, Args])
```

For the above example on `Mod`, dynamo will record a graph that looks like the following:
```py
def graph(tq, a, b):
  call_torchbind(tq, "push", (a,), {})
  torch.ops.MyCustomClass.queue_push(tq, (b,), {})
  torch.ops.MyCustomClass.queue_push(tq, (b.sin(),), {})
  popped1 = torch.ops.MyCustomClass.queue_pop(tq, (), {})
  popped2 = call_torchbind(tq, "pop", (), {})
  return (popped1, popped2)
```

In **dynamo**, this is done by constructing a `TorchScriptObjectVariable` and turns all the call_methods call into a higher order op call.

In **aot_autograd**, this is done by intercepting the `ScriptMethod.__call__`.

3. Functionalize `call_torchbind` and `torchbind` operators:
As custom objects carry state, it is possible for operations on these custom objects to mutate the underlying states. In order to prevent downstream optimization passes from accidentally reordering or deleting these method calls (e.g. when the method doesn’t have output), we make use of **effect tokens** to thread data dependency explicitly in the graph. A token is passed as an input to the graph, and  between each operator call which uses a custom class through the `with_effects` higher order operator, and outputted as a result of the graph. The schema of the `with_effects` operator is:

```py
with_effects(token, operator, *args, **kwargs)
```

For the above example on `Mod`, the functionalized `AOTAutograph` graph will look like the following:
```py
def graph(token, tq, a, b):
  token1 = with_effects(token, call_torchbind, tq, "push", a)
  token2 = with_effects(token1, torch.ops.MyCustomClass.queue_push.default, tq, b)
  token3 = with_effects(token2, torch.ops.MyCustomClass.queue_push.default, tq, b.sin())
  token4, popped1 = with_effects(token3, torch.ops.MyCustomClass.queue_pop.default, tq)
  token5, popped2 = with_effects(token4, call_torchbind, tq, "pop")
  return token5, tq, popped1, popped2
```

Custom classes' methods have schemas but these schemas are auto-generated and don’t have mutation or aliasing information attached to it. We can leverage the `auto_functionalize` infra built for custom ops to auto functionalize the method calls, where we assume all the inputs can be mutated and rely on the backend’s (e.g. inductor's) re-replacing pass to remove the data copies.

4. Inductor
Once we get to the Inductor IR, `with_effects` calls are converted to an `EffectfulKernel`. During scheduling, we will create a `StarDep` between each `EffectfulKernel` so that they don’t get reordered. The buffers look something like this:

```py
buf2: ExternKernelSchedulerNode(
  EffectfulKernel(
    inputs=[TorchBindObject("arg1_1"), InputBuffer("arg2_1")],
    constant_args=("push",),
    op_overload=call_torchbind,
  )
)
buf2.writes = [StarDep(name='buf2', mode=None)]
buf2.unmet_dependencies = []
buf2.met_dependencies = [StarDep(name='arg1_1', mode=None), StarDep(name='arg2_1', mode=None)]
buf2.users = [NodeUser(node=ExternKernelSchedulerNode(name='buf3'), can_inplace=False, is_weak=False)]


buf3: ExternKernelSchedulerNode(
  EffectfulKernel(
    inputs=[TorchBindObject("arg1_1"), InputBuffer("arg3_1")],
    op_overload=MyCustomClass.queue_push.default,
  )
)
buf3.writes = [StarDep(name='buf3', mode=None)]
buf3.unmet_dependencies = [StarDep(name='buf2', mode=None)]  # [!code highlight]
buf3.met_dependencies = [StarDep(name='arg1_1', mode=None), StarDep(name='arg3_1', mode=None)]
buf3.users = [NodeUser(node=ExternKernelSchedulerNode(name='buf5'), can_inplace=False, is_weak=False)]
```

The inductor generated code looks something like this:

```py
import torch
from torch._inductor.async_compile import AsyncCompile

empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
async_compile = AsyncCompile()

cpp_fused_sin_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_angelayi/ky/cky2bufythacofebk7ujv36e4pxyqcqbpsy5r4vojoprjiwcwfxf.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = std::sin(tmp0);
            out_ptr0[static_cast<long>(x0)] = tmp1;
        }
    }
}
''')

def call(args):
    arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg2_1, (2, 3), (3, 1))
    assert_size_stride(arg3_1, (2, 3), (3, 1))
    # Source Nodes: [call_torchbind], Original ATen: []
    torch.ops.higher_order.call_torchbind(arg1_1, 'push', arg2_1)
    del arg2_1
    # Source Nodes: [queue_push], Original ATen: []
    torch.ops.MyCustomClass.queue_push.default(arg1_1, arg3_1)
    buf4 = empty_strided_cpu((2, 3), (3, 1), torch.float32)
    cpp_fused_sin_0(arg3_1, buf4)
    del arg3_1
    # Source Nodes: [queue_push_1, sin], Original ATen: [aten.sin]
    torch.ops.MyCustomClass.queue_push.default(arg1_1, buf4)
    del buf4
    # Source Nodes: [popped1], Original ATen: []
    buf6 = torch.ops.MyCustomClass.queue_pop.default(arg1_1)
    buf7 = buf6
    # Source Nodes: [popped2], Original ATen: []
    buf8 = torch.ops.higher_order.call_torchbind(arg1_1, 'pop')
    del buf6
    buf9 = buf8
    return (buf7, buf9, )
```