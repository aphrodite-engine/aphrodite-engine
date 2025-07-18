###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from aphrodite.hpu_extension.environment import get_environment
from aphrodite.hpu_extension.kernels import fsdpa

detected = None


class FeatureTest:
    def __init__(self, *required_params):
        self.required_params = set(required_params)

    def __call__(self, **kwargs):
        missing_params = self.required_params - kwargs.keys()
        assert len(missing_params) == 0, f'Missing keys: {missing_params}!'
        params = {k: v for k, v in kwargs.items() if k in self.required_params}
        missing_values = {k for k, v in params.items() if v is None}
        assert len(missing_values) == 0, f'Missing values for parameters: {missing_values}!'
        return self.check(**params)

    def __and__(self, rhs):
        return And(self, rhs)

    def check(self, *_, **__):
        raise NotImplementedError("check needs to be implemented in subclasses")


class Not(FeatureTest):
    def __init__(self, child):
        super().__init__(*child.required_params)
        self.child = child

    def check(self, **kwargs):
        return not self.child.check(**kwargs)


class And(FeatureTest):
    def __init__(self, lhs, rhs):
        super().__init__(*lhs.required_params, *rhs.required_params)
        self.lhs = lhs
        self.rhs = rhs

    def check(self, **kwargs):
        return self.lhs(**kwargs) and self.rhs(**kwargs)


class Value(FeatureTest):
    def __init__(self, key, value):
        super().__init__(key)
        self.key = key
        self.value = value

    def check(self, **env):
        return env[self.key] == self.value


class Hardware(Value):
    def __init__(self, target_hw):
        super().__init__('hw', target_hw)


class ModelType(Value):
    def __init__(self, model_type):
        super().__init__('model_type', model_type)


class EnvFlag(FeatureTest):
    def __init__(self, flag, default_value):
        if isinstance(default_value, FeatureTest):
            required_params = default_value.required_params
        else:
            required_params = []
        super().__init__(*required_params)
        self.flag = flag
        self.default_value = default_value

    def check(self, **env):
        if isinstance(self.default_value, FeatureTest):
            default = self.default_value.check(**env)
        else:
            default = self.default_value
        return os.environ.get(self.flag, str(default)).lower() in ['true', '1']


class Kernel(FeatureTest):
    def __init__(self, loader_fn):
        super().__init__()
        self.loader_fn = loader_fn

    def check(self):
        return self.loader_fn() is not None


class VersionRange(FeatureTest):
    def __init__(self, *specifiers):
        super().__init__('build')
        self.specifiers = [SpecifierSet(s) for s in specifiers]

    def check(self, build):
        version = Version(build)
        return any(version in s for s in self.specifiers)


class Flags:
    def __init__(self, features, environment):
        self.all = set(features.keys())
        self.enabled = set(name for name, check in features.items() if check(**environment))
        self.disabled = self.all - self.enabled

    def is_enabled(self, *names):
        return all(n in self.enabled for n in names)

    def is_disabled(self, *names):
        return all(n in self.disabled for n in names)

    def __repr__(self):
        feature_list = [('+' if self.is_enabled(f) else '-') + f for f in sorted(self.all)]
        return f'[{(" ").join(feature_list)}]'

    def _check(self, name):
        if name.startswith('-'):
            return self.is_disabled(name[1:])
        if name.startswith('+'):
            return self.is_enabled(name[1:])
        return self.is_enabled(name)

    def __contains__(self, names):
        return all(self._check(name) for name in names.split(','))


def enabled_flags():

    global detected

    if detected:
        return detected

    supported_flags = {
        "gaudi": Hardware("gaudi"),
        "gaudi2": Hardware("gaudi2"),
        "gaudi3": Hardware("gaudi3"),
        "cpu": Hardware("cpu"),
        "fp32_softmax": EnvFlag("APHRODITE_FP32_SOFTMAX", ModelType('qwen2')),
        "fsdpa": (Not(Hardware("cpu"))
                  & Kernel(fsdpa)
                  & EnvFlag("APHRODITE_PROMPT_USE_FUSEDSDPA",
                            Not(ModelType('qwen2')) & Not(ModelType('mllama')))),
        "compile_one_hot": (VersionRange(">=1.20.0.370") & Not(EnvFlag("PT_HPU_LAZY_MODE", "1"))),
        "flex_attention": (Not(Hardware("cpu")) & Not(EnvFlag("PT_HPU_LAZY_MODE", "1"))
                           & ModelType("llama")
                           & Not(EnvFlag("APHRODITE_PROMPT_USE_FUSEDSDPA", "false"))
                           & EnvFlag("APHRODITE_PROMPT_USE_FLEX_ATTENTION", "false")),
    }
    environment = get_environment()
    detected = Flags(supported_flags, environment)
    print(f'Detected flags: {detected}')
    return detected
