###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from loguru import logger

def get_hw():
    import habana_frameworks.torch.utils.experimental as htexp
    device_type = htexp._get_device_type()
    match device_type:
        case htexp.synDeviceType.synDeviceGaudi2:
            return "gaudi2"
        case htexp.synDeviceType.synDeviceGaudi3:
            return "gaudi3"
    from aphrodite.hpu_extension.utils import is_fake_hpu
    if is_fake_hpu():
        return "cpu"
    logger.warning(f'Unknown device type: {device_type}')
    return None


def get_build():
    import re
    import subprocess
    output = subprocess.run("pip show habana-torch-plugin",
                            shell=True,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    version_re = re.compile(r'Version:\s*(?P<version>.*)')
    match = version_re.search(output.stdout)
    if output.returncode == 0 and match:
        return match.group('version')
    # In cpu-test environment we don't have access to habana-torch-plugin
    from aphrodite.hpu_extension.utils import is_fake_hpu
    result = '0.0.0.0' if is_fake_hpu() else None
    logger.warning(f"Unable to detect habana-torch-plugin version! Returning: {result}")
    return result


runtime_params = {}


def get_model_type():
    return runtime_params.get('model_type', None)


def set_model_config(cfg):
    global runtime_params
    runtime_params['model_type'] = cfg.hf_config.model_type


def get_environment(**overrides):
    overrides = {k: lambda: v for k, v in overrides.items()}
    getters = {
        "build": get_build,
        "hw": get_hw,
        "model_type": get_model_type,
    }
    return {k: g() for k, g, in (getters | overrides).items()}
