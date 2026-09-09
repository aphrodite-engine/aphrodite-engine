# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project

from unittest.mock import Mock, patch

import pytest

from aphrodite.omni.quantization.component_config import ComponentQuantizationConfig
from aphrodite.platforms.interface import DeviceCapability


@pytest.mark.parametrize("capability", [80, 90, 110])
def test_component_capability_uses_highest_requirement(capability):
    low = Mock(get_min_capability=Mock(return_value=80))
    high = Mock(get_min_capability=Mock(return_value=90))
    platform = Mock(
        is_cuda=Mock(return_value=True),
        get_device_capability=Mock(return_value=DeviceCapability(capability // 10, capability % 10)),
    )
    with patch("aphrodite.omni.quantization.component_config.current_platform", platform):
        if capability < 90:
            with pytest.raises(ValueError, match="requires CUDA capability 90"):
                ComponentQuantizationConfig({"encoder": low}, default_config=high)
        else:
            config = ComponentQuantizationConfig({"encoder": low}, default_config=high)
            assert config.resolve("encoder.layers.0") is low
            assert config.resolve("decoder.layers.0") is high
    assert ComponentQuantizationConfig.get_min_capability() == 0


def test_component_capability_does_not_query_non_cuda_device():
    platform = Mock(is_cuda=Mock(return_value=False))
    with patch("aphrodite.omni.quantization.component_config.current_platform", platform):
        ComponentQuantizationConfig({"encoder": Mock(get_min_capability=Mock(return_value=90))})
    platform.get_device_capability.assert_not_called()
