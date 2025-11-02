"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support Aphrodite.
"""

from aphrodite.transformers_utils.processors.deepseek_vl2 import (
    DeepseekVLV2Processor)
from aphrodite.transformers_utils.processors.ovis import OvisProcessor
from aphrodite.transformers_utils.processors.ovis2_5 import Ovis2_5Processor

__all__ = ["DeepseekVLV2Processor", "OvisProcessor", "Ovis2_5Processor"]
