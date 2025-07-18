# Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py

import json
import math
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal, Optional, Union

from loguru import logger

from aphrodite.common.config import LoRAConfig
from aphrodite.common.logger import log_once


@dataclass
class PEFTHelper:
    """ 
    A helper class for PEFT configurations, specifically designed for LoRA.
    This class handles configuration validation, compatibility checks for 
    various LoRA implementations.
    """

    # Required fields
    r: int
    lora_alpha: int
    target_modules: Union[list[str], str]

    bias: Literal["none", "all", "lora_only"] = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)
    # True to use Rank-Stabilized LoRA (rsLoRA, see: https://arxiv.org/abs/2312.03732)
    use_rslora: bool = field(default=False)
    # True to use Weight-Decomposed Low-Rank Adaptation (DoRA, see: https://arxiv.org/abs/2402.09353)
    use_dora: bool = field(default=False)
    # long context lora field
    context_length: int = field(default=0)
    # Extra aphrodite field, start with 'aphrodite_' to avoid conflict
    aphrodite_lora_scaling_factor: float = field(default=1.0)
    aphrodite_max_position_embeddings: Optional[int] = field(default=False)
    aphrodite_long_context_scaling_factor: Optional[float] = field(
        default=None)

    def _validate_features(self) -> tuple[list[str], list[str]]:
        """
        Check if there are any unsupported LoRA features.
        """
        error_msg = []
        warning_msg = []

        if self.modules_to_save:
            warning_msg.append(
                "Aphrodite only supports modules_to_save being None.")
        if self.use_dora:
            error_msg.append("Aphrodite does not yet support DoRA.")
        return error_msg, warning_msg

    def __post_init__(self):
        if self.use_rslora:
            log_once("INFO", "Loading LoRA weights trained with rsLoRA.")
            self.aphrodite_lora_scaling_factor = (self.lora_alpha /
                                                  math.sqrt(self.r))
        else:
            self.aphrodite_lora_scaling_factor = self.lora_alpha / self.r
        if self.context_length:
            if self.aphrodite_max_position_embeddings is None:
                self.aphrodite_max_position_embeddings = self.context_length
            self.aphrodite_long_context_scaling_factor = float(
                math.ceil(self.context_length /
                          self.aphrodite_max_position_embeddings))

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PEFTHelper":
        # Get all field information from the class
        class_fields = {f.name: f for f in fields(cls)}
        # Check for required fields
        required_fields = {
            name
            for name, f in class_fields.items()
            if f.default is MISSING and f.default_factory is MISSING
        }

        # Identify any missing required fields
        missing_fields = required_fields - set(config_dict.keys())
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {missing_fields}")

        # Filter out fields that aren't defined in the class
        filtered_dict = {
            k: v
            for k, v in config_dict.items() if k in class_fields
        }
        return cls(**filtered_dict)

    @classmethod
    def from_local_dir(cls, lora_path: str,
                       max_position_embeddings: Optional[int]) -> "PEFTHelper":
        lora_config_path = os.path.join(lora_path, "adapter_config.json")

        with open(lora_config_path) as f:
            config = json.load(f)
        config["aphrodite_max_position_embeddings"] = max_position_embeddings
        return cls.from_dict(config)

    def validate_legal(self, lora_config: LoRAConfig) -> None:
        """
        Validates the LoRA configuration settings against application 
        constraints and requirements.
        """
        error_msg, warning_msg = self._validate_features()
        if self.r > lora_config.max_lora_rank:
            error_msg.append(
                f"LoRA rank {self.r} is greater than max_lora_rank"
                f" {lora_config.max_lora_rank}.")
        if self.bias != "none" and not lora_config.bias_enabled:
            error_msg.append(
                "Adapter bias cannot be used without bias_enabled.")
        if error_msg:
            raise ValueError(f"{' '.join(error_msg)}")
        if warning_msg:
            logger.warning(
                "Aphrodite LoRA configuration has some unsupported features: "
                "{}", " ".join(warning_msg))
