# Copyright contributors to the Terratorch project

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class PluginConfig(BaseModel):
    output_path: str = None
    """
    Default output folder path to be used when the out_data_format is set to path. 
    If omitted, the plugin will default to the current user home directory.
    """

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.output_path:
            self.output_path = str(Path.home())
        elif os.path.exists(self.output_path):
            if not os.access(self.output_path, os.W_OK):
                raise ValueError(f"The path: {self.output_path} is not writable")
        else:
            raise ValueError(f"The path: {self.output_path} does not exist")

        return self


class TiledInferenceParameters(BaseModel):
    # Used in Aphrodite plugin
    h_crop: int = 512
    h_stride: int = None
    w_crop: int = 512
    w_stride: int = None
    average_patches: bool = None


class RequestData(BaseModel):
    data_format: Literal["b64_json", "path", "url"]
    """
    Data type for the input image.
    Allowed values are: [`b64_json`, `path`, `url`]
    """

    out_data_format: Literal["b64_json", "path"]
    """
    Data type for the output image.
    Allowed values are: [`b64_json`, `path`]
    """

    data: Any
    """
    Input image data
    """

    indices: list[int] | None = None
    """
    Indices for bands to be processed in the input file
    """


MultiModalPromptType = RequestData


class RequestOutput(BaseModel):
    data_format: Literal["b64_json", "path"]
    """
    Data type for the output image.
    Allowed values are: [`b64_json`, `path`]
    """

    data: Any
    """
    Output image data
    """

    request_id: str | None = None
    """
    The Aphrodite request ID if applicable
    """
