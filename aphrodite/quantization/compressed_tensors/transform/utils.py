from typing import NamedTuple

from compressed_tensors.transform import TransformArgs, TransformScheme

__all__ = ["TransformTuple"]


class TransformTuple(NamedTuple):
    scheme_name: str
    scheme: TransformScheme
    args: TransformArgs
